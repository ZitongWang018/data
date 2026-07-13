from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_policy import (
    GenerationResult,
    build_policy_prompt,
    load_alfworld_prompts,
    newline_stopping_criteria,
    parse_action,
    parse_react_line,
    resolve_device_map,
)
from .repetition import compute_sequence_weights


@dataclass
class ATrainConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 5e-4
    gradient_steps: int = 2
    ngram_size: int = 3
    min_weight: float = 0.05
    adaptive: bool = False
    use_token_reweighting: bool = True
    update_history: list[str] = field(default_factory=list)
    update_token_history: list[int] = field(default_factory=list)


class TrainableCausalPolicy:
    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 2048,
        history_window: int = 3,
        train_config: ATrainConfig | None = None,
        prompt_mode: str = "react_fewshot",
        use_chat_template: bool = False,
        device_map: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.history_window = history_window
        self.train_config = train_config or ATrainConfig()
        self.prompt_mode = prompt_mode
        self.use_chat_template = use_chat_template
        self.device_map = resolve_device_map(device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        lora_config = LoraConfig(
            r=self.train_config.lora_rank,
            lora_alpha=self.train_config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.train()
        self._initial_adapter_state = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.optimizer = self._build_optimizer()
        self.fewshot_prompts = load_alfworld_prompts() if prompt_mode == "react_fewshot" else None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.train_config.learning_rate)

    def reset_episode(self) -> None:
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in self._initial_adapter_state:
                    parameter.copy_(self._initial_adapter_state[name])
        self.train_config.update_history.clear()
        self.train_config.update_token_history.clear()
        self.optimizer = self._build_optimizer()
        self.model.train()

    def generate_action(
        self,
        *,
        task: str,
        observation: str,
        history: Sequence[tuple[str, str]],
        admissible_actions: Sequence[str],
        initial_observation: str | None = None,
        gamefile: str | None = None,
    ) -> GenerationResult:
        self.model.eval()
        prompt = build_policy_prompt(
            prompt_mode=self.prompt_mode,
            task=task,
            observation=observation,
            history=history,
            history_window=self.history_window,
            admissible_actions=admissible_actions,
            initial_observation=initial_observation,
            gamefile=gamefile,
            fewshot_prompts=self.fewshot_prompts,
        )
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stopping = None if self.prompt_mode == "author_admissible" else newline_stopping_criteria(
            self.tokenizer, inputs["input_ids"].shape[1]
        )
        generate_kwargs = {}
        if stopping is not None:
            generate_kwargs["stopping_criteria"] = stopping
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
        generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if self.prompt_mode in {"react_fewshot", "react_zero_shot"}:
            action = parse_react_line(generated)
        else:
            action = parse_action(generated, admissible_actions)
        return GenerationResult(text=generated, action=action)

    def update_on_text(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0

        current_tokens = input_ids[0].detach().cpu().tolist()
        if self.train_config.use_token_reweighting:
            weighting = compute_sequence_weights(
                current_tokens,
                self.train_config.update_token_history,
                ngram_size=self.train_config.ngram_size,
                min_weight=self.train_config.min_weight,
                adaptive=self.train_config.adaptive,
            )
            weights = weighting.weights
        else:
            weights = [1.0 for _ in current_tokens]
        token_weights = torch.tensor(weights, dtype=torch.float32, device=self.model.device)

        final_loss = 0.0
        self.model.train()
        for _ in range(self.train_config.gradient_steps):
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(**inputs)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view_as(labels)
            shifted_weights = token_weights[1:].to(dtype=logits.dtype)
            loss = (loss_per_token * shifted_weights.unsqueeze(0)).mean()
            loss.backward()
            self.optimizer.step()
            final_loss = float(loss.detach().cpu())

        self.train_config.update_history.append(text)
        self.train_config.update_token_history.extend(current_tokens)
        return final_loss
