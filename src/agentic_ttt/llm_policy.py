from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_HEADER = """You are playing ALFWorld, a text household environment.
Choose exactly one valid action from the provided admissible actions.
Respond with a short thought and then one line formatted as:
Action: <valid action>
"""


@dataclass
class GenerationResult:
    text: str
    action: str


class LocalCausalPolicy:
    def __init__(self, model_path: str, *, max_new_tokens: int = 64) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

    def generate_action(
        self,
        *,
        task: str,
        observation: str,
        history: Sequence[tuple[str, str]],
        admissible_actions: Sequence[str],
    ) -> GenerationResult:
        prompt = build_react_prompt(
            task=task,
            observation=observation,
            history=history,
            admissible_actions=admissible_actions,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return GenerationResult(text=generated, action=parse_action(generated, admissible_actions))


def build_react_prompt(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_actions: Sequence[str],
) -> str:
    history_text = "\n".join(
        f"Observation: {obs}\nAction: {action}" for obs, action in history[-6:]
    )
    actions = "\n".join(f"- {action}" for action in admissible_actions)
    return (
        f"{PROMPT_HEADER}\n"
        f"Task: {task}\n\n"
        f"{history_text}\n\n"
        f"Current observation: {observation}\n\n"
        f"Admissible actions:\n{actions}\n\n"
        "Thought:"
    )


def parse_action(text: str, admissible_actions: Sequence[str]) -> str:
    lowered = text.lower()
    match = re.search(r"action\s*:\s*(.+)", text, flags=re.IGNORECASE)
    candidates = [match.group(1).strip()] if match else []
    candidates.append(text.strip())

    for candidate in candidates:
        cleaned = candidate.splitlines()[0].strip().strip("`'\". ")
        for action in admissible_actions:
            if cleaned.lower() == action.lower():
                return action

    for action in admissible_actions:
        if action.lower() in lowered:
            return action

    if "look" in admissible_actions:
        return "look"
    return admissible_actions[0]

