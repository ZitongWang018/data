from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


PROMPT_HEADER = """You are playing ALFWorld, a text household environment.
Choose exactly one valid action from the provided admissible actions.
Respond with only one line formatted as:
Action: <valid action>
"""


@dataclass
class GenerationResult:
    text: str
    action: str


class LocalCausalPolicy:
    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 96,
        history_window: int = 3,
        prompt_mode: str = "react_fewshot",
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.history_window = history_window
        self.prompt_mode = prompt_mode
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
        self.fewshot_prompts = load_alfworld_prompts() if prompt_mode == "react_fewshot" else None

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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stopping = newline_stopping_criteria(self.tokenizer, inputs["input_ids"].shape[1])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if self.prompt_mode == "react_fewshot":
            action = parse_react_line(generated)
        else:
            action = parse_action(generated, admissible_actions)
        return GenerationResult(text=generated, action=action)


def build_policy_prompt(
    *,
    prompt_mode: str,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    history_window: int,
    admissible_actions: Sequence[str],
    initial_observation: str | None,
    gamefile: str | None,
    fewshot_prompts: dict[str, str] | None,
) -> str:
    if prompt_mode == "react_fewshot":
        if initial_observation is None or gamefile is None or fewshot_prompts is None:
            raise ValueError("react_fewshot requires initial_observation, gamefile, and prompt assets")
        return build_paper_react_prompt(
            initial_observation=initial_observation,
            current_observation=observation,
            history=history,
            gamefile=gamefile,
            prompts=fewshot_prompts,
        )
    if prompt_mode != "admissible":
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return build_react_prompt(
        task=task,
        observation=observation,
        history=history,
        history_window=history_window,
        admissible_actions=admissible_actions,
    )


def load_alfworld_prompts() -> dict[str, str]:
    path = Path(__file__).parent / "assets" / "alfworld_react_prompts.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing ReAct prompt asset: {path}")
    prompts = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in prompts.items() if key.startswith("react_")}


def task_prompt_prefix(gamefile: str) -> str:
    task_name = Path(gamefile).parent.parent.name
    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }
    for task_type, prefix in prefixes.items():
        if task_name.startswith(task_type):
            return prefix
    raise ValueError(f"Unsupported ALFWorld task path: {gamefile}")


def clean_initial_observation(observation: str) -> str:
    chunks = observation.split("\n\n")
    return "\n".join(chunks[1:]).strip() if len(chunks) > 1 else observation.strip()


def build_paper_react_prompt(
    *,
    initial_observation: str,
    current_observation: str,
    history: Sequence[tuple[str, str]],
    gamefile: str,
    prompts: dict[str, str],
) -> str:
    prefix = task_prompt_prefix(gamefile)
    demonstrations = prompts[f"react_{prefix}_1"] + prompts[f"react_{prefix}_0"]
    prompt = (
        "Interact with a household to solve a task. Here are two examples.\n"
        f"{demonstrations}\nHere is the task.\n"
        f"{clean_initial_observation(initial_observation)}\n>"
    )
    for index, (_obs_before, action) in enumerate(history):
        next_observation = history[index + 1][0] if index + 1 < len(history) else current_observation
        prompt += f" {action}\n{next_observation}\n>"
    return prompt


class StopAfterTokenSequence(StoppingCriteria):
    def __init__(self, prompt_length: int, sequences: Sequence[Sequence[int]]) -> None:
        self.prompt_length = prompt_length
        self.sequences = [list(sequence) for sequence in sequences if sequence]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: object) -> bool:
        generated = input_ids[0, self.prompt_length :].tolist()
        return any(len(generated) >= len(seq) and generated[-len(seq) :] == seq for seq in self.sequences)


def newline_stopping_criteria(tokenizer: object, prompt_length: int) -> StoppingCriteriaList:
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in ("\n", "\r\n")]
    return StoppingCriteriaList([StopAfterTokenSequence(prompt_length, sequences)])


def parse_react_line(text: str) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "look")
    line = line.lstrip("> ").strip().strip("`'\"")
    line = re.sub(r"^(?:action|act)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.rstrip(". ") or "look"


def build_react_prompt(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    history_window: int = 3,
    admissible_actions: Sequence[str],
) -> str:
    history_text = "\n".join(
        f"Observation: {obs}\nAction: {action}" for obs, action in history[-history_window:]
    )
    actions = "\n".join(f"- {action}" for action in admissible_actions)
    return (
        f"{PROMPT_HEADER}\n"
        f"Task: {task}\n\n"
        f"{history_text}\n\n"
        f"Current observation: {observation}\n\n"
        f"Admissible actions:\n{actions}\n\n"
        "Action:"
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
