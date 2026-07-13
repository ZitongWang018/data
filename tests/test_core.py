from __future__ import annotations

import unittest
from pathlib import Path

from agentic_ttt.alfworld_env import order_game_files
from agentic_ttt.llm_policy import (
    build_author_admissible_prompt,
    build_paper_react_prompt,
    build_zero_shot_react_prompt,
    parse_action,
    parse_react_line,
    task_prompt_prefix,
)
from agentic_ttt.repetition import compute_sequence_weights
from scripts.run_attt_alfworld import is_progress_transition, reached_repeated_action_stop


class CoreLogicTests(unittest.TestCase):
    def test_interleaved_order_avoids_task_family_prefix(self) -> None:
        files = [
            f"/tmp/{family}-item-{index}/trial/game.tw-pddl"
            for family in ("look_at_obj_in_light", "pick_and_place")
            for index in range(3)
        ]
        ordered = order_game_files(files, order="interleaved", seed=0)
        families = [Path(item).parents[1].name.split("-item-")[0] for item in ordered]
        self.assertEqual(families[:4], ["look_at_obj_in_light", "pick_and_place"] * 2)

    def test_paper_prompt_has_two_examples_without_action_leak(self) -> None:
        prompts = {"react_put_0": "example zero\n", "react_put_1": "example one\n"}
        prompt = build_paper_react_prompt(
            initial_observation="header\n\nroom\n\nYour task is to: put apple on table.",
            current_observation="room",
            history=[],
            gamefile="/tmp/pick_and_place_simple-Apple-None-Table-1/trial/game.tw-pddl",
            prompts=prompts,
        )
        self.assertIn("example one\nexample zero", prompt)
        self.assertNotIn("Admissible actions", prompt)
        self.assertTrue(prompt.endswith("\n>"))

    def test_task_type_mapping(self) -> None:
        gamefile = "/tmp/pick_cool_then_place_in_recep-Pan-None-Stove-1/trial/game.tw-pddl"
        self.assertEqual(task_prompt_prefix(gamefile), "cool")

    def test_zero_shot_prompt_has_grammar_without_demonstrations(self) -> None:
        prompt = build_zero_shot_react_prompt(
            initial_observation="header\n\nroom\n\nYour task is to: put apple on table.",
            current_observation="room",
            history=[],
        )
        self.assertIn("Use `think: ...`", prompt)
        self.assertIn("put ... in/on ...", prompt)
        self.assertNotIn("Here are two examples", prompt)
        self.assertNotIn("Admissible actions", prompt)

    def test_author_prompt_matches_admissible_action_protocol(self) -> None:
        prompt = build_author_admissible_prompt(
            task="Your task is to: put apple on table.",
            observation="You are in a room.",
            history=[],
            admissible_actions=["go to table 1", "look"],
        )
        self.assertIn("Thought: <your reasoning>", prompt)
        self.assertIn("Action: <one action copied EXACTLY", prompt)
        self.assertIn("Observation: You are in a room.", prompt)
        self.assertIn("Admissible actions: ['go to table 1', 'look']", prompt)
        self.assertTrue(prompt.endswith("Thought:"))

    def test_author_action_parser_reads_two_line_output(self) -> None:
        text = "Thought: I should inspect the table.\nAction: go to table 1"
        self.assertEqual(parse_action(text, ["go to table 1", "look"]), "go to table 1")

    def test_react_line_does_not_snap_to_admissible_action(self) -> None:
        self.assertEqual(parse_react_line("Action: open drawer  to open drawer 3\n"), "open drawer  to open drawer 3")
        self.assertEqual(parse_react_line("> think: inspect the desk\n"), "think: inspect the desk")

    def test_repeat_action_stop_after_three_identical_actions(self) -> None:
        history = [("obs0", "look"), ("obs1", "look")]
        self.assertTrue(reached_repeated_action_stop(history, "look", threshold=3))
        self.assertFalse(reached_repeated_action_stop(history, "inventory", threshold=3))

    def test_token_exposure_matches_paper_formula(self) -> None:
        weighting = compute_sequence_weights([1, 2, 3, 4], [1, 2, 3, 1, 2, 3], ngram_size=3)
        self.assertEqual(weighting.repeated_positions, [0, 1, 2])
        self.assertEqual(weighting.weights, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0])

    def test_progress_transition_rejects_thoughts_and_invalid_actions(self) -> None:
        common = {
            "admissible": ["go to desk 1", "look"],
            "previous_observation": "You are in a room.",
            "next_observation": "On the desk, you see a key.",
        }
        self.assertFalse(is_progress_transition(action="think: inspect", is_thought=True, **common))
        self.assertFalse(is_progress_transition(action="go to desk 2", is_thought=False, **common))
        self.assertTrue(is_progress_transition(action="go to desk 1", is_thought=False, **common))

    def test_progress_transition_rejects_noop_observations(self) -> None:
        self.assertFalse(
            is_progress_transition(
                action="look",
                admissible=["look"],
                previous_observation="room",
                next_observation="room",
                is_thought=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
