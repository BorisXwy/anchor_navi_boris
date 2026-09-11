import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_judge_round import (
    categorize_reason,
    full_fidelity_rows,
    parse_judge_prompt,
    raw_judge_call_rows,
    turn_form_without_turn_commands,
)


def _judge_prompt(sub_instruction, action_summary):
    return (
        "You are the RGB-only edge completion judge for an R2R robot.\n\n"
        "ACTIVE SUB-INSTRUCTION:\n"
        + json.dumps(sub_instruction, indent=2)
        + "\n\nFOLLOWING SUB-INSTRUCTION:\nnull\n\n"
        "RGB-only commanded action history:\n"
        + json.dumps(action_summary, indent=2)
        + "\n\nRGB detector evidence at PREVIOUS node:\n{}\n")


class AnalyzeJudgeRoundTest(unittest.TestCase):
    def test_categorize_reason_matches_each_category(self):
        cases = {
            "missing_turn_evidence": (
                "The action history shows 12 forward moves and 0 right-turn "
                "commands."),
            "wrong_turn_direction": (
                "The commanded history shows a right turn followed by forward "
                "moves, but the active sub-instruction requires turning left."),
            "crossed_into_next_stage": (
                "The agent appears to have already executed the next "
                "instruction by entering the hallway."),
            "reversed_or_stationary": (
                "The edge is stationary with no net progress."),
            "not_there_yet": (
                "The current panorama still shows the bedroom interior and "
                "the doorway has not been crossed."),
            "landmark_absent_or_wrong_place": (
                "No railing or balcony is visible; the agent is in a bedroom "
                "rather than the hallway."),
            "insufficient_evidence": (
                "The evidence is ambiguous and the relation is not confirmed."),
            "uncategorized": "Something entirely different happened.",
        }
        for expected, text in cases.items():
            self.assertEqual(categorize_reason(text), expected, text)

    def test_turn_form_check_is_direction_specific(self):
        self.assertTrue(turn_form_without_turn_commands(
            "TURN_RIGHT", {"left_turn_command_count": 3,
                           "right_turn_command_count": 0}))
        self.assertFalse(turn_form_without_turn_commands(
            "TURN_RIGHT", {"right_turn_command_count": 2}))
        self.assertTrue(turn_form_without_turn_commands("TURN_AROUND", {}))
        self.assertFalse(turn_form_without_turn_commands(
            "PASS_LANDMARK", {"left_turn_command_count": 0}))

    def test_parse_judge_prompt_recovers_form_and_turn_counts(self):
        prompt = _judge_prompt(
            {"sub_instruction_id": 1, "form": "TURN_RIGHT",
             "navigation_instruction": "Turn right",
             "point_selection_strategy": {"nested": {"deep": [1, 2]}}},
            {"control_steps": 5, "forward_command_count": 3,
             "left_turn_command_count": 0, "right_turn_command_count": 2,
             "chronological_actions": [{"step": 0, "action": "turn_right"}]})
        parsed = parse_judge_prompt(prompt)
        self.assertEqual(parsed["sub_instruction"]["form"], "TURN_RIGHT")
        self.assertEqual(parsed["action_summary"]["right_turn_command_count"], 2)
        self.assertEqual(parse_judge_prompt("no markers here"),
                         {"sub_instruction": {}, "action_summary": {}})

    def test_raw_rows_cover_crashed_episodes_without_trajectory(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "shard_0" / "episode_0042"
            episode.mkdir(parents=True)
            (episode / "vlm_calls.json").write_text(json.dumps([
                {"task": "select_ground_target", "result": {}},
                {"task": "judge_edge_instruction_completion_rgb_only",
                 "prompt": _judge_prompt(
                     {"sub_instruction_id": 0, "form": "TURN_LEFT"},
                     {"forward_command_count": 4,
                      "left_turn_command_count": 0,
                      "right_turn_command_count": 0}),
                 "result": {"status": "unknown", "confidence": 0.6,
                            "reason": "only forward commands were issued"},
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            ]))
            rows = raw_judge_call_rows(root)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["episode_id"], 42)
        self.assertTrue(row["crashed_episode"])
        self.assertEqual(row["form"], "TURN_LEFT")
        self.assertTrue(row["turn_form_without_turn_commands"])
        self.assertEqual(row["reason_category"], "missing_turn_evidence")

    def test_full_fidelity_rows_flag_stop_blocking_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "shard_0" / "episode_0007"
            (episode / "evaluation_only").mkdir(parents=True)
            (episode / "trajectory.json").write_text(json.dumps({
                "episode_id": 7,
                "config": {"episode_index": 0},
                "reference_state": {"position_xyz": [0.0, 0.0, 0.0]},
                "r2r_goal_positions": [[0.0, 0.0, 5.0]],
                "instruction_stages": [
                    {"sub_instruction_id": 0, "form": "ADVANCE_STRAIGHT"},
                    {"sub_instruction_id": 1, "form": "STOP_WAIT"}],
                "targets": [{
                    "target_index": 0,
                    "policy_input_contract": "rgb_only_v1",
                    "point_target_arrived": True,
                    "node_created_after_point_arrival": True,
                    "navigation_graph_node_id": "node_0001",
                    "navigation_graph_edge_id": "edge_0000",
                    "sub_instruction": {"sub_instruction_id": 1,
                                        "form": "STOP_WAIT"},
                    "instruction_completion": {
                        "status": "unknown", "instruction_completed": False,
                        "expected_sub_instruction_id": 1, "confidence": 0.6,
                        "reason": "the doorway has not been reached",
                        "motion_evidence": {"forward_command_count": 4,
                                            "left_turn_command_count": 0,
                                            "right_turn_command_count": 0}},
                }],
            }))
            (episode / "evaluation_only" / "evaluation_geometry.json").write_text(
                json.dumps({"point_events": [
                    {"event": "point_selected", "payload": {"target_index": 0},
                     "position_xyz": [0.0, 0.0, 0.0],
                     "selected_point_navmesh_xyz": [0.0, 0.0, 4.5]},
                    {"event": "point_navigation_stopped",
                     "payload": {"target_index": 0,
                                 "policy_declared_arrival": True},
                     "position_xyz": [0.0, 0.0, 4.0],
                     "final_target_geodesic_distance_m": 0.5},
                ]}))
            dataset = [{"episode_id": 7, "reference_path": [
                [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]}]
            rows = full_fidelity_rows(root, dataset)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["is_last_stage"])
        self.assertAlmostEqual(row["distance_to_goal_xz_m"], 1.0)
        self.assertTrue(row["stop_blocking_unknown"])
        self.assertTrue(row["geometry_gate_passed"])
        self.assertEqual(row["online_reason_category"], "not_there_yet")
        self.assertEqual(row["verdict_class"], "not_verified")
        self.assertFalse(row["independent_available"])


if __name__ == "__main__":
    unittest.main()
