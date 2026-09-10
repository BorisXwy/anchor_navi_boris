"""R2R task-level STOP semantics shared by runtime and evaluators.

Habitat-Sim exposes locomotion actions, while the R2R task's STOP is a
task-level terminal action. We record the explicit STOP request that ends
navigation and score simulator success only at that request.
"""


def build_stop_action_record(
        *, sequence_completed_in_order, complete_instruction_was_evaluated,
        global_step, position_xyz):
    """Return the task STOP emitted after the final stage completes."""
    issued = bool(
        sequence_completed_in_order and complete_instruction_was_evaluated)
    return {
        "action": "STOP" if issued else None,
        "issued": issued,
        "issued_after_final_sub_instruction_completion": issued,
        "global_step": int(global_step) if issued else None,
        "position_xyz": (
            [float(value) for value in position_xyz] if issued else None),
        "semantics": (
            "Habitat R2R task-level terminal action; success is evaluated at "
            "this pose and no locomotion action follows"),
    }


def simulator_stop_success(*, stop_action_issued, goal_radius_hit):
    """Habitat-style success requires an active STOP inside the goal radius."""
    return bool(stop_action_issued and goal_radius_hit)
