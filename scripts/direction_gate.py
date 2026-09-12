"""Explicit turn-sentence direction gate: shared vocabulary and fallback plan.

``vlm_harness.select_ground_target`` raises :class:`DirectionGateEmptyError`
when an explicit ``turn left/right/around`` clause has no floor-bearing view
inside its commanded sector.  ``rgb_only_instruction_sequence`` catches it and
may rotate in place to the sector centre once per sub-instruction.  This
module keeps that contract free of any other project import so the harness
and the strategy never depend on each other.
"""

TURN_GATE_CENTER_DEG = {"left": 90.0, "right": -90.0, "rear": 180.0}
IN_PLACE_TURN_PHASE = "direction_gate_in_place_turn"


class DirectionGateEmptyError(RuntimeError):
    """No floor-bearing candidate survives an explicit turn direction gate.

    ``sector`` is ``left``/``right``/``rear`` for the commanded side gate, or
    ``forward`` when the post-in-place-turn forward gate is empty as well; the
    strategy only attempts the in-place turn for the first group.  The default
    message is the historical RuntimeError text so evaluator substring
    matching (``no_floor_bearing_candidate``) is unchanged.
    """

    def __init__(self, sector, message=None):
        self.sector = str(sector)
        super().__init__(message or (
            "No floor-bearing candidate remains inside the explicit "
            f"{self.sector} direction gate"))


def in_place_turn_step_count(sector, turn_step_deg):
    """Signed number of discrete turn commands to face ``sector``'s centre."""
    if sector not in TURN_GATE_CENTER_DEG:
        raise ValueError(f"unknown turn-gate sector {sector!r}")
    turns = TURN_GATE_CENTER_DEG[sector] / float(turn_step_deg)
    if abs(turns - round(turns)) > 1e-6:
        raise ValueError(
            "direction-gate in-place turn needs "
            f"{TURN_GATE_CENTER_DEG[sector]} degrees to be an integer multiple "
            f"of the turn step, got turn_step_deg={turn_step_deg} for "
            f"sector={sector!r}")
    return int(round(turns))


def plan_in_place_turn(sector, turn_step_deg):
    """Action records that rotate in place to ``sector``'s centre.

    Same record shape as ``rgb_only_instruction_sequence.plan_action_reversal``
    so the plan can be stored as an edge action history.  Physical left turns
    increase yaw, so ``left`` and ``rear`` use ``turn_left``.
    """
    turn_step_deg = float(turn_step_deg)
    signed_steps = in_place_turn_step_count(sector, turn_step_deg)
    action = "turn_left" if signed_steps >= 0 else "turn_right"
    commanded_deg = turn_step_deg if signed_steps >= 0 else -turn_step_deg
    return [{
        "step": index,
        "action": action,
        "commanded_turn_deg": commanded_deg,
        "forward_commanded": False,
        "orientation_only": True,
        "phase": IN_PLACE_TURN_PHASE,
        "policy_input_contract": "rgb_only_v1",
    } for index in range(abs(signed_steps))]
