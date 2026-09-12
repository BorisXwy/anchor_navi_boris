"""Per-episode failure attribution for round 20260911_235744_opennav100_aligned_actrev.

Deterministic post-run analysis only: reads trajectory.json, navigation graph,
vlm_calls.json, the hidden evaluation_only geometry, and the judge_audit /
stage_completion_verification outputs.  Nothing here feeds back into online
navigation.
"""
import collections
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, "/workspace/anchor_navi_boris/scripts")
from audit_active_stop_round import (  # noqa: E402
    _load_dataset, _polyline, _project_to_path, _forward_reference_heading,
    _heading_error)

ROOT = "/workspace/anchor_navi_boris/outputs/e2e_eval/20260911_235744_opennav100_aligned_actrev"
DATASET = ("/workspace/anchor_navi_boris/data/datasets/opennav100_start_aligned/"
           "val_unseen_opennav100ids_start_aligned.json.gz")
TURN_FORMS = {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND", "TURN_TO_LANDMARK"}
STRICT_REACH_M = 0.75

summary = json.load(open(f"{ROOT}/summary.json"))
results = {r["episode_id"]: r for r in summary["results"]}
dataset = {int(e["episode_id"]): e for e in _load_dataset(DATASET)}

full = list(csv.DictReader(open(f"{ROOT}/judge_audit/judge_audit.csv")))
full_by_ep = collections.defaultdict(list)
for row in full:
    full_by_ep[int(row["episode_id"])].append(row)


def ep_dir(eid):
    return glob.glob(f"{ROOT}/shard_*/episode_{eid:04d}")[0]


def hop_geometry(events, geometry):
    """Per-hop hidden geometry keyed by target_index."""
    selected = {}
    stopped = {}
    for event in events:
        payload = event.get("payload") or {}
        index = payload.get("target_index")
        if index is None:
            continue
        index = int(index)
        if event["event"] == "point_selected":
            selected[index] = event
        elif event["event"] == "point_navigation_stopped":
            stopped[index] = event
    out = {}
    for index in selected:
        sel = selected[index]
        start = np.asarray(sel["position_xyz"], np.float64)
        point = sel.get("selected_point_navmesh_xyz") or sel.get("selected_point_world_xyz")
        item = {"start_progress_m": _project_to_path(start, geometry)["progress_m"],
                "start_dist_to_path_m": _project_to_path(start, geometry)["distance_m"]}
        if point is not None:
            point = np.asarray(point, np.float64)
            horizon = float(np.linalg.norm((point - start)[[0, 2]]))
            ref = _forward_reference_heading(start, geometry, horizon)
            item["selection_heading_error_deg"] = _heading_error(point - start, ref)
            item["selected_point_dist_to_path_m"] = _project_to_path(point, geometry)["distance_m"]
            item["selected_point_progress_delta_m"] = (
                _project_to_path(point, geometry)["progress_m"] - item["start_progress_m"])
            item["selection_horizon_m"] = horizon
        stop = stopped.get(index)
        if stop is not None:
            end = np.asarray(stop["position_xyz"], np.float64)
            proj = _project_to_path(end, geometry)
            item["end_dist_to_path_m"] = proj["distance_m"]
            item["progress_delta_m"] = proj["progress_m"] - item["start_progress_m"]
            item["moved_m"] = float(np.linalg.norm((end - start)[[0, 2]]))
            item["final_target_geodesic_m"] = stop.get("final_target_geodesic_distance_m")
            item["declared_arrival"] = stop["payload"].get("policy_declared_arrival") in (True, "True")
            item["end_reason"] = stop["payload"].get("end_reason")
            horizon = item.get("selection_horizon_m") or item["moved_m"]
            ref = _forward_reference_heading(start, geometry, horizon)
            item["executed_heading_error_deg"] = _heading_error(end - start, ref)
        out[index] = item
    return out


rows = []
for eid, r in sorted(results.items()):
    d = ep_dir(eid)
    t = json.load(open(f"{d}/trajectory.json"))
    dec = json.load(open(f"{d}/instruction_decomposition.json"))
    subs = dec["all_sub_instructions"]
    forms = [s["form"] for s in subs]
    seq = t["instruction_sequence_exploration"] or {}
    state = seq.get("state") or {}
    geom = json.load(open(f"{d}/evaluation_only/evaluation_geometry.json"))
    ref_geometry = _polyline(dataset[eid]["reference_path"])
    hg = hop_geometry(geom["point_events"], ref_geometry)

    hops = []
    for tg in t["targets"]:
        si = tg["sub_instruction"] if isinstance(tg["sub_instruction"], dict) else {}
        ic = tg.get("instruction_completion") or {}
        directive = tg.get("sequence_directive") or {}
        g = hg.get(tg["target_index"], {})
        hops.append({
            "target_index": tg["target_index"],
            "sub_id": si.get("sub_instruction_id"),
            "form": si.get("form"),
            "arrived": bool(tg.get("arrived")),
            "end_reason": tg.get("end_reason"),
            "n_steps": len(tg["steps"]) if isinstance(tg.get("steps"), list) else None,
            "judge": ic.get("status"),
            "judge_reason": (ic.get("reason") or "")[:400],
            "directive": directive.get("action"),
            "blocked_yaw": directive.get("direction_to_block_yaw_rad"),
            "selection_reason": ((tg.get("selection") or {}).get("reason") or "")[:300],
            "allowed_views": (tg.get("selection") or {}).get("allowed_views"),
            "direction_gate": ((tg.get("selection") or {}).get("direction_gate") or {}).get("sector"),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()},
        })

    n_hops = len(hops)
    arrived_hops = [h for h in hops if h["arrived"]]
    physical_fail = [h for h in hops if not h["arrived"]]
    physical_fail_reasons = collections.Counter(h["end_reason"] for h in physical_fail)
    judged = [h for h in hops if h["judge"]]
    n_completed = sum(1 for h in judged if h["judge"] == "completed")
    n_unknown = sum(1 for h in judged if h["judge"] == "unknown")
    completed_ids = sorted({h["sub_id"] for h in judged if h["judge"] == "completed"})
    stage_at_end = state.get("expected_sub_instruction_id", 0)
    if stage_at_end is None or stage_at_end >= len(subs):
        stage_at_end = len(subs) - 1
    form_at_end = forms[stage_at_end] if subs else None
    is_last = stage_at_end == len(subs) - 1
    final_stage_hops = [h for h in hops if h["sub_id"] == stage_at_end]
    final_stage_unknown = sum(1 for h in final_stage_hops if h["judge"] == "unknown")
    final_stage_physical_fail = sum(1 for h in final_stage_hops if not h["arrived"])

    # false arrivals: executor declared arrival but > STRICT_REACH_M from the point
    false_arrivals = [h for h in arrived_hops
                      if h.get("final_target_geodesic_m") is not None
                      and h["final_target_geodesic_m"] > STRICT_REACH_M]
    arrival_dists = [h["final_target_geodesic_m"] for h in arrived_hops
                     if h.get("final_target_geodesic_m") is not None]

    # recovery (action-reversal) records
    recs = seq.get("recovery_records") or []
    rec_summary = []
    for rec in recs:
        att = (rec.get("attempts") or [{}])[-1]
        revisit = att.get("vlm_node_revisit") or {}
        rec_summary.append({
            "trigger": rec.get("trigger"), "success": rec.get("success"),
            "end_reason": rec.get("end_reason"), "method": rec.get("backtrack_method"),
            "same_place": revisit.get("same_place"), "confidence": revisit.get("confidence"),
            "similarity": att.get("target_panorama_similarity_after"),
            "forward_steps": att.get("forward_step_count"),
        })
    blocked = state.get("blocked_yaws_by_verified_node") or {}
    blocked_count = sum(len(v) for v in blocked.values())
    max_blocked_at_node = max([len(v) for v in blocked.values()], default=0)

    # selection heading statistics (all hops with geometry)
    sel_errs = [h["selection_heading_error_deg"] for h in hops if h.get("selection_heading_error_deg") is not None]
    prog = [h["progress_delta_m"] for h in hops if h.get("progress_delta_m") is not None]
    end_dist = [h["end_dist_to_path_m"] for h in hops if h.get("end_dist_to_path_m") is not None]

    ff = full_by_ep.get(eid, [])
    unk_but_verified = sum(1 for c in ff if c["verdict_class"] == "unknown_but_verified")
    unk_confirmed = sum(1 for c in ff if c["verdict_class"] == "unknown_confirmed")
    completed_refuted = sum(1 for c in ff if c["verdict_class"] == "completed_refuted")
    completed_confirmed = sum(1 for c in ff if c["verdict_class"] == "completed_confirmed")
    final_stage_ubv = sum(1 for c in ff if c["verdict_class"] == "unknown_but_verified"
                          and int(c["sub_instruction_id"]) == stage_at_end)
    final_stage_uc = sum(1 for c in ff if c["verdict_class"] == "unknown_confirmed"
                         and int(c["sub_instruction_id"]) == stage_at_end)

    term = r["termination_category"]
    stop_issued = bool(r["stop_action_issued"])
    success = bool(r["simulator_reported_success"])
    if success:
        direct = "success"
    elif stop_issued:
        direct = "wrong_stop"
    elif term == "no_floor_bearing_candidate":
        direct = "no_candidate_after_filter"
    else:
        direct = term

    rows.append(dict(
        id=eid, scene=os.path.basename(os.path.dirname(t["scene"])) if t["scene"].endswith(".glb") else os.path.basename(t["scene"]),
        instruction=dec["instruction"].strip(), n_sub=len(subs), forms=forms,
        direct=direct, termination=term, end_reason=seq.get("end_reason"),
        stop_issued=stop_issued, success=success,
        goal_hit=bool(r["goal_radius_hit_diagnostic"]),
        init_geo=r["initial_geodesic_distance_m"], final_geo=r["final_geodesic_distance_m"],
        control_steps=r["control_steps"],
        n_hops=n_hops, n_arrived=len(arrived_hops), n_physical_fail=len(physical_fail),
        physical_fail_reasons=dict(physical_fail_reasons),
        n_false_arrivals=len(false_arrivals),
        median_arrival_dist=round(statistics.median(arrival_dists), 2) if arrival_dists else None,
        judge_completed=n_completed, judge_unknown=n_unknown,
        completed_stage_ids=completed_ids, stage_at_end=stage_at_end, form_at_end=form_at_end,
        is_last_stage=is_last, final_stage_hops=len(final_stage_hops),
        final_stage_unknown=final_stage_unknown, final_stage_physical_fail=final_stage_physical_fail,
        n_recoveries=len(recs),
        recovery_triggers=dict(collections.Counter(x["trigger"] for x in rec_summary)),
        recovery_success=sum(1 for x in rec_summary if x["success"]),
        recovery_same_place_false=sum(1 for x in rec_summary if x["same_place"] is False),
        recovery=rec_summary,
        blocked_count=blocked_count, max_blocked_at_node=max_blocked_at_node,
        sel_err_median=round(statistics.median(sel_errs), 1) if sel_errs else None,
        sel_err_gt90=sum(1 for e in sel_errs if e > 90), sel_err_n=len(sel_errs),
        prog_nonpos=sum(1 for p in prog if p <= 0), prog_n=len(prog),
        end_dist_gt2=sum(1 for e in end_dist if e > 2), end_dist_n=len(end_dist),
        unk_but_verified=unk_but_verified, unk_confirmed=unk_confirmed,
        completed_refuted=completed_refuted, completed_confirmed=completed_confirmed,
        final_stage_ubv=final_stage_ubv, final_stage_uc=final_stage_uc,
        hops=hops,
    ))

def track_bucket(hop):
    """Coarse hidden-geometry label of one hop: did it move along the GT path?"""
    if hop.get("selection_heading_error_deg") is None or hop.get("progress_delta_m") is None:
        return "na"
    end_dist = hop.get("end_dist_to_path_m") or 0.0
    if (hop["selection_heading_error_deg"] <= 30 and hop["progress_delta_m"] > 0.5
            and end_dist < 1.5):
        return "on_track"
    if (hop["selection_heading_error_deg"] > 60 or hop["progress_delta_m"] <= 0
            or end_dist > 2.5):
        return "off_track"
    return "ambiguous"


# Goal proximity of every hop end and of the whole trace (hidden geometry).
for x in rows:
    d = ep_dir(x["id"])
    t = json.load(open(f"{d}/trajectory.json"))
    geom = json.load(open(f"{d}/evaluation_only/evaluation_geometry.json"))
    goal = np.asarray(t["r2r_goal_positions"][0], np.float64)
    stops = {int(e["payload"]["target_index"]): e for e in geom["point_events"]
             if e["event"] == "point_navigation_stopped"}
    for h in x["hops"]:
        e = stops.get(h["target_index"])
        if e is not None:
            h["dist_goal_at_hop_end_m"] = round(float(np.linalg.norm(
                (np.asarray(e["position_xyz"]) - goal)[[0, 2]])), 2)
        h["track"] = track_bucket(h)
    trace = geom["action_trace"]
    if trace:
        pos = np.asarray([a["position_xyz"] for a in trace], np.float64)
        dxz = np.linalg.norm((pos - goal)[:, [0, 2]], axis=1)
        x["min_goal_xz_m"] = round(float(dxz.min()), 2)
    else:
        x["min_goal_xz_m"] = None
    final_hops = [h for h in x["hops"] if h["sub_id"] == x["stage_at_end"]]
    x["final_stage_on_track_unknown"] = sum(
        1 for h in final_hops if h["track"] == "on_track" and h["judge"] == "unknown")
    x["final_stage_off_track"] = sum(1 for h in final_hops if h["track"] == "off_track")
    x["final_stage_in_goal_radius_hop"] = any(
        h.get("dist_goal_at_hop_end_m") is not None and h["dist_goal_at_hop_end_m"] <= 3.0
        for h in final_hops)


def failure_group(x):
    if x["success"]:
        return "success"
    if x["direct"] == "wrong_stop":
        return "G4_wrong_stop"
    if "recovery_failed" in x["direct"]:
        return "G5_backtrack_rejected"
    if x["form_at_end"] in TURN_FORMS:
        return "G1_turn_gate_empty"
    if x["is_last_stage"] and x["form_at_end"] == "STOP_WAIT":
        return "G2_final_stop_wait_stuck"
    return "G3_mid_stage_no_candidate"


for x in rows:
    x["group"] = failure_group(x)

json.dump(rows, open(f"{ROOT}/analysis_95_failures.json", "w"), ensure_ascii=False, indent=1)

failed = [x for x in rows if not x["success"]]
print("groups:", collections.Counter(x["group"] for x in failed))
print("episodes:", len(rows), "failed:", len(failed))
print("direct:", collections.Counter(x["direct"] for x in failed))
print("form_at_end:", collections.Counter(x["form_at_end"] for x in failed))
print("is_last_stage:", sum(1 for x in failed if x["is_last_stage"]))
print("completed stages dist:", collections.Counter(len(x["completed_stage_ids"]) for x in failed))
print("zero hops:", [x["id"] for x in failed if x["n_hops"] == 0])
print("total hops:", sum(x["n_hops"] for x in rows), "arrived:", sum(x["n_arrived"] for x in rows),
      "physical fail:", sum(x["n_physical_fail"] for x in rows))
pf = collections.Counter()
for x in rows:
    for k, v in x["physical_fail_reasons"].items():
        pf[k] += v
print("physical fail reasons:", pf)
print("false arrivals:", sum(x["n_false_arrivals"] for x in rows), "of arrived", sum(x["n_arrived"] for x in rows))
print("recoveries:", sum(x["n_recoveries"] for x in rows), "success:", sum(x["recovery_success"] for x in rows),
      "same_place false:", sum(x["recovery_same_place_false"] for x in rows))
rt = collections.Counter()
for x in rows:
    for k, v in x["recovery_triggers"].items():
        rt[k] += v
print("recovery triggers:", rt)
print("episodes with recovery:", sum(1 for x in rows if x["n_recoveries"]))
print("blocked total:", sum(x["blocked_count"] for x in rows))
