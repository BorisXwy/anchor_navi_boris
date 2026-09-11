import csv
import json
import collections
import glob
import os
import re

ROOT = "/workspace/anchor_navi_boris/outputs/e2e_eval/20260911_001436_e2e_opennav"
summary = json.load(open(f"{ROOT}/summary.json"))
results = {r["episode_id"]: r for r in summary["results"]}

raw_calls = list(csv.DictReader(open(f"{ROOT}/judge_audit/judge_audit_raw_calls.csv")))
full = list(csv.DictReader(open(f"{ROOT}/judge_audit/judge_audit.csv")))
calls_by_ep = collections.defaultdict(list)
for c in raw_calls:
    calls_by_ep[int(c["episode_id"])].append(c)
full_by_ep = collections.defaultdict(list)
for c in full:
    full_by_ep[int(c["episode_id"])].append(c)

TURN_FORMS = {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND", "TURN_TO_LANDMARK"}

def ep_dir(eid):
    m = glob.glob(f"{ROOT}/shard_*/episode_{eid:04d}")
    return m[0]

def crash_type(d):
    log = open(f"{d}/process.log", errors="replace").read().strip().splitlines()
    last = log[-1] if log else ""
    if "No floor-bearing candidate can be sent" in last:
        return "crash_no_candidate"
    m = re.search(r"explicit (\w+) direction gate", last)
    if m:
        return f"crash_gate_{m.group(1)}"
    if "exhausted retries" in last:
        return "crash_vlm_parse"
    return "crash_other:" + last[:80]

rows = []
for eid, r in sorted(results.items()):
    if r["simulator_reported_success"]:
        continue
    d = ep_dir(eid)
    dec = json.load(open(f"{d}/instruction_decomposition.json"))
    subs = dec["all_sub_instructions"]
    forms = [s["form"] for s in subs]
    graph = json.load(open(f"{d}/navigation_graph/navigation_graph.json"))
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    steps = sum(e.get("control_step_count", 0) for e in graph["edges"])
    calls = calls_by_ep.get(eid, [])
    n_completed = sum(1 for c in calls if c["status"] == "completed")
    n_unknown = sum(1 for c in calls if c["status"] == "unknown")
    # highest sub_instruction reached
    stage_reached = max([int(c["sub_instruction_id"]) for c in calls], default=0)
    # stage active at end: if the last judgment was completed, the next stage began
    if calls and calls[-1]["status"] == "completed":
        stage_at_end = min(int(calls[-1]["sub_instruction_id"]) + 1, len(subs) - 1)
    else:
        stage_at_end = stage_reached
    # stages completed online = distinct sub_instruction ids with completed
    completed_ids = sorted({int(c["sub_instruction_id"]) for c in calls if c["status"] == "completed"})
    form_at_end = forms[stage_at_end] if subs else None
    unk_cats = collections.Counter(c["reason_category"] for c in calls if c["status"] == "unknown")
    turn_unknown_no_cmd = sum(1 for c in calls if c["status"] == "unknown" and c["turn_form_without_turn_commands"] == "True")
    turn_unknown = sum(1 for c in calls if c["status"] == "unknown" and c["is_turn_form"] == "True")
    crashed = r["termination_category"] == "episode_process_failed"
    if crashed:
        direct = crash_type(d)
    elif r["stop_action_issued"]:
        direct = "wrong_stop"
    else:
        direct = r["termination_category"]
    # full fidelity info (non crashed)
    ff = full_by_ep.get(eid, [])
    unk_but_verified = sum(1 for c in ff if c["verdict_class"] == "unknown_but_verified")
    unk_confirmed = sum(1 for c in ff if c["verdict_class"] == "unknown_confirmed")
    completed_refuted = sum(1 for c in ff if c["verdict_class"] == "completed_refuted")
    geom_gate = sum(1 for c in ff if c["geometry_gate_passed"] == "True")
    max_steps_hops = sum(1 for c in ff if c["end_reason"] and "arrival" not in c["end_reason"])
    stop_blocking = sum(1 for c in ff if c["stop_blocking_unknown"] == "True")
    is_last = (stage_at_end == len(subs) - 1)
    zero_step = steps == 0

    # root cause attribution (ordered rules)
    root = []
    if direct == "crash_vlm_parse":
        root.append("R6_vlm_json_parse")
    if direct.startswith("crash_gate") or (direct == "crash_no_candidate" and form_at_end in TURN_FORMS):
        if zero_step:
            root.append("R2_turn_gate_empty_at_start")
        else:
            if turn_unknown_no_cmd > 0:
                root.append("R1_turn_cmds_invisible_to_judge")
            root.append("R2_turn_gate_empty")
    elif direct == "crash_no_candidate":
        if turn_unknown_no_cmd > 0:
            root.append("R1_turn_cmds_invisible_to_judge")
        root.append("R3_judge_unknown_streak_exhausts_directions")
    elif direct == "all_candidate_directions_blocked_at_verified_node":
        if turn_unknown_no_cmd > 0:
            root.append("R1_turn_cmds_invisible_to_judge")
        if unk_but_verified > 0:
            root.append("R3a_judge_too_conservative_verified")
        if unk_confirmed > 0:
            root.append("R3b_not_actually_there_select_or_walk")
        if max_steps_hops > 0:
            root.append("R4_walk_layer_max_steps")
    elif direct == "wrong_stop":
        root.append("R5_judge_false_positive_chain")
        if turn_unknown_no_cmd > 0:
            root.append("R1_turn_cmds_invisible_to_judge")
    if not root:
        root.append("unattributed")
    if crashed:
        root.append("R0_uncaught_exception")

    rows.append(dict(
        id=eid, scene=os.path.basename(dec.get("scene", "") or ""),
        instruction=dec["instruction"].strip(), n_sub=len(subs), forms=forms,
        direct=direct, crashed=crashed, zero_step=zero_step,
        nodes=n_nodes, edges=n_edges, steps=steps,
        judge_completed=n_completed, judge_unknown=n_unknown,
        completed_stage_ids=completed_ids, stage_at_end=stage_at_end, form_at_end=form_at_end,
        is_last_stage=is_last, unknown_categories=dict(unk_cats),
        turn_unknown=turn_unknown, turn_unknown_no_cmd=turn_unknown_no_cmd,
        init_geo=r["initial_geodesic_distance_m"], final_geo=r["final_geodesic_distance_m"],
        blocked=r["blocked_direction_count"], backtracks=r["sequence_recovery_backtracks"],
        unk_but_verified=unk_but_verified, unk_confirmed=unk_confirmed, completed_refuted=completed_refuted,
        geom_gate=geom_gate, max_steps_hops=max_steps_hops, stop_blocking=stop_blocking,
        root=root,
    ))

json.dump(rows, open(f"{ROOT}/analysis_94_failures.json", "w"), ensure_ascii=False, indent=1)
print("failed episodes:", len(rows))
print("direct:", collections.Counter(x["direct"] for x in rows))
print("crashed:", sum(x["crashed"] for x in rows))
print("zero_step:", [x["id"] for x in rows if x["zero_step"]])
print("form_at_end:", collections.Counter(x["form_at_end"] for x in rows))
print("form_at_end turn:", sum(1 for x in rows if x["form_at_end"] in TURN_FORMS))
print("stage_at_end:", collections.Counter(x["stage_at_end"] for x in rows))
print("zero completed stages:", sum(1 for x in rows if not x["completed_stage_ids"]))
print("is_last_stage at end:", [x["id"] for x in rows if x["is_last_stage"]])
print("last-stage & STOP_WAIT:", [x["id"] for x in rows if x["is_last_stage"] and x["form_at_end"] == "STOP_WAIT"])
rc = collections.Counter()
for x in rows:
    for t in x["root"]:
        rc[t] += 1
print("root:", rc)
print("any turn_unknown_no_cmd>0:", sum(1 for x in rows if x["turn_unknown_no_cmd"] > 0))
print("episodes with any turn form:", sum(1 for x in rows if set(x["forms"]) & TURN_FORMS))
print("first form is turn:", sum(1 for x in rows if x["forms"] and x["forms"][0] in TURN_FORMS), [x["id"] for x in rows if x["forms"] and x["forms"][0] in TURN_FORMS and x["stage_at_end"] == 0])
cat = collections.Counter()
for x in rows:
    for k, v in x["unknown_categories"].items():
        cat[k] += v
print("unknown categories:", cat)
# non-crashed detail
nc = [x for x in rows if not x["crashed"]]
print("non-crashed:", len(nc))
for x in nc:
    print(x["id"], x["direct"], f"{x['init_geo']:.1f}->{x['final_geo']:.1f}", "completed", x["completed_stage_ids"], "/", x["n_sub"], "ubv", x["unk_but_verified"], "uc", x["unk_confirmed"], "gate", x["geom_gate"], "maxsteps", x["max_steps_hops"], "turn_no_cmd", x["turn_unknown_no_cmd"], x["root"])
# crashed grouped
print("\n--- crashed by direct/form ---")
g = collections.defaultdict(list)
for x in rows:
    if x["crashed"]:
        g[(x["direct"], x["form_at_end"])].append(x["id"])
for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
    print(k, len(v), v)
print("\n--- progress before crash ---")
print(collections.Counter(len(x["completed_stage_ids"]) for x in rows if x["crashed"]))
print("unknown streak before no_candidate crash:", collections.Counter(x["judge_unknown"] - 0 for x in rows if x["direct"] == "crash_no_candidate"))
# instruction length / distance
import statistics
print("n_sub dist failed:", collections.Counter(x["n_sub"] for x in rows))
print("init geo nc:", statistics.median([x["init_geo"] for x in nc]))
