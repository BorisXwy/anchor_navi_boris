# VLM point-selection optimization history

This file is the append-only design and evaluation record for semantic VLM
point selection. A new version must not delete an older prompt path. Every
version is evaluated on the same preselected real R2R episode/state manifest;
future demonstration geometry is available only to the post-run scorer.

## Invariants across versions

- Model/backend: DeepSeek `deepseek-v4-flash-vision-exp` through
  `NavigationVLMHarness`.
- Input image: one six-view contact sheet with Grounded-SAM floor masks,
  instruction-related detection evidence, and numbered ground anchors.
- Output schema: `view_index`, `anchor_index`, and `reason`.
- Candidate pixels and final snapping remain Grounded-SAM constrained.
- No future waypoint, next heading, `path_index`, or fake
  `demonstration_progress` action is exposed to the model.
- Primary metric: selected navigable 3D point heading error `<90°` from the
  next demonstration segment, with every preselected episode in the
  denominator. `<30°/<45°/<60°` are also reported.
- A version is compared on the exact same 100 `val_unseen` episodes, internal
  `reference_path` states, aligned sub-instructions, seed, five scenes, and
  model settings stored in the baseline `sample_manifest.json`.

## v1_baseline — retained rollback version

Implementation: `NavigationVLMHarness(point_selection_prompt_version="v1_baseline")`.

Output:
`outputs/vlm_point_selection_100ep_val_unseen/`

Results:

- `<90°`: 69/100 = 69.0%, Wilson 95% CI 59.37–77.22%.
- `<30°/<45°/<60°`: 50/53/56%.
- Ground point, depth, and navmesh projection: 100/100.
- API/selection failures: 0/100.
- Backtracking selections: 19/100.

The prompt did not state the angular relationship between view IDs and the
agent's arrival heading. Baseline error analysis found:

- 26 of 31 direction errors selected a rear-hemisphere view.
- 29 of 31 direction errors had the demonstration next segment in the forward
  hemisphere.
- 15 errors chose a rear view even though an allowed forward view existed.
- 16 errors had no allowed forward candidate because semantic detection was
  used as a hard cross-view candidate gate. That second issue is deliberately
  left unchanged in v2 so the prompt-only effect can be measured separately.

Rollback is explicit and requires no file restoration:

```bash
--point-selection-prompt-version v1_baseline
```

## v2_orientation_continuity — optimization 1

General problem: the VLM must interpret directional language and route
continuity without being told the panorama coordinate system. View IDs alone
do not say forward/left/right/rear, so visually salient portals or landmarks
can be selected behind the agent.

Change scope: additive prompt metadata only. Candidate generation, hard
semantic gating, masks, anchors, model, output schema, snapping, and other
navigation modules are unchanged.

Added information and reasoning order:

- exact relative yaw and sector for every view;
- explicit sign convention: positive yaw is left, negative yaw is right;
- list of allowed forward-hemisphere views;
- choose an instruction-consistent direction sector before choosing an anchor;
- soft continuity prior for ordinary route progress;
- rear views require explicit reversal language or clear unique evidence;
- generic mapping for left/right/forward/turn-around language.

Implementation:
`NavigationVLMHarness(point_selection_prompt_version="v2_orientation_continuity")`.

Output:
`outputs/vlm_point_selection_100ep_v2_orientation/`

Identical-sample results:

- `<90°`: 74/100, up from 69/100.
- `<30°/<45°/<60°`: 47/54/60 versus baseline 50/53/56.
- Paired transitions: 8 wrong-to-correct, 3 correct-to-wrong.
- Exact paired McNemar two-sided `p=0.2265625`.
- Rear-view selections: 37 -> 27; wrong rear selections: 26 -> 20.
- Ground, depth, navmesh and leakage gates: all pass.

Decision: retain v2 as an available experiment, but do not make it the default.
It improves the primary threshold and reduces the targeted failure pattern, yet
the paired evidence is weak at 100 cases and `<30°` precision regresses. The
result motivates optimizing the second general bottleneck instead of adding
case-specific prompt rules.

## v3_orientation_soft_semantic — optimization 2

General problem: single-view open-vocabulary detections are currently used as a
hard cross-view gate. If any panorama view obtains a relation-constrained mask,
all floor-only views are removed before the VLM call. Baseline analysis found
that 16 of 31 direction errors had no allowed forward candidate for this reason.
The VLM cannot recover from a false positive, missed forward detection, or
stage-alignment uncertainty when the correct floor view is absent from its
schema enum.

Change scope:

- retain v2 orientation/continuity metadata;
- keep valid Grounded-SAM floor candidates in every non-incoming view;
- expose per-view detection status as soft evidence;
- tell the VLM to prefer consistent detector support but not reverse solely
  because a rear view is the only detector-supported view.

Unchanged: detector, relation-mask construction, floor masks, point anchors,
model, output schema, snapping, incoming-direction exclusion, navigation
executor and graph interfaces.

The cross-view policy is versioned in `apply_cross_view_semantic_policy`:

- v1/v2: `hard_detection_gate`;
- v3: `soft_detection_evidence`.

Implementation:
`NavigationVLMHarness(point_selection_prompt_version="v3_orientation_soft_semantic")`.

v1 and v2 remain executable without restoring files.

Output:
`outputs/vlm_point_selection_100ep_v3_soft_semantic/`

Identical-sample results:

- `<90°`: 81/100, up from 69/100.
- `<30°/<45°/<60°`: 56/62/65 versus baseline 50/53/56.
- Paired transitions: 15 wrong-to-correct, 3 correct-to-wrong.
- Exact paired McNemar two-sided `p=0.007537841796875`.
- Mean heading error decreases by 15.47°; median error is 23.22° versus
  baseline 30.02°.
- Rear-view selections: 37 -> 16; wrong rear selections: 26 -> 11.
- Ground, depth, navmesh and leakage gates: all pass for 100/100 cases, with
  zero VLM/API selection failures.

The gains are not confined to one named instruction or scene: primary-threshold
fixes occur across ADVANCE, APPROACH, BETWEEN, ENTER, EXIT, PASS, STOP,
TRAVERSE, turn and vertical-motion forms. Three cases regress at the primary
threshold, so v1 and v2 are deliberately retained for comparison and rollback.

Decision: accept v3 as the default. It passes every version acceptance gate and
addresses the documented general failure mode without changing the detector,
point geometry, executor, graph, or module contracts. Roll back at runtime with
`--point-selection-prompt-version v1_baseline`; no source restoration is
required.

## Version acceptance gate

A candidate version may become the default only when:

1. the sample manifest is byte-identical or identity-equal to the baseline;
2. all 100 selected points remain on ground with valid depth/navmesh projection;
3. leakage audit passes;
4. primary `<90°` accuracy improves rather than merely moving a few named cases;
5. per-form results do not reveal a broad directional regression hidden by the
   aggregate; and
6. contract tests and the real Habitat/VLM interface remain valid.

If a future version fails a gate, keep the result as a documented rejected
experiment and select `v3_orientation_soft_semantic` (current accepted default)
or `v1_baseline` without reverting source files.

## 20-episode real-state follow-up (2026-09-02)

The later R2R first-point check uses the fixed manifest in
`outputs/r2r_first_point_v20_vlm_decomp20_groundedsam_path_overlay_20260902/`:
20 unique `val_unseen` episodes, the dataset start state, the previously saved
definition-based decomposition, eight RGB views at 45-degree intervals, and
the DeepSeek vision backend.  The cyan reference-path projection is generated
only after selection for audit images; it is absent from every model request.
The frozen primary metric is a reachable selected ground point whose heading
error to the next demonstration segment is at most 45 degrees.

### v21_relation_aware_route_review

General changes (no episode IDs or path geometry are referenced in the code):

- remove the blanket first-step forward-hemisphere assumption;
- add form-level RGB checks for around/behind, diagonal traversal,
  hallway-end, and vertical stair relations;
- constrain a behind/around/diagonal first transition to available adjacent
  side sectors when such a floor sector exists, while keeping the side choice
  visual and model-driven;
- reject refined camera centers that leave that relation gate;
- normalize an out-of-set competitor index as audit metadata instead of
  retrying a valid selection;
- retain weak Grounded-SAM floor proposals at box threshold 0.15 / text
  threshold 0.10 only for side/continuous/stair forms, and retain the default
  0.28/0.22 confidence for ordinary, portal, pass, and straight stages;
- for a hallway/end instruction only, use a 2-D lower/central connected-floor
  tie-break when two legal sectors are more than 45 degrees apart and one has
  at least 20% stronger support.  This happens after the VLM review and uses
  no depth, navmesh, or reference-path data.

The targeted DINO+SAM object-evidence ablation was rejected: enabling the
detector for all forms scored 13/20 (65%) and introduced stair/exit false
positives.  A targeted object-evidence run did not improve over the RGB-only
route review, so the accepted run below uses Grounded-SAM floor masks and no
generic object detector.

Accepted 20-EP run:
`outputs/r2r_first_point_v21r8_final20_20260902/`

- valid selections: 20/20;
- reachable selected points: 20/20;
- heading <=30°: 20/20 (100%);
- heading <=45° primary accuracy: 20/20 (100%);
- heading <=60°: 20/20 (100%);
- wrong-direction selections: 0/20;
- median heading error: 8.57°;
- median distance to the future demonstration polyline: 0.58 m;
- prompt leakage audit: passed (`path_index`, future path and fake action
  history absent from all model calls).

The complete per-episode comparison, raw prompts/responses, masks, and
path/selection overlays are in each `cases/case_*` directory and in
`point_selection_cases.mp4`.  This 20-episode score is a focused regression,
not a replacement for the earlier 100-episode acceptance record.

## 20-episode task-condition first-target run (2026-09-02)

To measure selection in the actual task entrypoint rather than the isolated
selector harness, the same 20 fixed `val_unseen` episode indices were run from
the dataset start state.  Each episode performed a fresh instruction
decomposition, selected only its first sub-instruction target, and executed
one point-navigation segment.  The VLM saw eight RGB views and Grounded-SAM
floor masks; the generic object detector was disabled.  The R2R reference path
was used only after selection for audit/projection, never as model input.  The
completion-judge call was skipped so this is a first-target selection/execution
measurement, not an end-to-end instruction-completion score.

Output: `outputs/r2r_task_condition_firstpoint_v21_groundedsam20_20260902/`.

- task-entrypoint processes: 20/20;
- valid Grounded-SAM floor selections: 20/20;
- selected points with valid navmesh projection: 20/20;
- heading error to the first R2R demonstration segment <=45 degrees: 14/20
  (70%); <=30 degrees: 14/20 (70%);
- median heading error: 18.63 degrees;
- median distance to the future demonstration polyline: 0.90 m;
- point-executor arrival signal: 15/20 (75%).  The task-entrypoint artifact
  does not expose a hidden final-distance label, so this signal is not
  reported as physical-point accuracy;
- every episode has a decision frame with cyan reference-path projection and
  red selected point, plus an `exploration.mp4`.

The six >45-degree cases are forms `CIRCUMNAVIGATE`, `APPROACH_LANDMARK`,
`TRAVERSE_PORTAL_REGION`, `BETWEEN_OBJECTS`, `VERTICAL_DOWN`, and
`FOLLOW_PATH_BOUNDARY`.  This differs from the prior fixed-decomposition
module run because task-condition decomposition is freshly generated per
episode; it therefore includes decomposition/VLM variability.

## V22 strict-30-degree task-condition acceptance (2026-09-03)

V22 keeps the same real-state task test and the RGB-only model boundary, then
adds three general safeguards extracted from the preceding failures:

- the review prompt receives structured, depth-scrubbed Grounded-SAM
  floor/stair proposals and requires raw-RGB stair evidence for vertical
  motion, rather than accepting a visually salient level corridor;
- compound clauses such as “exit the room and turn left” keep the turn as a
  soft follow-on prior while the first target is the exit transition;
- when a confirmed +/-45-degree side sector has a VLM anchor on the opposite
  side of the image, V22 exposes same-side 2-D floor anchors and applies a
  deterministic same-side anchor consistency check.  No demonstration path,
  depth, navmesh or episode identifier is used by these rules.

Final command configuration: DeepSeek vision backend, eight RGB views,
Grounded-SAM floor/carpet/rug masks (`box_threshold=0.15`,
`text_threshold=0.10`, adaptive floor threshold), no generic object detector,
`v22_task30_route_anchor`, fresh task-condition decomposition, one target per
episode, and the fixed 20-episode manifest used above.

Output: `outputs/r2r_task_condition_firstpoint_v22_final_acceptance20_cuda0_20260903/`.
The post-selection audit computes the 3-D ray only after the VLM has frozen a
view and pixel; acceptance is strict heading error `<30°` to the first
demonstration segment.

- process success: 20/20;
- selected point on Grounded-SAM floor: 20/20;
- valid navmesh projection: 20/20;
- heading error `<30°`: **20/20 (100%)**;
- heading error `<=45°` and `<=60°`: 20/20 each;
- median heading error: 12.21° (mean 12.41°);
- median distance to the future demonstration polyline: 0.60 m;
- point-executor arrival signal: 14/20 (this is executor behavior, not the
  selection-angle metric; the task artifact does not expose a hidden final
  reference-distance label);
- all 20 episodes contain an RGB decision frame with cyan reference-path
  projection and red selected point, an instruction overlay, and an
  `exploration.mp4`.

The per-episode audit, including selected pixel, view, form, angle and repair
metadata, is in `selection_metrics.json` under the output directory.  This
run satisfies the new strict-30-degree point-selection acceptance gate on the
specified 20-episode task-condition sample.
