# Eight-view instruction completion (`v8_eight_view_spatial_relations`)

This prompt profile adds a completion-only Habitat panorama while preserving
the existing six-view point-selection, graph embedding, and backtracking APIs.

The 4x2 VLM sheet is labeled relative to the agent heading:

| View | Sector | Relative yaw |
|---:|---|---:|
| 0 | front | 0° |
| 1 | front-left | +45° |
| 2 | left | +90° |
| 3 | rear-left | +135° |
| 4 | rear | 180° |
| 5 | rear-right | -135° |
| 6 | right | -90° |
| 7 | front-right | -45° |

## Form-specific evidence

| Forms | Completion evidence |
|---|---|
| `PASS_LANDMARK` | The same landmark instance moving from front/side to view 3, 4, or 5 after forward progress is sufficient pass evidence; it need not disappear from every view. Multiple similar instances force `unknown`. |
| `CIRCUMNAVIGATE` | Obstacle moves front -> instructed side -> rear, and the route beyond it is clear. |
| `EXIT_REGION` | Source/exit frame moves behind; destination space dominates front and side sectors. |
| `ENTER_REGION` | Entry frame/source moves behind; destination surrounds the camera. |
| `SELECT_PORTAL`, `TRAVERSE_PORTAL_REGION` | Correct portal identity plus near-side -> frame -> beyond-side transition; frame ends behind. |
| `CROSS_SPACE` | Near boundary/entry context moves behind and the far-side context surrounds the node. |
| `BETWEEN_OBJECTS` | Pair brackets opposite sides in the gap; for “clear/pass between,” both shift side-rear/rear. |
| `TURN_LEFT`, `TURN_RIGHT` | Previous instructed side becomes current front, supported by the action yaw history. |
| `TURN_AROUND` | Previous rear becomes current front and previous front becomes current rear. |
| `ADVANCE_STRAIGHT` | Forward progression with little net rotation; earlier front content moves toward side/rear. |
| `FOLLOW_PATH_BOUNDARY` | Boundary remains on the instructed side while progress reaches the requested cue. |
| `APPROACH_LANDMARK` | Landmark becomes nearer/larger in front sectors; rear-only visibility indicates overshoot. |
| `STOP_WAIT` | Requested near/beside relation and safe distance hold; rear visibility alone is insufficient. |
| `VERTICAL_UP`, `VERTICAL_DOWN` | Combine signed height change, chronological stair keyframes, and landing/stairs distribution across front/rear sectors. |
| `OTHER` | Infer the explicit relation and require a before/after change; rear evidence is used only for pass/cross/clear/exit semantics. |

Every v8 response includes `directional_evidence`: previous/current reference
sectors and exact view indices, same-instance and multiple-instance checks,
whether a current rear sector supports completion, whether a front sector
contradicts completion, and whether the temporal relation changed. A sector is
counted from the landmark center, not overlapping edge pixels from an adjacent
90-degree camera. For `PASS_LANDMARK`, the final status also checks the existing
Grounded-SAM six-view detections: a strong target-class detection that still
dominates the front over the rear changes a VLM `completed` claim to `unknown`
because same-instance passage is ambiguous.
