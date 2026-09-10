# Lightweight VisionNav workspace

> **Current online contract:** production R2R navigation is strictly RGB-only.
> Selectors, point navigation/arrival, graph memory, completion judgment and
> backtracking receive no pose, depth, navmesh/pathfinder, geodesic distance,
> collision feedback, goal coordinates or reference path. Habitat geometry is
> isolated in the evaluator for initialization, video rendering and post-run
> validation only. See section 0 of [project_rulle.md](project_rulle.md).

> 所有单点系统实验必须遵守 [project_rulle.md](project_rulle.md)：从真实
> R2R 示范轨迹状态初始化，并完整验证选点、导航、节点、指令判定和物理回溯。

One Python 3.10 environment now contains point tracking, ground segmentation,
goal-image navigation, Habitat-Lab, and a source-built headless Habitat-Sim.

## Included projects

| Component | Checkout/model | Interface |
|---|---|---|
| Point tracking | `models/co-tracker` | CoTracker 3 online sliding window |
| Point tracking | `models/tapnet` | causal BootsTAPIR, frame by frame |
| Ground segmentation | OneFormer + Mask2Former + SegFormer ADE20K | pixelwise 2/3 majority floor/rug/carpet/stairs mask |
| Open-vocabulary perception | replaceable Grounded-SAM or DINO+SAM | RGB text query -> 2-D boxes and instance masks |
| Goal-image navigation | `models/visualnav-transformer` | GNM, ViNT, and NoMaD checkpoints |
| Diffusion module | `models/diffusion_policy` | NoMaD ConditionalUnet1D dependency |
| Simulator | `models/habitat-lab`, `models/habitat-sim` | headless RGB observations and navigation |

The default ground path is RGB-only dense semantic segmentation. OneFormer,
Mask2Former, and SegFormer predict ADE20K support surfaces independently and a
pixel is selectable only with at least 2/3 agreement. Grounded-SAM and
DINO+SAM remain replaceable open-vocabulary detectors for instruction objects
and explicit floor ablations; their object masks are not merged into the
production ground mask.

The active environment is `.venv` (Python 3.10, CUDA 12.1 PyTorch 2.2.2),
managed with `uv` and `requirements-unified.txt`. The previous Python 3.11
point-tracker environment is retained as `.venv.py311-pointtrack` for rollback.

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/point_tracker
source .venv/bin/activate
```

Recreate/update it with `bash scripts/setup_unified_env.sh`. Habitat-Sim is
compiled headlessly into the same venv; Bullet physics is disabled because the
current navigation adapter only needs RGB rendering and pathfinding.

## Ground segmentation

```bash
HF_HOME=weights/huggingface .venv/bin/python scripts/segment_ground.py \
  some_rgb.png --output outputs/ground.png --device cuda:0
```

The active navigation backend is `dense-majority`: `[OneFormer | Mask2Former |
SegFormer]` ADE20K predictions are reduced to a strict 2/3 support-surface
mask. Point-selection masks may only remove pixels from this result, and all
VLM/navigation/stop anchors are snapped and revalidated inside it. The RGB
pixel goes directly to the point executor. Hidden depth/navmesh labels may be
computed only after the choice by the evaluator and cannot repair, reject or
redirect the online decision.

## Goal-image policies

All policies use the same CLI contract: observation history plus exactly one
goal image produces distance and waypoint trajectories. Passing one obs image
is supported for cold start by repeating it to fill the model context.

```bash
.venv/bin/python scripts/image_goal_policy.py --model gnm \
  --obs current.png --goal goal.png --output outputs/gnm.json
.venv/bin/python scripts/image_goal_policy.py --model vint \
  --obs current.png --goal goal.png --output outputs/vint.json
.venv/bin/python scripts/image_goal_policy.py --model nomad \
  --obs current.png --goal goal.png --samples 8 --output outputs/nomad.json
```

GNM and ViNT output one deterministic 5-waypoint trajectory. NoMaD outputs a
configurable set of sampled 8-waypoint trajectories. For normal streaming use,
pass 6 obs frames to GNM/ViNT and 4 to NoMaD instead of using cold start.

## Habitat image-goal adapter

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet .venv/bin/python \
  scripts/habitat_imagegoal_demo.py --device cuda:0
```

This loads a real Habitat `.glb`, samples two navigable positions, renders the
current observation and goal image, and runs GNM, ViNT, and NoMaD on the pair.
Results are under `outputs/habitat_imagegoal/`. The released policies were
trained on ground-robot camera data, not Habitat, so the demo validates the
adapter and tensor contract; benchmark-quality closed-loop performance requires
camera calibration, waypoint scaling, collision checking, and domain adaptation.

## Cloud VLM configuration (DMXAPI relay)

Semantic navigation uses the DeepSeek multimodal model
`deepseek-v4-flash-vision-exp` through the DMXAPI OpenAI-compatible relay
(`https://www.dmxapi.cn/v1/chat/completions`), the same route as Navi-Agent's
Final Method. Copy `local_env.sh.example` to the ignored `local_env.sh`
(`chmod 600`), fill in the key, and `source local_env.sh` before running:

```bash
export NAVI_LLM_BACKEND="openrouter"          # historical name, routes to DMXAPI
export NAVI_OPENROUTER_MODEL="deepseek-v4-flash-vision-exp"
export OPENROUTER_API_KEY="your_dmxapi_key_here"
export OPENROUTER_API_URL="https://www.dmxapi.cn/v1/chat/completions"
export NAVI_LLM_DISABLE_THINKING=1            # vision-exp thinks by default (~60 s/call)
export NAVI_OPENROUTER_DISABLE_PROXY=1        # ignore inherited http(s)_proxy
```

`DeepSeekBackend` resolves every setting as explicit argument > environment >
`.env.deepseek` (legacy `DEEPSEEK_*` names, optional) > a direct scan of the
`export` lines in `local_env.sh` > built-in default, so Python entry points
also work when the shell was not sourced. Requests carry
`Authorization: Bearer`, `response_format={"type":"json_object"}` and
`thinking={"type":"disabled"}`; HTTP 408/429/5xx and timeouts are retried with
exponential backoff (`LLM_HTTP_RETRY_BASE_S`, `LLM_HTTP_RETRY_CAP_S`), while
401/402/403 abort the run as `VLMProviderFatalError`. Placeholder keys
(`your_dmxapi_key_here`) are rejected before any network call. Validate the
configuration with one small request:

```bash
python scripts/check_deepseek_vlm.py
```

Each navigation request sends an OpenAI-compatible user content array containing
one text block followed by JPEG `image_url` data-URL blocks. The response is
validated against the task schema, and `vlm_calls.json` records `usage`,
`finish_reason` and a `call_meta` block (endpoint, credential source, attempts,
HTTP status, elapsed time) per call. The CLI backend name stays
`--vlm-backend deepseek`. Ollama remains available via
`--vlm-backend ollama --vlm-model llama3.2-vision:latest`.

## Five-target point-guided random exploration

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet HF_HOME=weights/huggingface \
  .venv/bin/python scripts/random_exploration.py \
  --episode-index 0 --policy gnm \
  --max-steps-per-target 32 --device cuda:0 \
  --tracking-cluster-profile rgb_only_dense_stop_v1 \
  --instruction-completion-prompt-version v8_eight_view_spatial_relations \
  --vlm-backend deepseek --vlm-model deepseek-v4-flash-vision-exp \
  --output-dir outputs/r2r_ep1_deepseek_stages
```

The controller implements this state machine:

1. Load an R2R episode and start at its exact MP3D scene pose and heading.
2. Use the replaceable VLM harness (DeepSeek by default) to decompose the R2R
   instruction into ordered, visually grounded stages.
3. For each stage, retain six or eight Habitat RGB views for
   selection/backtracking and instruction-completion analysis. No navigation
   module receives the paired depth sensors maintained by the evaluator.
   A separate eight-view RGB panorama is captured at
   45-degree intervals for instruction-completion analysis. Grounded-SAM (Grounding-DINO Swin-T + SAM ViT-H) segments floor, wooden/tile floor,
   carpet, rug, walkable ground, stair tread and landing into one selectable mask.
   Generic object segmentation is disabled. Grounding DINO also detects
   instruction landmarks/portals and SAM converts boxes to 2-D instance masks.
4. Use commanded turn/action history and visually stored panorama sectors to
   discourage reversals. Simulator pose and true incoming bearings are absent.
5. Send one labeled six-view contact sheet with green valid-floor overlays and
   colored semantic instance masks/boxes to the VLM, together with structured
   label, confidence and 2-D box/mask evidence (depth is stripped). The harness validates its JSON
   and snaps the requested pixel to the floor mask before handing that external
   selection to the point-navigation executor. There is no depth projection or
   navmesh reachability repair. Use `--semantic-detector none`
   for an ablation. The accepted default point-selection policy is
   `v3_orientation_soft_semantic`: panorama orientation and route continuity
   are explicit, while cross-view detector hits are soft evidence rather than a
   hard gate that can erase a valid forward-floor candidate. The unchanged
   baseline remains directly selectable with
   `--point-selection-prompt-version v1_baseline`; see
   `docs/vlm_point_selection_optimization.md` for the paired 100-case record.
   For the focused 20-episode initial-state regression, the optional
   `v21_relation_aware_route_review` adds generic behind/around/diagonal,
   hallway-end and stair routing checks. The tested configuration uses
   Grounded-SAM floor proposals at `--floor-box-threshold 0.15
   --floor-text-threshold 0.10` with `--adaptive-floor-threshold` (weak masks
   only for side/continuous/stair forms; ordinary/pass/portal stages retain
   the 0.28/0.22 confidence). It uses no generic object detector and never
   exposes the hidden reference path to the model. Reproducible artifacts are
   under `outputs/r2r_first_point_v21r8_final20_20260902/`.
6. VLM is called once per instruction stage, not once per control step. The
   selected ground anchor first defines a crop whose top midpoint is its
   reflection across the horizontal image axis. The vertical crop has at least
   one-third observation height and is warped to the active policy input ratio
   (GNM/ViNT 85x64; NoMaD 96x96). After the crop exists, initialize three roles:
   a 3x3 central navigation cluster, a 3x3 legacy bottom goal/crop cluster, and
   a 5x9 dense bottom arrival cluster. All points snap to the selectable ground
   mask. The arrival cluster has independent causal TAPIR state (sharing model
   weights only), so its 45 queries cannot perturb controller tracking.
7. Fuse the navigation-cluster bearing (80%) with the GNM/ViNT/NoMaD waypoint
   bearing (20%), while the legacy goal cluster anchors the changing crop.
   Then issue only discrete `turn_left`, `turn_right` or `move_forward` actions.
8. `PointNavigationExecutor` alone owns physical arrival. The RGB-only rule
   requires at least half of the dense bottom cluster to disappear for three
   frames plus commanded-forward and RGB ego-motion guards. It reads neither
   moved distance nor initial depth. After arrival,
   the graph must persist a node, its six new views, and an edge containing all
   actions plus five chronological RGB keyframes.
9. `NodeTransitionInstructionCompletionJudge` consumes the previous node,
   current node, both six-view semantics/poses, incoming actions, keyframes, and
   active sub-instruction. It returns only `completed` or `unknown`. Unknown does
   not mean on-route or off-route and is left to an outer exploration strategy.

The VLM backend is isolated behind `VLMBackend`; another local or remote model
can replace DeepSeek without changing the controller. `vlm_calls.json` contains
validated decomposition/selection records. Outputs also include the annotated MP4,
one changing goal crop per control step, six-view strips for every replan, and a
complete `trajectory.json`. Step length, turn angle, optional stage cap, R2R
split/episode, VLM backend/model, and navigation policy are configurable CLI options.

Every MP4 frame uses one fixed 960x480 three-panel layout: the annotated live
observation fills the 640x480 left panel, the current instruction is rendered as
white text on the black 320x240 upper-right panel, and a live 320x240 top-down
navmesh trajectory fills the lower-right panel. Decision observations show the
six view IDs, floor masks, the cyan hidden-reference-path projection, and the red
VLM-selected point; navigation observations show the same diagnostic path
projection together with TAPIR tracks, goal crop, visible-point count and action.
The reference path is rendered only after model selection and is never sent to a
VLM, detector, tracker, or navigation policy.

## Modular pipeline layout

The runnable Habitat pipeline is split by responsibility:

The latest isolated real-state arrival/completion iterations and held-out
results are in
[`docs/point_arrival_and_instruction_completion_iterations.md`](docs/point_arrival_and_instruction_completion_iterations.md).

- `scripts/instruction_decomposer.py`: `InstructionDecomposer` converts the
  route instruction into ordered `SubInstruction` objects. The DeepSeek/Ollama
  harness is injected, while deterministic R2R form definitions add typed
  selection metadata without changing the VLM's spatial grounding.
- `scripts/point_selectors.py`: the external selection contract,
  `RandomExplorationPointSelector`, and `InstructionVLMPointSelector`. It owns
  candidate views, Grounded-SAM masks, novelty/frontier state, or instruction
  VLM calls, but it never executes robot motion.
- `scripts/point_navigation_executor.py`: `PointNavigationExecutor`, crop and
  three-role TAPIR clusters, low-level image-goal control, motion, and the explicit
  `point_navigation_arrived` return signal. It contains no selection policy.
- `scripts/path_projection.py`: evaluation-only projection/drawing utility for
  overlaying a hidden R2R reference path and the selected pixel on RGB frames;
  it is deliberately outside every model-facing prompt.
- `scripts/habitat_point_navigation.py`: `HabitatPointNavigationInterface` and
  `run_habitat_episode()`. This is the main R2R/Habitat adapter that wires one
  selector to the executor and owns episode setup, sensors, logging, metrics,
  and video output.
- `scripts/navigation_graph_memory.py`: persists an initial root and one node
  after every point-navigation executor stop. Successful stops are marked with
  `point_navigation_arrived`; failed/max-step stops remain available as graph
  evidence but cannot complete a sub-instruction. A node stores the incoming
  purpose (`SubInstruction`), position, six RGB views, DINO+SAM environment
  semantics, and a replaceable visual embedding. Each inter-node edge stores
  the executor's complete action history and chronological keyframes.
- `scripts/instruction_completion_judge.py`: binary previous-node -> edge ->
  current-node completion judgment. The replaceable VLM harness sees both
  panoramas, DINO+SAM semantics, node geometry, actions, and keyframes. It
  never returns route membership. `sub_instruction_node_matcher.py` remains a
  legacy diagnostic and is not used to declare semantic completion.
- `scripts/node_backtracking.py`: resolves an ancestor route such as
  `node_0003 -> node_0002 -> node_0001`, compares each prior node's stored six
  views with the live current six views, selects a Grounded-SAM floor point,
  and calls `PointNavigationExecutor` for each reverse hop. Every attempt is
  appended as a `node_backtrack_attempt` edge with its action history; loop
  closure requires both positional proximity and panorama similarity.
- `scripts/instruction_sequence_exploration.py`: a separate outer strategy
  which decomposes the route, calls the VLM for every forward/recovery-hop
  ground point, executes point navigation, stores the stopped node, and runs the
  binary completion judge after arrival. How an `unknown` node is treated as
  progress, a wrong branch, another exploration attempt, or recovery remains
  policy-level behavior and is not part of completion-module accuracy.
- `scripts/evaluate_point_navigation.py`: repeatable multi-episode tests. Use
  `--num-episodes N --start-episode K`, or override both with
  `--episodes 0,3,8`.

`scripts/random_exploration.py` is now only an eight-line backward-compatible
CLI forwarding to the Habitat interface.

Run all deterministic per-module contract tests with:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

After a real one-point Habitat run, audit every runtime artifact and module
handoff in that same trajectory with:

```bash
.venv/bin/python scripts/validate_single_point_run.py outputs/deepseek_smoke_ep0_14
```

The audit writes `module_validation.json` beside the trajectory. Forward-only,
backtracking, sequence-recovery, and pure-exploration branches are mutually
exclusive, so the latter three are covered by isolated deterministic tests.

Each episode writes the full graph to
`<output-dir>/navigation_graph/navigation_graph.json`; the corresponding node
panoramas are under `navigation_graph/nodes/node_XXXX/`. The compact episode
summary in `trajectory.json` includes graph counts and the path to that file.

```bash
# One episode through the main Habitat interface.
.venv/bin/python scripts/habitat_point_navigation.py \
  --mode pure-exploration --episode-index 0

# Five consecutive episodes, or choose another count with --num-episodes.
.venv/bin/python scripts/evaluate_point_navigation.py \
  --mode pure-exploration --num-episodes 5 --start-episode 0

# Execute one forward target, then use the two nodes' six views to go back.
.venv/bin/python scripts/habitat_point_navigation.py \
  --mode pure-exploration --targets 1 --max-steps-per-target 20 \
  --backtrack-target-node previous --backtrack-selector hybrid
```

`--backtrack-target-node` accepts `previous` or an ancestor ID such as
`node_0000`. The controller walks long routes predecessor by predecessor rather
than attempting one unsafe direct jump. `--backtrack-selector hybrid` combines
graph bearing, panorama similarity and floor viability. `vlm` combines two
six-view contact sheets (stored target on top, current floor candidates below)
into one VLM image for single-image local models; `auto` uses VLM when the
semantic harness exists and otherwise uses the hybrid selector.

The strict instruction-sequence exploration strategy is enabled separately:

```bash
.venv/bin/python scripts/habitat_point_navigation.py \
  --mode semantic \
  --exploration-strategy instruction-sequence-recovery \
  --vlm-backend deepseek \
  --vlm-model deepseek-v4-flash-vision-exp \
  --sequence-max-exploration-hops 30
```

The module boundary inside that strategy is fixed:

```text
VLM point -> point executor arrival -> persist node/edge/keyframes
          -> binary edge completion judge
  completed -> advance to the next sub-instruction
  unknown   -> return uncertainty to the outer policy
```

The existing recovery state machine is an outer experimental policy. Its
decision to explore or backtrack after `unknown` is not an output or accuracy
claim of the completion module. Successful backtracking still writes a visual
loop-closure edge.

## End-to-end evaluation launcher

`run_e2e_eval.sh` (thin wrapper over `scripts/run_end_to_end_eval.py`) mirrors
the usage of VLN-CE-master's `run_final_method_eval.sh`: one episode, an
explicit id list, the full OpenNav100 set, parallel shards and a preflight-only
dry run.

```bash
bash run_e2e_eval.sh 7                       # one OpenNav episode id
bash run_e2e_eval.sh 7,11,13 --workers 2     # explicit ids, two parallel shards
bash run_e2e_eval.sh --all --workers 2       # all 100 OpenNav ids
bash run_e2e_eval.sh --episode-indices 0,3,6,9,18,27,45,126,204,219   # ten-EP protocol
bash run_e2e_eval.sh --dry-run 7             # preflight only, nothing launched
bash run_e2e_eval.sh --list                  # ids with dataset index and scene
bash run_e2e_eval.sh --resume outputs/e2e_eval/<round>   # retry unfinished episodes
# smoke without VLM cost (heuristic backend is a rule stub, test only)
bash run_e2e_eval.sh 7 --vlm-backend heuristic --targets 1 \
  --max-steps-per-target 12 --sequence-max-exploration-hops 1 --run-tag smoke
```

- `--all` and positional ids are the 100 OpenNav R2R-CE episode ids frozen in
  `data/opennav100_episode_ids.json`; poses always come from the official
  `val_unseen.json.gz` (the OpenNav release's `start_rotation` is wrong and
  any `OpenNav_R2R-CE_100_bertidx*` dataset path is refused). Episode
  directories and the three-panel video are named by dataset `episode_id`
  (`shard_N/episode_0007/episode_0007.mp4` for id 7, which is `val_unseen`
  index 6); the dataset index stays in `trajectory.json` `config.episode_index`
  and is what `audit_active_stop_round.py`, `verify_round_stage_completions.py`
  and the round merge read, so rounds from before 2026-09-10 (index-named
  `episode_XXXX/exploration.mp4`) still audit correctly.
- Preflight checks dataset/scene/weight presence, CUDA availability, free GPU
  memory (`--gpu-memory-per-worker-gb`, default 11 GB measured per worker) and
  the DMXAPI credential without any network call; failures exit before
  anything is created.
- Each worker is one sequential `evaluate_point_navigation.py` process in
  `<round>/shard_N/`, so scoring, the RGB-only contract audit and the provider
  interruption guard are unchanged. Unknown options are forwarded to it.
- The round directory holds `manifest.json` (selection, config, git commit,
  shard assignment, return codes), `process.log`, `summary.json` (shard
  summaries merged through `evaluate_point_navigation.summarize_results`, with
  `process_failed_episode_indices`, `unscored_episode_indices` and
  `not_run_episode_indices` kept explicit) and `shard_N/{shard.log,
  run_manifest.json, summary.json, episode_<id>/}`.
- No audit runs automatically; the closing log prints the
  `audit_active_stop_round.py` / `verify_round_stage_completions.py` commands.

## Point navigation executor API

All post-selection control is consolidated in
`scripts/point_navigation_executor.py`. `PointNavigationExecutor` deliberately
contains no VLM, panorama ranking, semantic grounding, or exploration-frontier
logic. An external strategy selects a point and supplies its observation mask;
the executor owns crop construction, the three TAPIR cluster roles, image-goal
policy control, Habitat motion, visualization, and arrival detection.

```python
from scripts.point_navigation_executor import (
    POINT_NAVIGATION_ARRIVED,
    PointNavigationExecutor,
    PointNavigationRequest,
    execute_point_navigation,
)

executor = PointNavigationExecutor(
    sim=sim, tracker=tracker, policy=policy, policy_config=policy_cfg,
    policy_name="gnm", predict_fn=predict, device="cuda:0",
)
request = PointNavigationRequest(
    rgb=current_rgb,
    selected_point_xy=point_from_external_strategy,
    selectable_mask=floor_mask_for_that_view,
    yaw=current_yaw,
    position_history=sim_position_history,
    instruction=current_instruction,
)
result = execute_point_navigation(executor, request)
if result.signal == POINT_NAVIGATION_ARRIVED:
    # Persist an arrival node/edge, then run the separate completion judge.
    select_the_next_point()
```

`result.arrived` is the boolean convenience form. On failure it is `False`,
`result.signal` is `None`, and `result.end_reason` distinguishes max steps,
navigation-cluster loss, and crop failure. The complete per-step audit is in
`result.record`.

## Pure point-tracker exploration

The semantic instruction/VLM path remains the default `--mode semantic`.
`--mode pure-exploration` is a separate branch that never constructs or calls a
VLM and ignores the R2R instruction. It records every Habitat simulator XYZ
position, projects multiple DINO+SAM ground candidates from all six RGB-D
views onto the navmesh, and ranks reachable candidates by endpoint distance
from position history plus low path overlap. The initial ground anchor defines
the crop, its central cluster supplies the navigation bearing, its legacy goal
cluster updates the crop, and its independent 45-point bottom cluster supplies
arrival evidence. The v7 temporal/terminal consensus and motion guard decide
arrival before the six-view selection repeats. Failed frontiers are
blacklisted. At local dead ends, the controller follows currently visible
ground back toward remembered unvisited junctions; a remembered frontier is
blacklisted only after a configurable number of unsuccessful transit segments.
Exploration ends when neither a local nor remembered reachable ground candidate
remains at least the configured novelty radius from history.

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet HF_HOME=weights/huggingface \
  .venv/bin/python scripts/random_exploration.py \
  --mode pure-exploration --episode-index 0 --policy gnm \
  --max-exploration-targets 120 --max-steps-per-target 30 \
  --exploration-novelty-radius 1.0 \
  --max-frontier-transit-attempts 3 \
  --output-dir outputs/pure_exploration
```

Outputs include the annotated video, every six-view frontier decision, changing
goal crops, complete simulator position history, per-candidate novelty/path
overlap, blocked frontiers, a navmesh top-down trajectory, swept-area estimate,
and the explicit termination reason in `trajectory.json`. Long runs also update
`exploration_checkpoint.json` after every segment.

Run a reproducible R2R subset and aggregate Success/SPL with:

```bash
.venv/bin/python scripts/evaluate_r2r_vlm.py \
  --episodes 0,1,2,3,4 --max-steps-per-target 20 \
  --output-root outputs/r2r_vlm_eval_0000_0004
```

## Point tracking

```bash
PYTHONPATH=models/tapnet .venv/bin/python scripts/track_cluster.py \
  --backend cotracker --video input.mp4
PYTHONPATH=models/tapnet .venv/bin/python scripts/track_cluster.py \
  --backend tapir --video input.mp4
```

Use `--center X Y`, `--shape ROWS COLS`, and `--spacing PIXELS` to change the
first-frame point cluster. See `RESULTS.md` for the original comparison.
