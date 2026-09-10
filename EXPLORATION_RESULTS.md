# Point-guided Habitat exploration

Command:

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet HF_HOME=weights/huggingface \
  .venv/bin/python scripts/random_exploration.py \
  --episode-index 0 --policy gnm --max-steps-per-target 14 --device cuda:0 \
  --vlm-backend ollama --vlm-model llama3.2-vision:latest \
  --output-dir outputs/r2r_ep1_llava_stages
```

The current controller uses local Ollama `llama3.2-vision:latest` to split the
instruction into `Exit the bedroom`, `Turn left`, `Walk straight`, and `Stop near
the rug`. For each stage the model sees a labeled six-view contact sheet with
SegFormer floor overlays and selects a view plus normalized floor coordinate.
The harness enforces an allowed-view JSON enum and snaps the point onto the mask.

In the verified run, the first three stages reached the strict arrival threshold
at visible fractions `4/9`, `0/9`, and `3/9`. The fourth retained `6/9` points
after 14 steps, so it was correctly marked `max_steps`, not reached. The full
trajectory moved 5.370 m in 31 control steps.

Artifacts:

- `outputs/r2r_ep1_llava_stages/exploration.mp4`: scan and point tracking video
- `outputs/r2r_ep1_llava_stages/trajectory.json`: stages and full motion log
- `outputs/r2r_ep1_llava_stages/vlm_calls.json`: structured VLM harness records

The visual controller deliberately weights TAPIR pixel bearing more than the
navigation model waypoint bearing (80/20). This keeps the baseline faithful to
the selected image-space ground target despite the domain gap between Habitat
and the real-robot training data used by GNM/ViNT/NoMaD.

## Stage-level VLM with action-history handoff

The corrected controller calls the VLM exactly once per decomposed instruction
stage. It selects a floor point from six current surrounding views, then TAPIR
tracks that fixed cluster for all low-level navigation steps. Once at least half
the points disappear, the controller stores the executed action history,
captures six new views at the reached pose, and includes that history when
selecting the next instruction stage's point.

The verified episode-3 run is in
`outputs/r2r_ep3_direct6_video_backtrack_v2/`. It made one decomposition call and
four stage-selection calls for four instruction stages, with zero per-step VLM
calls. Six views are acquired simultaneously from Habitat sensors. From stage 2
onward, the panorama view pointing back toward the prior stage origin is marked
`BACKTRACK BLOCKED` and excluded from the VLM schema. The video includes the
decision and tracking/control visualizations.
