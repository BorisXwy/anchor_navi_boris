# First-frame cluster experiment

Input: `../3d_wm_vln/fantasy-world/assets/aether/video_1.mp4` (41 frames,
528x352, 10 fps). The query is a 5x5 point cluster with 10-pixel spacing,
centered at `(310, 270)` on the foreground sofa in frame 0.

| Backend | Streaming behavior | Time incl. load | Effective rate | Peak VRAM | Mean visible |
|---|---|---:|---:|---:|---:|
| CoTracker 3 Online | 16-frame sliding window, 8-frame step | 5.50 s | 7.46 fps | 1.59 GiB | 30.44% |
| Causal BootsTAPIR | strictly causal, one frame per step | 5.04 s | 8.14 fps | 0.52 GiB | 29.85% |

Both trackers stay visually attached to the sofa while it is in view. The sofa
leaves the right edge around frames 11--14; both models progressively mark the
cluster invisible and report all 25 points invisible from frame 14 onward.
Thus the low whole-video visible fraction is expected occlusion/out-of-frame
behavior rather than early tracking failure. Where both models report points
visible (305 point-frames), their mean position disagreement is 1.77 pixels
(median 1.05 pixels).

These timings are a smoke-test comparison, not steady-state latency: they
include checkpoint loading and the models use different internal resolutions.
CoTracker additionally has an 8-frame scheduling step, while BootsTAPIR emits a
causal update for every incoming frame.
