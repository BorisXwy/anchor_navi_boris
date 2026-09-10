#!/usr/bin/env python3
"""Track a first-frame point cluster with a causal/streaming point tracker."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TAPNET_ROOT = ROOT / "models" / "tapnet"
sys.path.insert(0, str(TAPNET_ROOT))


def read_video(path: Path, max_frames: int | None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return np.stack(frames), float(fps)


def make_cluster(cx: float, cy: float, rows: int, cols: int, spacing: float):
    xs = cx + (np.arange(cols) - (cols - 1) / 2) * spacing
    ys = cy + (np.arange(rows) - (rows - 1) / 2) * spacing
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)


def run_cotracker(frames, points, device, checkpoint):
    from cotracker.predictor import CoTrackerOnlinePredictor

    model = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).to(device).eval()
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.from_numpy(
        np.concatenate([np.zeros((len(points), 1), np.float32), points], axis=1)
    )[None].to(device)  # t, x, y
    model(video[:, :1], is_first_step=True, queries=queries, grid_size=0,
          add_support_grid=True)
    tracks = visibility = None
    step = model.step
    for start in range(0, len(frames) - step, step):
        chunk = video[:, start : start + 2 * step]
        if chunk.shape[1] < 2 * step:
            pad = chunk[:, -1:].expand(-1, 2 * step - chunk.shape[1], -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=1)
        tracks, visibility = model(
            chunk, queries=None, grid_size=0, add_support_grid=True
        )
    if tracks is None:
        raise RuntimeError("Video is shorter than CoTracker's streaming step")
    return (tracks[0, : len(frames)].cpu().numpy(),
            visibility[0, : len(frames)].cpu().numpy().astype(bool),
            {"streaming_step": int(step), "window": int(2 * step)})


def run_tapir(frames, points, device, checkpoint):
    import tree
    import torch.nn.functional as F
    from tapnet.torch import tapir_model

    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    h, w = frames.shape[1:3]
    rh = rw = 256
    resized = np.stack([
        cv2.resize(f, (rw, rh), interpolation=cv2.INTER_AREA) for f in frames
    ])
    video = torch.from_numpy(resized).to(device)
    q = np.concatenate([
        np.zeros((len(points), 1), np.float32),
        points[:, 1:2] * (rh / h),
        points[:, 0:1] * (rw / w),
    ], axis=1)  # t, y, x
    q = torch.from_numpy(q)[None].to(device)

    def preprocess(x):
        return x.float() / 255.0 * 2.0 - 1.0

    first = preprocess(video[None, 0:1])
    grids = model.get_feature_grids(first, is_training=False)
    features = model.get_query_features(
        first, is_training=False, query_points=q, feature_grids=grids
    )
    causal = model.construct_initial_causal_state(
        len(points), len(features.resolutions) - 1
    )
    causal = tree.map_structure(lambda x: x.to(device), causal)
    all_tracks, all_visible = [], []
    with torch.no_grad():
        for frame in video:
            x = preprocess(frame[None, None])
            grids = model.get_feature_grids(x, is_training=False)
            out = model.estimate_trajectories(
                x.shape[-3:-1], is_training=False, feature_grids=grids,
                query_features=features, query_points_in_video=None,
                query_chunk_size=64, causal_context=causal,
                get_causal_context=True,
            )
            causal = out["causal_context"]
            tr = out["tracks"][-1][0, :, 0]  # points, xy
            occ = out["occlusion"][-1][0, :, 0]
            dist = out["expected_dist"][-1][0, :, 0]
            vis = (1 - torch.sigmoid(occ)) * (1 - torch.sigmoid(dist)) > 0.5
            all_tracks.append(tr)
            all_visible.append(vis)
    tracks = torch.stack(all_tracks).cpu().numpy()
    tracks[..., 0] *= w / rw
    tracks[..., 1] *= h / rh
    visible = torch.stack(all_visible).cpu().numpy().astype(bool)
    return tracks, visible, {"streaming_step": 1, "window": 1,
                             "inference_resolution": [rh, rw]}


def render(frames, tracks, visible, initial_points, output, fps):
    colors = cv2.applyColorMap(
        np.linspace(20, 235, len(initial_points), dtype=np.uint8),
        cv2.COLORMAP_TURBO,
    )[:, 0]
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    history = [[] for _ in initial_points]
    for ti, rgb in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for pi, color in enumerate(colors):
            if visible[ti, pi]:
                p = tuple(np.round(tracks[ti, pi]).astype(int))
                history[pi].append(p)
                for a, b in zip(history[pi][-12:-1], history[pi][-11:]):
                    cv2.line(bgr, a, b, color.tolist(), 1, cv2.LINE_AA)
                cv2.circle(bgr, p, 4, color.tolist(), -1, cv2.LINE_AA)
        cv2.putText(bgr, f"frame {ti:03d}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(bgr)
    writer.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cotracker", "tapir"], required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--center", type=float, nargs=2, default=(310, 270), metavar=("X", "Y"))
    p.add_argument("--shape", type=int, nargs=2, default=(5, 5), metavar=("ROWS", "COLS"))
    p.add_argument("--spacing", type=float, default=10)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            p.error(
                f"{args.device} was requested, but CUDA is unavailable; "
                "refusing a CPU fallback for point tracking")
        device_index = (requested_device.index
                        if requested_device.index is not None
                        else torch.cuda.current_device())
        if device_index >= torch.cuda.device_count():
            p.error(
                f"requested CUDA device {device_index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible")
        torch.cuda.set_device(device_index)
    frames, fps = read_video(args.video, args.max_frames)
    points = make_cluster(*args.center, *args.shape, args.spacing)
    h, w = frames.shape[1:3]
    if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
        raise ValueError("Point cluster extends outside the video frame")
    checkpoint = (ROOT / "models/co-tracker/checkpoints/scaled_online.pth"
                  if args.backend == "cotracker" else
                  ROOT / "models/tapnet/tapnet/checkpoints/causal_bootstapir_checkpoint.pt")
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if args.backend == "cotracker":
        tracks, visible, extra = run_cotracker(frames, points, args.device, checkpoint)
    else:
        tracks, visible, extra = run_tapir(frames, points, args.device, checkpoint)
    elapsed = time.perf_counter() - start
    stem = f"{args.video.stem}_{args.backend}_cluster"
    video_out = args.output_dir / f"{stem}.mp4"
    npz_out = args.output_dir / f"{stem}.npz"
    json_out = args.output_dir / f"{stem}.json"
    render(frames, tracks, visible, points, video_out, fps)
    np.savez_compressed(npz_out, tracks=tracks, visibility=visible,
                        query_points=points)
    metrics = {
        "backend": args.backend, "input": str(args.video.resolve()),
        "frames": len(frames), "size": [w, h], "fps": fps,
        "points": len(points), "center_xy": list(args.center),
        "cluster_shape": list(args.shape), "spacing": args.spacing,
        "elapsed_sec_including_load": elapsed,
        "effective_fps_including_load": len(frames) / elapsed,
        "mean_visible_fraction": float(visible.mean()),
        "last_frame_visible_fraction": float(visible[-1].mean()),
        "peak_cuda_memory_gib": (torch.cuda.max_memory_allocated() / 2**30
                                 if str(args.device).startswith("cuda") else 0.0),
        **extra,
    }
    json_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"video: {video_out}\ntracks: {npz_out}")


if __name__ == "__main__":
    main()
