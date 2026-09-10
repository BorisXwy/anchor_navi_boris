#!/usr/bin/env python3
"""ROS-free ViNT/NoMaD inference: observation context + one goal image -> waypoints."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


ROOT = Path(__file__).resolve().parents[1]
VINT_ROOT = ROOT / "models/visualnav-transformer"
DIFFUSION_ROOT = ROOT / "models/diffusion_policy"
sys.path.insert(0, str(VINT_ROOT / "train"))
sys.path.insert(0, str(DIFFUSION_ROOT))

from vint_train.models.vint.vint import ViNT
from vint_train.models.gnm.gnm import GNM
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


WEIGHTS = VINT_ROOT / "deployment/model_weights"
CONFIGS = VINT_ROOT / "train/config"
ACTION_MIN = np.array([-2.5, -4.0], np.float32)
ACTION_MAX = np.array([5.0, 4.0], np.float32)


def transform_images(images, size):
    tfm = transforms.Compose([
        transforms.Resize((size[1], size[0])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return torch.cat([tfm(im.convert("RGB"))[None] for im in images], dim=1)


def load_config(name):
    with (CONFIGS / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)


def checkpoint_state(path, model_type):
    checkpoint = torch.load(path, map_location="cpu")
    if model_type == "nomad":
        return checkpoint
    saved = checkpoint["model"]
    return saved.module.state_dict() if hasattr(saved, "module") else saved.state_dict()


def build_policy(name, device):
    cfg = load_config(name)
    if name == "gnm":
        model = GNM(
            context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"],
            learn_angle=cfg["learn_angle"],
            obs_encoding_size=cfg["obs_encoding_size"],
            goal_encoding_size=cfg["goal_encoding_size"],
        )
    elif name == "vint":
        model = ViNT(
            context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"],
            learn_angle=cfg["learn_angle"], obs_encoder=cfg["obs_encoder"],
            obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
            mha_num_attention_heads=cfg["mha_num_attention_heads"],
            mha_num_attention_layers=cfg["mha_num_attention_layers"],
            mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
        )
    elif name == "nomad":
        encoder = NoMaD_ViNT(
            obs_encoding_size=cfg["encoding_size"], context_size=cfg["context_size"],
            mha_num_attention_heads=cfg["mha_num_attention_heads"],
            mha_num_attention_layers=cfg["mha_num_attention_layers"],
            mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
        )
        encoder = replace_bn_with_gn(encoder)
        model = NoMaD(
            encoder,
            ConditionalUnet1D(input_dim=2, global_cond_dim=cfg["encoding_size"],
                              down_dims=cfg["down_dims"],
                              cond_predict_scale=cfg["cond_predict_scale"]),
            DenseNetwork(cfg["encoding_size"]),
        )
    else:
        raise ValueError(name)
    missing, unexpected = model.load_state_dict(
        checkpoint_state(WEIGHTS / f"{name}.pth", cfg["model_type"]), strict=False
    )
    if missing or unexpected:
        print(f"checkpoint compatibility: missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval(), cfg


@torch.inference_mode()
def predict(model, cfg, context, goal, name, device, samples=8, seed=0):
    required = cfg["context_size"] + 1
    if len(context) == 1:
        context = context * required
    if len(context) != required:
        raise ValueError(f"{name} needs {required} observation frames, got {len(context)}")
    obs = transform_images(context, cfg["image_size"]).to(device)
    goal_tensor = transform_images([goal], cfg["image_size"]).to(device)
    if name in {"gnm", "vint"}:
        distance, actions = model(obs, goal_tensor)
        return float(distance.item()), actions[0].cpu().numpy()[None]

    goal_mask = torch.zeros(1, dtype=torch.long, device=device)
    cond = model("vision_encoder", obs_img=obs, goal_img=goal_tensor,
                 input_goal_mask=goal_mask)
    distance = model("dist_pred_net", obsgoal_cond=cond).item()
    cond = cond.repeat_interleave(samples, 0)
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.randn((samples, cfg["len_traj_pred"], 2),
                          generator=generator, device=device)
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(cfg["num_diffusion_iters"], device=device)
    for k in scheduler.timesteps:
        noise = model("noise_pred_net", sample=actions, timestep=k, global_cond=cond)
        actions = scheduler.step(noise, k, actions).prev_sample
    deltas = actions.cpu().numpy()
    deltas = (deltas + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
    return float(distance), np.cumsum(deltas, axis=1)


def infer_paths(name, observations, goal_path, device="cuda:0", samples=8, seed=0):
    device = torch.device(device)
    model, cfg = build_policy(name, device)
    context = [Image.open(p).convert("RGB") for p in observations]
    goal = Image.open(goal_path).convert("RGB")
    start = time.perf_counter()
    distance, trajectories = predict(model, cfg, context, goal, name, device, samples, seed)
    elapsed = time.perf_counter() - start
    return {
        "model": name, "distance": distance,
        "trajectories": trajectories.tolist(), "elapsed_sec": elapsed,
        "context_frames": len(context), "goal_image": str(Path(goal_path).resolve()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gnm", "vint", "nomad"], required=True)
    p.add_argument("--obs", type=Path, nargs="+", required=True,
                   help="One image (repeated for cold start), or context_size+1 frames")
    p.add_argument("--goal", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = infer_paths(args.model, args.obs, args.goal, args.device, args.samples, args.seed)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
