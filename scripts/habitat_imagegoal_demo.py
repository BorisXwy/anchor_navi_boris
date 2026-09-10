#!/usr/bin/env python3
"""Minimal Habitat RGB observation + goal-image adapter and policy smoke test."""

import argparse
import json
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis

from image_goal_policy import infer_paths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "data/habitat/scene_datasets/habitat-test-scenes/apartment_1.glb"


def make_sim(scene, width=320, height=240):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.enable_physics = False
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [height, width]
    sensor.position = [0.0, 1.25, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def capture(sim, point, yaw=0.0):
    state = sim.get_agent(0).get_state()
    state.position = point
    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)
    return sim.get_sensor_observations()["rgb"][..., :3]


def capture_best_view(sim, point):
    """Avoid an arbitrary wall-facing pose in the smoke-test visualization."""
    candidates = [capture(sim, point, yaw) for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    scores = [im[im.shape[0] // 2 :].std() + 0.25 * im.mean() for im in candidates]
    return candidates[int(np.argmax(scores))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--models", nargs="+", choices=["gnm", "vint", "nomad"],
                   default=["gnm", "vint", "nomad"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/habitat_imagegoal")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = make_sim(args.scene)
    np.random.seed(7)
    obs_point = sim.pathfinder.get_random_navigable_point()
    goal_point = sim.pathfinder.get_random_navigable_point()
    obs = capture_best_view(sim, obs_point)
    goal = capture_best_view(sim, goal_point)
    sim.close()
    obs_path, goal_path = args.output_dir / "observation.png", args.output_dir / "goal.png"
    Image.fromarray(obs).save(obs_path); Image.fromarray(goal).save(goal_path)
    Image.fromarray(np.concatenate([obs, goal], axis=1)).save(args.output_dir / "obs_goal_pair.png")
    results = {
        "scene": str(args.scene.resolve()),
        "observation_xyz": np.asarray(obs_point, dtype=float).tolist(),
        "goal_xyz": np.asarray(goal_point, dtype=float).tolist(), "policies": {},
    }
    for model in args.models:
        results["policies"][model] = infer_paths(
            model, [obs_path], goal_path, args.device, samples=4, seed=7
        )
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
