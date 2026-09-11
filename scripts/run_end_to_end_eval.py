#!/usr/bin/env python3
"""Shard, launch and merge end-to-end R2R episode evaluations.

Usage mirrors ``run_final_method_eval.sh`` from VLN-CE-master: a single
episode id, a comma-separated id list, ``--all`` (the OpenNav100 episode ids),
``--workers N`` parallel shards and ``--dry-run`` preflight.  The dataset is
chosen by ``--start-pose-source``: ``aligned`` (default) is the 100-episode
subset whose ``start_rotation`` is aligned to the instruction's opening turn and
the GT path (``data/datasets/opennav100_start_aligned``, see
``docs/opennav100_start_rotation_alignment_zh.md``); ``official`` is the full
R2R ``val_unseen.json.gz`` as published.  Every shard is one sequential
``evaluate_point_navigation.py`` process, so navigation, scoring and the
RGB-only contract audit stay exactly as they are; this script only adds the
orchestration layer and the round-level ``manifest.json`` / ``process.log`` /
``summary.json``.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_point_navigation import DEFAULT_R2R_DATA, summarize_results  # noqa: E402

DEFAULT_MP3D_ROOT = ROOT.parent / "3d_wm_vln/StreamVLN/data/scene_datasets/mp3d"
DEFAULT_SHARD_RUNNER = ROOT / "scripts/evaluate_point_navigation.py"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/e2e_eval"
OPENNAV100_IDS_FILE = ROOT / "data/opennav100_episode_ids.json"
ALIGNED_OPENNAV100_DATA = (
    ROOT / "data/datasets/opennav100_start_aligned/"
    "val_unseen_opennav100ids_start_aligned.json.gz")
START_POSE_SOURCES = ("aligned", "official")
BENCHMARK_BY_START_POSE_SOURCE = {
    "aligned": "OpenNav_R2R-CE_100 episode ids on official R2R VLN-CE "
               "val_unseen with start_rotation aligned to the opening turn + "
               "GT path (data/datasets/opennav100_start_aligned)",
    "official": "OpenNav_R2R-CE_100 episode ids replayed on official "
                "R2R VLN-CE val_unseen (OpenNav start_rotation not used)",
    "explicit": "explicit --r2r-data file (see dataset_path)",
}
FORBIDDEN_DATASET_MARKER = "OpenNav_R2R-CE_100_bertidx"
MANIFEST_SCHEMA_VERSION = "e2e_round_manifest_v1"
PROVIDER_INTERRUPTION_RETURNCODE = 86
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class PreflightError(RuntimeError):
    pass


class RoundLogger:
    """Print to stdout and, once the round directory exists, to process.log."""

    def __init__(self):
        self._buffer = []
        self._handle = None

    def attach(self, path):
        self._handle = Path(path).open("a")
        for line in self._buffer:
            self._handle.write(line + "\n")
        self._handle.flush()
        self._buffer = []

    def log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if self._handle is None:
            self._buffer.append(line)
        else:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def parse_id_list(text, label):
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise PreflightError(f"{label}: empty list")
    non_numeric = [value for value in values if not value.isdigit()]
    if non_numeric:
        raise PreflightError(f"{label}: non-numeric entries {non_numeric}")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise PreflightError(f"{label}: duplicate entries {duplicates}")
    return values


def load_opennav100_ids(path=OPENNAV100_IDS_FILE):
    payload = json.loads(Path(path).read_text())
    ids = [str(value) for value in payload["episode_ids"]]
    if len(ids) != int(payload.get("n_episodes", len(ids))):
        raise PreflightError(f"{path}: episode_ids count mismatch")
    return ids


def load_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    if FORBIDDEN_DATASET_MARKER in str(dataset_path.resolve()):
        raise PreflightError(
            f"refusing dataset {dataset_path}: OpenNav start_rotation is wrong; "
            "use the official val_unseen.json.gz")
    if not dataset_path.exists():
        raise PreflightError(f"dataset not found: {dataset_path}")
    with gzip.open(dataset_path, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    id_to_index = {}
    for index, episode in enumerate(episodes):
        id_to_index.setdefault(str(episode["episode_id"]), index)
    return episodes, id_to_index


def dataset_sha256(dataset_path):
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def resolve_dataset_path(r2r_data, start_pose_source):
    """Return ``(dataset_path, effective_source)``; an explicit --r2r-data wins."""
    if r2r_data is not None:
        return Path(r2r_data), "explicit"
    if start_pose_source == "official":
        return Path(DEFAULT_R2R_DATA), "official"
    if start_pose_source != "aligned":
        raise PreflightError(f"unknown --start-pose-source {start_pose_source!r}")
    if not ALIGNED_OPENNAV100_DATA.exists():
        raise PreflightError(
            f"aligned dataset missing: {ALIGNED_OPENNAV100_DATA}; build it with "
            "python scripts/build_opennav100_start_aligned_dataset.py or pass "
            "--start-pose-source official")
    return ALIGNED_OPENNAV100_DATA, "aligned"


def episode_record(episode, index):
    return {
        "episode_id": str(episode["episode_id"]),
        "episode_index": index,
        "trajectory_id": episode.get("trajectory_id"),
        "scene_id": episode.get("scene_id"),
        "instruction": (episode.get("instruction") or {}).get(
            "instruction_text", ""),
    }


def resolve_selection(episodes, id_to_index, episode_ids=None,
                      episode_indices=None):
    """Map requested ids or indices onto dataset rows, sorted by index."""
    if episode_ids is not None:
        missing = [value for value in episode_ids if value not in id_to_index]
        if missing:
            raise PreflightError(f"episode ids absent from dataset: {missing}")
        indices = [id_to_index[value] for value in episode_ids]
    else:
        indices = [int(value) for value in episode_indices]
        out_of_range = [
            index for index in indices if not 0 <= index < len(episodes)]
        if out_of_range:
            raise PreflightError(
                f"episode indices outside [0, {len(episodes) - 1}]: "
                f"{out_of_range}")
    return [episode_record(episodes[index], index) for index in sorted(indices)]


def shard_assignment(count, workers):
    """Round-robin positions across workers, like run_final_method_eval.sh."""
    if workers < 1:
        raise PreflightError("--workers must be a positive integer")
    workers = min(workers, max(count, 1))
    return [list(range(worker, count, workers)) for worker in range(workers)]


def resolve_mp3d_scene(scene_id, mp3d_root):
    scene_name = Path(scene_id).stem
    return Path(mp3d_root) / scene_name / f"{scene_name}.glb"


def validate_run_tag(run_tag):
    if not RUN_TAG_PATTERN.match(run_tag) or "/" in run_tag or "\\" in run_tag:
        raise PreflightError(
            f"--run-tag {run_tag!r} must be a plain name without path "
            "separators or a leading dot")
    return run_tag


def shard_command(shard_runner, shard_indices, shard_dir, device, run_options,
                  forwarded_args):
    command = [
        sys.executable, str(shard_runner),
        "--mode", "semantic",
        "--exploration-strategy", "instruction-sequence-recovery",
        "--tracking-cluster-profile", "rgb_only_dense_stop_v1",
        "--episodes", ",".join(str(index) for index in shard_indices),
        "--output-root", str(shard_dir),
        "--device", device,
    ]
    for option, value in run_options.items():
        if value is None:
            continue
        command.extend([f"--{option.replace('_', '-')}", str(value)])
    command.extend(forwarded_args)
    return command


def check_assets(assets_root, policy, hf_home):
    assets_root = Path(assets_root)
    required = {
        "tapir_checkpoint": assets_root / (
            "models/tapnet/tapnet/checkpoints/causal_bootstapir_checkpoint.pt"),
        f"{policy}_policy_weights": (
            assets_root /
            f"models/visualnav-transformer/deployment/model_weights/{policy}.pth"),
        "huggingface_cache": Path(hf_home),
    }
    missing = {name: str(path) for name, path in required.items()
               if not path.exists()}
    if missing:
        raise PreflightError(f"missing local model assets: {missing}")
    return {name: str(path) for name, path in required.items()}


def check_cuda_capacity(devices, workers_per_device, memory_per_worker_gb,
                        force=False):
    import torch  # heavy import, only needed for the GPU preflight

    if not torch.cuda.is_available():
        raise PreflightError("CUDA is not available; refusing to run on CPU")
    report = {}
    problems = []
    for device in devices:
        match = re.fullmatch(r"cuda(?::(\d+))?", device)
        if match is None:
            raise PreflightError(f"unsupported device {device!r}")
        index = int(match.group(1) or 0)
        if index >= torch.cuda.device_count():
            raise PreflightError(
                f"{device} is absent; only {torch.cuda.device_count()} "
                "CUDA device(s) visible")
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        needed_gb = workers_per_device[device] * memory_per_worker_gb
        free_gb = free_bytes / 1e9
        report[device] = {
            "name": torch.cuda.get_device_name(index),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_bytes / 1e9, 2),
            "workers": workers_per_device[device],
            "required_gb": round(needed_gb, 2),
            "sufficient": free_gb >= needed_gb,
        }
        if free_gb < needed_gb:
            problems.append(
                f"{device}: free {free_gb:.1f} GB < "
                f"{workers_per_device[device]} worker(s) x "
                f"{memory_per_worker_gb} GB")
    if problems and not force:
        raise PreflightError(
            "insufficient GPU memory (" + "; ".join(problems) +
            "); reduce --workers, lower --gpu-memory-per-worker-gb, or pass "
            "--force-gpu")
    return report


def check_vlm_credentials(backend, deepseek_env, vlm_model, base_url):
    if backend != "deepseek":
        return {"backend": backend, "credential_source": "not_required",
                "note": ("heuristic backend is a deterministic rule stub for "
                         "smoke tests only" if backend == "heuristic" else "")}
    from vlm_harness import DeepSeekBackend  # noqa: WPS433 (lazy, heavy)

    client = DeepSeekBackend(model=vlm_model, base_url=base_url,
                             env_file=deepseek_env)
    if not client.api_key:
        raise PreflightError(
            "no usable DeepSeek/DMXAPI key: set OPENROUTER_API_KEY in "
            "local_env.sh or provide --deepseek-env")
    return {
        "backend": backend,
        "credential_source": client.credential_source,
        "base_url": client.base_url,
        "model": client.model,
        "thinking": client.thinking,
    }


def git_revision():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  run_end_to_end_eval.py 7                    # one OpenNav episode id\n"
            "  run_end_to_end_eval.py 7,11,13 --workers 2  # explicit ids\n"
            "  run_end_to_end_eval.py --all --workers 2    # all 100 OpenNav ids\n"
            "  run_end_to_end_eval.py 7 --start-pose-source official"
            "  # official start_rotation\n"
            "  run_end_to_end_eval.py --start-pose-source official "
            "--episode-indices 0,3,6,9,18,27,45,126,204,219\n"
            "  run_end_to_end_eval.py --dry-run 7          # preflight only\n"
            "  run_end_to_end_eval.py --list               # print the 100 ids\n"
            "unknown options are forwarded verbatim to evaluate_point_navigation.py"))
    selection = parser.add_argument_group("episode selection (choose one)")
    selection.add_argument("episode_ids", nargs="?", default=None,
                           help="comma-separated R2R episode ids")
    selection.add_argument("--episode-ids", dest="episode_ids_option",
                           default=None, help="same as the positional list")
    selection.add_argument("--episode-indices", default=None,
                           help="comma-separated dataset indices; the fixed "
                                "ten-EP protocol uses indices into the full "
                                "official split, so this requires "
                                "--start-pose-source official (or --r2r-data)")
    selection.add_argument("--all", action="store_true",
                           help="all 100 OpenNav ids from "
                                "data/opennav100_episode_ids.json")
    selection.add_argument("--list", action="store_true",
                           help="print the OpenNav100 ids with dataset "
                                "index/scene and exit")
    selection.add_argument("--resume", type=Path, default=None,
                           help="relaunch the shards of an existing round "
                                "directory; finished episodes are skipped")

    launch = parser.add_argument_group("launch")
    launch.add_argument("--workers", type=int, default=1)
    launch.add_argument("--devices", default=None,
                        help="comma-separated CUDA devices assigned "
                             "round-robin to workers (default: --device)")
    launch.add_argument("--device", default="cuda:0")
    launch.add_argument("--gpu-memory-per-worker-gb", type=float, default=11.0)
    launch.add_argument("--force-gpu", action="store_true",
                        help="launch even if the free-memory check fails")
    launch.add_argument("--run-tag", default="e2e_opennav")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    launch.add_argument("--dry-run", action="store_true",
                        help="run every preflight check and print the "
                             "configuration without creating anything")

    run = parser.add_argument_group("per-episode configuration "
                                    "(omitted = evaluator default)")
    run.add_argument("--start-pose-source", choices=START_POSE_SOURCES,
                     default="aligned",
                     help="aligned (default): the OpenNav100 subset whose "
                          "start_rotation follows the opening turn + GT path "
                          "(data/datasets/opennav100_start_aligned); official: "
                          "the full R2R val_unseen split as published")
    run.add_argument("--r2r-data", type=Path, default=None,
                     help="explicit dataset file; overrides --start-pose-source")
    run.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D_ROOT)
    run.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                     default="deepseek")
    run.add_argument("--vlm-model", default=None)
    run.add_argument("--deepseek-env", type=Path, default=None)
    run.add_argument("--deepseek-base-url", default=None)
    run.add_argument("--policy", choices=["gnm", "vint", "nomad"], default="gnm")
    run.add_argument("--max-steps-per-target", type=int, default=None)
    run.add_argument("--targets", type=int, default=None)
    run.add_argument("--sequence-max-exploration-hops", type=int, default=None)
    run.add_argument("--point-selection-prompt-version", default=None)
    run.add_argument("--instruction-completion-prompt-version", default=None)
    run.add_argument("--seed", type=int, default=None)

    hidden = parser.add_argument_group("test hooks")
    hidden.add_argument("--shard-runner", type=Path, default=DEFAULT_SHARD_RUNNER,
                        help=argparse.SUPPRESS)
    hidden.add_argument("--assets-root", type=Path, default=ROOT,
                        help=argparse.SUPPRESS)
    hidden.add_argument("--allow-cpu", action="store_true",
                        help=argparse.SUPPRESS)
    hidden.add_argument("--poll-interval-s", type=float, default=5.0,
                        help=argparse.SUPPRESS)
    return parser


def run_options_from_args(args):
    return {
        "r2r_data": str(args.r2r_data),
        "policy": args.policy,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "deepseek_env": (str(args.deepseek_env)
                         if args.deepseek_env is not None else None),
        "deepseek_base_url": args.deepseek_base_url,
        "max_steps_per_target": args.max_steps_per_target,
        "targets": args.targets,
        "sequence_max_exploration_hops": args.sequence_max_exploration_hops,
        "point_selection_prompt_version": args.point_selection_prompt_version,
        "instruction_completion_prompt_version": (
            args.instruction_completion_prompt_version),
        "seed": args.seed,
    }


def select_episodes(args, episodes, id_to_index):
    positional = args.episode_ids
    option = args.episode_ids_option
    sources = [name for name, value in (
        ("ids", positional or option), ("indices", args.episode_indices),
        ("all", args.all)) if value]
    if positional and option:
        raise PreflightError("episode ids must be given once, comma-separated")
    if len(sources) != 1:
        raise PreflightError(
            "choose exactly one of: <ids>, --episode-ids, --episode-indices, "
            "--all")
    source = sources[0]
    if source == "indices" and args.start_pose_source_effective == "aligned":
        raise PreflightError(
            "--episode-indices refers to rows of the full official split; the "
            "aligned dataset holds only the 100 OpenNav episodes, so pass "
            "--start-pose-source official (or episode ids)")
    if source == "all":
        selection = resolve_selection(
            episodes, id_to_index, episode_ids=load_opennav100_ids())
    elif source == "ids":
        selection = resolve_selection(
            episodes, id_to_index,
            episode_ids=parse_id_list(positional or option, "episode ids"))
    else:
        selection = resolve_selection(
            episodes, id_to_index,
            episode_indices=parse_id_list(args.episode_indices,
                                          "episode indices"))
    return source, selection


def print_listing(episodes, id_to_index):
    ids = load_opennav100_ids()
    for value in ids:
        index = id_to_index.get(value)
        if index is None:
            print(f"{value}\tabsent")
            continue
        record = episode_record(episodes[index], index)
        print(f"{value}\tindex={index}\t{Path(record['scene_id']).stem}\t"
              f"{record['instruction'][:70]}")
    print(",".join(ids))
    print(f"# n={len(ids)}")


def collect_provider_interruptions(round_dir):
    return [json.loads(path.read_text())
            for path in sorted(Path(round_dir).glob(
                "shard_*/provider_interruption_*.json"))]


def episode_trajectory_exists(shard_dir, index, episode_id):
    """Episode directories are named by episode_id; rounds before 2026-09-10
    used the dataset row index, so a legacy directory only counts when its
    trajectory really belongs to this index."""
    def trajectory_index(path):
        try:
            trajectory = json.loads(path.read_text())
            return int((trajectory.get("config") or {}).get(
                "episode_index", trajectory.get("episode_index")))
        except (OSError, ValueError, TypeError, AttributeError,
                json.JSONDecodeError):
            return None

    by_id = shard_dir / f"episode_{int(episode_id):04d}" / "trajectory.json"
    if by_id.exists() and trajectory_index(by_id) in (None, index):
        return True
    legacy = shard_dir / f"episode_{index:04d}" / "trajectory.json"
    return legacy.exists() and trajectory_index(legacy) == index


def merge_shards(round_dir, shards):
    """Merge shard summaries; episodes without a score stay explicit."""
    round_dir = Path(round_dir)
    results = []
    unscored = []
    not_run = []
    configuration = None
    for shard in shards:
        shard_dir = round_dir / shard["directory"]
        summary_path = shard_dir / "summary.json"
        scored = {}
        if summary_path.exists():
            shard_summary = json.loads(summary_path.read_text())
            configuration = configuration or shard_summary.get("configuration")
            scored = {int(item["episode_index"]): item
                      for item in shard_summary.get("results", [])}
        for index, episode_id in zip(shard["episode_indices"],
                                     shard["episode_ids"]):
            if index in scored:
                results.append(scored[index])
            elif episode_trajectory_exists(shard_dir, index, episode_id):
                unscored.append(index)
            else:
                not_run.append(index)
    results.sort(key=lambda item: int(item["episode_index"]))
    scored_indices = [int(item["episode_index"]) for item in results]
    if results:
        summary = summarize_results(
            results, scored_indices, "semantic", configuration or {})
    else:
        summary = {"episode_count": 0, "episode_indices": [],
                   "configuration": configuration or {}, "results": []}
    summary.update({
        "round_dir": str(round_dir),
        "shards": [{**shard, "summary_present": (
            round_dir / shard["directory"] / "summary.json").exists()}
            for shard in shards],
        "scored_episode_indices": scored_indices,
        "process_failed_episode_indices": [
            int(item["episode_index"]) for item in results
            if item.get("termination_category") == "episode_process_failed"],
        "unscored_episode_indices": sorted(unscored),
        "not_run_episode_indices": sorted(not_run),
        "provider_interruptions": collect_provider_interruptions(round_dir),
    })
    return summary


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def launch_shards(round_dir, shards, logger, poll_interval_s):
    round_dir = Path(round_dir)
    processes = {}
    for shard in shards:
        shard_dir = round_dir / shard["directory"]
        shard_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update({
            "MAGNUM_LOG": "quiet",
            "HABITAT_SIM_LOG": "quiet",
            "PYTHONPATH": str(ROOT / "scripts"),
        })
        log_handle = (shard_dir / "shard.log").open("a")
        process = subprocess.Popen(
            shard["command"], cwd=ROOT, env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT)
        processes[shard["directory"]] = (process, log_handle, shard)
        logger.log(
            f"launched {shard['directory']} pid={process.pid} "
            f"device={shard['device']} episodes={shard['episode_indices']}")
    try:
        pending = set(processes)
        while pending:
            for name in sorted(pending):
                process, log_handle, shard = processes[name]
                code = process.poll()
                if code is None:
                    continue
                log_handle.close()
                shard["returncode"] = code
                shard["finished_utc"] = datetime.now(timezone.utc).isoformat()
                pending.discard(name)
                note = (" (VLM provider interruption)"
                        if code == PROVIDER_INTERRUPTION_RETURNCODE else "")
                logger.log(f"finished {name} returncode={code}{note}")
            if pending:
                time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        logger.log("interrupted; terminating shards")
        for process, log_handle, shard in processes.values():
            if process.poll() is None:
                process.terminate()
        for process, log_handle, shard in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()
            shard["returncode"] = process.returncode
        raise


def print_config_snapshot(logger, selection, shards, devices, gpu_report,
                          vlm_report, assets, run_options, forwarded_args,
                          round_dir, dataset_path, dataset_digest,
                          start_pose_source):
    logger.log("=== configuration ===")
    logger.log(f"episodes       : {len(selection)} -> "
               f"ids={[item['episode_id'] for item in selection]}")
    logger.log(f"indices        : {[item['episode_index'] for item in selection]}")
    logger.log(f"start pose src : {start_pose_source}")
    logger.log(f"dataset        : {dataset_path} sha256={dataset_digest[:12]}")
    logger.log(f"workers        : {len(shards)} devices={devices}")
    for shard in shards:
        logger.log(f"  {shard['directory']}: device={shard['device']} "
                   f"episodes={shard['episode_indices']}")
    for device, item in gpu_report.items():
        logger.log(f"gpu {device}       : {item}")
    logger.log(f"vlm            : {vlm_report}")
    logger.log(f"assets         : {assets}")
    logger.log(f"run options    : "
               f"{ {k: v for k, v in run_options.items() if v is not None} }")
    logger.log(f"forwarded args : {forwarded_args}")
    logger.log(f"round dir      : {round_dir}")


def main(argv=None):
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    unknown_flags = [item for item in forwarded_args if item.startswith("-")]
    if forwarded_args and not unknown_flags:
        parser.error(f"unrecognized arguments: {forwarded_args}")
    logger = RoundLogger()
    try:
        return orchestrate(args, forwarded_args, logger)
    except PreflightError as error:
        logger.log(f"preflight failed: {error}")
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        logger.close()


def orchestrate(args, forwarded_args, logger):
    resumed_manifest = None
    if args.resume is not None:
        manifest_path = Path(args.resume) / "manifest.json"
        if not manifest_path.exists():
            raise PreflightError(f"--resume: no manifest at {manifest_path}")
        resumed_manifest = json.loads(manifest_path.read_text())
        if any((args.episode_ids, args.episode_ids_option,
                args.episode_indices, args.all)):
            raise PreflightError("--resume takes its episode list from the "
                                 "manifest; do not pass a selection")
        args.r2r_data = Path(resumed_manifest["dataset_path"])
        args.mp3d_root = Path(resumed_manifest["mp3d_root"])
        start_pose_source = resumed_manifest.get("start_pose_source", "explicit")
    else:
        args.r2r_data, start_pose_source = resolve_dataset_path(
            args.r2r_data, args.start_pose_source)
    args.start_pose_source_effective = start_pose_source

    episodes, id_to_index = load_dataset(args.r2r_data)
    if args.list:
        print_listing(episodes, id_to_index)
        return 0

    if resumed_manifest is not None:
        selection = resumed_manifest["episode_selection"]["episodes"]
        selection_source = resumed_manifest["episode_selection"]["source"]
        run_options = resumed_manifest["run_options"]
        forwarded_args = list(resumed_manifest["forwarded_args"])
        workers = int(resumed_manifest["workers"])
        devices = list(resumed_manifest["devices"])
        run_tag = resumed_manifest["run_tag"]
        round_dir = Path(args.resume)
        digest = dataset_sha256(args.r2r_data)
        if digest != resumed_manifest["dataset_sha256"]:
            raise PreflightError("--resume: dataset sha256 differs from manifest")
    else:
        run_tag = validate_run_tag(args.run_tag)
        selection_source, selection = select_episodes(
            args, episodes, id_to_index)
        run_options = run_options_from_args(args)
        workers = args.workers
        if workers < 1:
            raise PreflightError("--workers must be a positive integer")
        devices = ([item.strip() for item in args.devices.split(",")
                    if item.strip()] if args.devices else [args.device])
        if workers > len(selection):
            logger.log(f"Info: reducing --workers {workers} -> "
                       f"{len(selection)} (episode count)")
            workers = len(selection)
        digest = dataset_sha256(args.r2r_data)
        round_dir = (args.output_root /
                     f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}")

    missing_scenes = sorted({
        str(resolve_mp3d_scene(item["scene_id"], args.mp3d_root))
        for item in selection
        if not resolve_mp3d_scene(item["scene_id"], args.mp3d_root).exists()})
    if missing_scenes:
        raise PreflightError(f"missing MP3D scenes: {missing_scenes}")

    assets = check_assets(
        args.assets_root, run_options.get("policy") or "gnm",
        os.environ.get("HF_HOME", str(Path(args.assets_root) /
                                      "weights/huggingface")))

    positions = shard_assignment(len(selection), workers)
    shards = []
    for worker, chunk in enumerate(positions):
        device = devices[worker % len(devices)]
        shard_indices = [selection[position]["episode_index"]
                         for position in chunk]
        shard_dir_name = f"shard_{worker}"
        shards.append({
            "directory": shard_dir_name,
            "device": device,
            "episode_indices": shard_indices,
            "episode_ids": [selection[position]["episode_id"]
                            for position in chunk],
            "command": shard_command(
                args.shard_runner, shard_indices, round_dir / shard_dir_name,
                device, run_options, forwarded_args),
            "returncode": None,
        })
    workers_per_device = {device: 0 for device in devices}
    for shard in shards:
        workers_per_device[shard["device"]] += 1

    if all(device.startswith("cuda") for device in devices):
        gpu_report = check_cuda_capacity(
            devices, workers_per_device, args.gpu_memory_per_worker_gb,
            force=args.force_gpu)
    elif args.allow_cpu:
        gpu_report = {device: {"note": "cpu device allowed by --allow-cpu "
                                       "(unit tests only)"}
                      for device in devices}
    else:
        raise PreflightError(
            f"devices {devices} are not CUDA; production runs must use CUDA "
            "(project_rulle.md 16.4)")

    vlm_report = check_vlm_credentials(
        run_options.get("vlm_backend") or "deepseek",
        run_options.get("deepseek_env"), run_options.get("vlm_model"),
        run_options.get("deepseek_base_url"))

    print_config_snapshot(
        logger, selection, shards, devices, gpu_report, vlm_report, assets,
        run_options, forwarded_args, round_dir, args.r2r_data, digest,
        start_pose_source)
    if args.dry_run:
        logger.log("dry run: preflight passed; nothing launched")
        return 0

    round_dir.mkdir(parents=True, exist_ok=True)
    logger.attach(round_dir / "process.log")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "round_id": round_dir.name,
        "run_tag": run_tag,
        "created_utc": (resumed_manifest or {}).get(
            "created_utc", datetime.now(timezone.utc).isoformat()),
        "resumed_utc": (datetime.now(timezone.utc).isoformat()
                        if resumed_manifest else None),
        "status": "running",
        "test_scope": "end_to_end_from_dataset_start",
        "benchmark": BENCHMARK_BY_START_POSE_SOURCE[start_pose_source],
        "start_pose_source": start_pose_source,
        "git": git_revision(),
        "dataset_path": str(Path(args.r2r_data).resolve()),
        "dataset_sha256": digest,
        "mp3d_root": str(args.mp3d_root),
        "episode_selection": {
            "source": selection_source,
            "count": len(selection),
            "episodes": selection,
        },
        "workers": len(shards),
        "devices": devices,
        "gpu_memory_check": gpu_report,
        "gpu_memory_per_worker_gb": args.gpu_memory_per_worker_gb,
        "vlm": vlm_report,
        "assets": assets,
        "run_options": run_options,
        "forwarded_args": forwarded_args,
        "shard_runner": str(args.shard_runner),
        "shards": shards,
        "not_run_episode_indices": [item["episode_index"] for item in selection],
        "unscored_episode_indices": [],
    }
    write_json(round_dir / "manifest.json", manifest)

    try:
        launch_shards(round_dir, shards, logger, args.poll_interval_s)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["shards"] = shards
        write_json(round_dir / "manifest.json", manifest)
        raise

    summary = merge_shards(round_dir, shards)
    write_json(round_dir / "summary.json", summary)
    failed = [shard["directory"] for shard in shards if shard["returncode"] != 0]
    manifest.update({
        "status": "complete" if not failed else "complete_with_failures",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "shards": shards,
        "not_run_episode_indices": summary["not_run_episode_indices"],
        "unscored_episode_indices": summary["unscored_episode_indices"],
    })
    write_json(round_dir / "manifest.json", manifest)

    logger.log("=== results ===")
    logger.log(f"round dir      : {round_dir}")
    logger.log(f"summary        : {round_dir / 'summary.json'}")
    logger.log(f"scored         : {summary['episode_count']}/{len(selection)}")
    logger.log("simulator_reported_success: "
               f"{summary.get('simulator_reported_success_count', 0)}/"
               f"{summary['episode_count']}")
    if summary["process_failed_episode_indices"]:
        logger.log("episode process crashed (scored as failure, see "
                   "episode_<id>/process.log): "
                   f"{summary['process_failed_episode_indices']}")
    if summary["unscored_episode_indices"]:
        logger.log(f"unscored (trajectory present, shard died before scoring; "
                   f"rerun with --resume {round_dir}): "
                   f"{summary['unscored_episode_indices']}")
    if summary["not_run_episode_indices"]:
        logger.log(f"not_run: {summary['not_run_episode_indices']}")
    for shard in shards:
        logger.log(f"  {shard['directory']}: returncode={shard['returncode']} "
                   f"log={round_dir / shard['directory'] / 'shard.log'}")
    logger.log("post-run audits (manual): "
               f"python scripts/audit_active_stop_round.py {round_dir} "
               f"--dataset {args.r2r_data}; "
               f"python scripts/verify_round_stage_completions.py {round_dir} "
               f"--dataset {args.r2r_data}")
    if failed:
        logger.log(f"one or more shards exited non-zero: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
