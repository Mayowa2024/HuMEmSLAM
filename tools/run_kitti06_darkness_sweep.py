#!/usr/bin/env python3
"""Run paired repeated ORB-SLAM3/HuMemSLAM KITTI 06 darkness trials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/home/teleopbike/Documents/slam_experiments/datasets/"
    "KITTI/dataset/sequences/06"
)
DEFAULT_SETTINGS = Path(
    "/home/teleopbike/ORB_SLAM3/Examples/Stereo/KITTI04-12.yaml"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=828)
    parser.add_argument("--end-frame", type=int, default=1100)
    parser.add_argument(
        "--darkening-levels",
        type=int,
        nargs="+",
        default=list(range(10, 100, 10)),
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--semantic-threshold", type=float, default=0.70)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def generate_dataset(args: argparse.Namespace, darkening: int) -> Path:
    dataset = args.output / "datasets" / f"darkening_{darkening:02d}"
    if (dataset / "scenario.json").exists():
        return dataset
    command = [
        sys.executable,
        str(Path(__file__).with_name("make_relocalisation_scenario.py")),
        "--source", str(args.source),
        "--output", str(dataset),
        "--no-dropout",
        "--night-start", str(args.start_frame),
        "--night-end", str(args.end_frame),
        "--brightness-scale", f"{1.0 - darkening / 100.0:.2f}",
    ]
    subprocess.run(command, check=True)
    return dataset


def run_trial(
    args: argparse.Namespace,
    dataset: Path,
    darkening: int,
    run_number: int,
    human: bool,
) -> None:
    condition = "humanslam_only" if human else "baseline"
    run_dir = (
        args.output / f"darkening_{darkening:02d}" /
        condition / f"run_{run_number:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    completion = run_dir / "run_complete.json"
    if completion.exists():
        print(f"SKIP completed {darkening}% {condition} run {run_number:02d}", flush=True)
        return

    command = [
        "timeout", f"{args.timeout_seconds}s",
        "ros2", "launch", "slam", "offline_benchmark.launch.py",
        f"dataset_path:={dataset}",
        f"settings_file:={args.settings}",
        f"use_human_slam:={'true' if human else 'false'}",
        f"semantic_threshold:={args.semantic_threshold}",
        f"results_dir:={run_dir}",
    ]
    started = time.time()
    print(f"START {darkening}% {condition} run {run_number:02d}", flush=True)
    with (run_dir / "run_console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    trajectory = run_dir / "trajectory_kitti.txt"
    pose_count = 0
    if trajectory.exists():
        with trajectory.open(encoding="utf-8", errors="replace") as stream:
            pose_count = sum(1 for line in stream if line.strip())
    record = {
        "darkening_percent": darkening,
        "brightness_retained": 1.0 - darkening / 100.0,
        "condition": condition,
        "run": run_number,
        "return_code": result.returncode,
        "elapsed_seconds": time.time() - started,
        "trajectory_pose_count": pose_count,
        "complete_trajectory": pose_count == 1101,
        "dataset": str(dataset),
        "semantic_threshold": args.semantic_threshold,
    }
    completion.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE {darkening}% {condition} run {run_number:02d}: "
        f"rc={result.returncode} poses={pose_count}",
        flush=True,
    )


def main() -> None:
    args = arguments()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    levels = sorted(set(args.darkening_levels))
    if any(level < 0 or level >= 100 for level in levels):
        raise ValueError("darkening levels must be between 0 and 99")
    run_count = 1 if args.smoke_only else args.runs

    manifest = {
        "source": str(args.source.resolve()),
        "settings": str(args.settings.resolve()),
        "perturbation_frames_inclusive": [args.start_frame, args.end_frame],
        "darkening_levels_percent": levels,
        "runs_per_condition_per_level": run_count,
        "conditions": ["ORB-SLAM3 baseline", "HuMemSLAM only"],
        "semantic_threshold": args.semantic_threshold,
        "total_planned_runs": len(levels) * run_count * 2,
    }
    (args.output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    datasets = {level: generate_dataset(args, level) for level in levels}


    for level in levels:
        for run_number in range(1, run_count + 1):
            run_trial(args, datasets[level], level, run_number, human=False)
            run_trial(args, datasets[level], level, run_number, human=True)


if __name__ == "__main__":
    main()
