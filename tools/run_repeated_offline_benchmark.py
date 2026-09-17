#!/usr/bin/env python3
"""Run resumable repeated matched baseline/HuMemSLAM offline benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--perturb-start", type=int, default=828)
    parser.add_argument(
        "--save-semantic-frames", action="store_true",
        help="Save large per-frame semantic debug images for every HuMemSLAM run.",
    )
    return parser.parse_args()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = arguments()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "dataset": str(args.dataset.resolve()),
        "settings": str(args.settings.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "runs": args.runs,
        "conditions_per_repetition": ["baseline", "humanslam"],
        "perturbation_start_frame": args.perturb_start,
        "semantic_threshold": 0.70,
        "geometry_profile": "unchanged",
        "save_semantic_frames": args.save_semantic_frames,
    }
    write_json(args.output / "experiment_manifest.json", manifest)
    status = {"state": "running", "completed": [], "failed": []}
    write_json(args.output / "status.json", status)

    benchmark = Path(__file__).with_name("run_offline_benchmark.py")
    for number in range(1, args.runs + 1):
        run = args.output / f"run_{number:02d}"
        baseline_done = (run / "baseline" / "run_summary.json").is_file()
        human_done = (run / "humanslam" / "run_summary.json").is_file()
        if baseline_done and human_done:
            print(f"SKIP complete run {number:02d}", flush=True)
            status["completed"].append(number)
            write_json(args.output / "status.json", status)
            continue
        if baseline_done:
            mode = "humanslam"
        elif human_done:
            mode = "baseline"
        else:
            mode = "both"
        command = [
            sys.executable, str(benchmark),
            "--dataset", str(args.dataset),
            "--settings", str(args.settings),
            "--ground-truth", str(args.ground_truth),
            "--mode", mode, "--output", str(run),
        ]
        if run.exists() and any(run.iterdir()):
            command.append("--overwrite")
        if args.save_semantic_frames:
            command.append("--save-semantic-frames")
        print(f"START matched run {number:02d}/{args.runs}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            status["completed"].append(number)
            print(f"DONE matched run {number:02d}/{args.runs}", flush=True)
        else:
            status["failed"].append({"run": number, "return_code": result.returncode})
            print(f"FAILED matched run {number:02d}: rc={result.returncode}", flush=True)
        write_json(args.output / "status.json", status)

    status["state"] = "complete" if not status["failed"] else "complete_with_failures"
    status["finished"] = datetime.now().astimezone().isoformat()
    write_json(args.output / "status.json", status)
    print(f"EXPERIMENT {status['state']}", flush=True)


if __name__ == "__main__":
    main()
