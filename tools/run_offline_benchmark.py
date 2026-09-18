#!/usr/bin/env python3
"""Run reproducible ORB-SLAM3/HuMemSLAM offline experiments and evaluate them."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from monitor_hardware import HardwareMonitor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORB_SLAM3 = Path(
    os.environ.get("ORB_SLAM3_ROOT", ROOT / "external/ORB_SLAM3")
)
DEFAULT_ORB_WRAPPER = Path(
    os.environ.get(
        "ORB_SLAM3_ROS2_WRAPPER",
        ROOT / "external/orbslam3_ros2_ws/src/orbslam3_zed_stereo",
    )
)
DEFAULT_VOCABULARY = Path(
    os.environ.get(
        "ORB_SLAM3_VOCABULARY",
        DEFAULT_ORB_SLAM3 / "Vocabulary/ORBvoc.txt",
    )
)
DEFAULT_CONFIG = Path(
    os.environ.get(
        "HUMEMSLAM_CONFIG",
        ROOT.parent / "config/human_slam_params.yaml",
    )
)


def arguments():
    parser = argparse.ArgumentParser(
        description="Run baseline, HuMemSLAM, or a matched pair and create metrics automatically."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Experiment root; mode subdirectories are created")
    parser.add_argument("--mode", choices=("baseline", "humanslam", "both"),
                        default="both")
    parser.add_argument("--ground-truth", type=Path,
                        help="KITTI 3x4 poses; enables APE and retrieval labels")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--human-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--left-dir", default="image_0")
    parser.add_argument("--right-dir", default="image_1")
    parser.add_argument("--times-file", default="times.txt")
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--semantic-threshold", type=float, default=0.70)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument(
        "--ocr-text-det-limit-side-len", type=int, default=640,
        help="Override PaddleOCR detector maximum input side for this run.",
    )
    parser.add_argument("--use-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-object", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--position-threshold-m", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--min-frame-separation", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-semantic-frames", action="store_true",
        help="Save per-frame HuMemSLAM debug images (large; disabled by default).",
    )
    return parser.parse_args()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def require(path: Path, label: str):
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, capture_output=True, timeout=5, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def numeric_column(path: Path, column: str) -> list[float]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        values = []
        for row in csv.DictReader(stream):
            text = row.get(column, "").strip()
            if text:
                try:
                    value = float(text)
                    if value >= 0:
                        values.append(value)
                except ValueError:
                    pass
        return values


def stats(values: list[float]):
    if not values:
        return None
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "p95": p95,
        "minimum": min(values), "maximum": max(values),
    }


def run_logged(command: list[str], log: Path, env=None, monitor_interval=5.0) -> int:
    monitor = HardwareMonitor(log.parent, monitor_interval)
    monitor.start()
    code = 1
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                stream.write(line)
            code = process.wait()
            return code
        finally:
            monitor.stop(code)


def evaluate(run_dir: Path, args, human: bool):
    summary = {
        "mode": "humanslam" if human else "baseline",
        "orb_tracking_ms": stats(numeric_column(run_dir / "orb_latency.csv", "track_stereo_ms")),
        "human_total_ms": stats(numeric_column(run_dir / "human_latency.csv", "total_ms")),
        "artifacts": {path.name: path.stat().st_size for path in run_dir.iterdir()
                      if path.is_file()},
    }
    if args.ground_truth:
        retrieval_command = [
            sys.executable, str(ROOT / "tools/evaluate_loop_retrieval.py"),
            "--events", str(run_dir / "orb_events.csv"),
            "--keyframes", str(run_dir / "orb_keyframes.csv"),
            "--ground-truth", str(args.ground_truth),
            "--times", str(args.dataset / args.times_file),
            "--position-threshold-m", str(args.position_threshold_m),
            "--rotation-threshold-deg", str(args.rotation_threshold_deg),
            "--min-frame-separation", str(args.min_frame_separation),
            "--output", str(run_dir / "loop_retrieval_metrics.json"),
        ]
        subprocess.run(retrieval_command, check=True)
        relocalisation_command = [
            sys.executable, str(ROOT / "tools/evaluate_relocalisation.py"),
            "--events", str(run_dir / "orb_events.csv"),
            "--keyframes", str(run_dir / "orb_keyframes.csv"),
            "--frame-index", str(run_dir / "trajectory_frame_ids.csv"),
            "--ground-truth", str(args.ground_truth),
            "--position-threshold-m", str(args.position_threshold_m),
            "--rotation-threshold-deg", str(args.rotation_threshold_deg),
            "--output", str(run_dir / "relocalisation_metrics.json"),
        ]
        subprocess.run(relocalisation_command, check=True)
        trajectory_evaluation = subprocess.run([
            sys.executable, str(ROOT / "tools/evaluate_indexed_trajectory.py"),
            "--trajectory", str(run_dir / "trajectory_kitti.txt"),
            "--frame-index", str(run_dir / "trajectory_frame_ids.csv"),
            "--ground-truth", str(args.ground_truth),
            "--times", str(args.dataset / args.times_file),
            "--output", str(run_dir / "trajectory_evaluation"),
            "--expected-start", str(args.start_frame),
            "--expected-end", str(args.end_frame),
        ], text=True, capture_output=True)
        (run_dir / "ape_metrics.txt").write_text(
            trajectory_evaluation.stdout + trajectory_evaluation.stderr,
            encoding="utf-8")
        summary["ape_exit_code"] = trajectory_evaluation.returncode
        evaluation_summary = run_dir / "trajectory_evaluation" / "summary.json"
        if evaluation_summary.exists():
            summary["trajectory_evaluation"] = json.loads(
                evaluation_summary.read_text(encoding="utf-8"))
        relocalisation_summary = run_dir / "relocalisation_metrics.json"
        if relocalisation_summary.exists():
            summary["relocalisation"] = json.loads(
                relocalisation_summary.read_text(encoding="utf-8"))
    summary["artifacts"] = {
        path.name: path.stat().st_size for path in run_dir.iterdir()
        if path.is_file()
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_mode(args, human: bool):
    name = "humanslam" if human else "baseline"
    run_dir = args.output.resolve() / name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Refusing non-empty output: {run_dir} (use --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "hostname": platform.node(), "mode": name,
        "dataset": str(args.dataset.resolve()), "settings": str(args.settings.resolve()),
        "ground_truth": str(args.ground_truth.resolve()) if args.ground_truth else None,
        "start_frame": args.start_frame, "end_frame": args.end_frame,
        "playback_rate": args.playback_rate,
        "semantic_threshold": args.semantic_threshold,
        "ocr_text_det_limit_side_len": args.ocr_text_det_limit_side_len,
        "save_semantic_frames": args.save_semantic_frames,
        "layers": {"scene": args.use_scene, "object": args.use_object, "text": args.use_text},
        "git_commits": {
            "humanslam": git_commit(ROOT),
            "orbslam3": git_commit(DEFAULT_ORB_SLAM3),
            "orbslam3_ros2_wrapper": git_commit(DEFAULT_ORB_WRAPPER),
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.settings, run_dir / "orbslam_settings_snapshot.yaml")
    shutil.copy2(args.human_config, run_dir / "humanslam_config_snapshot.yaml")
    command = [
        "ros2", "launch", "slam", "offline_benchmark.launch.py",
        f"dataset_path:={args.dataset.resolve()}", f"settings_file:={args.settings.resolve()}",
        f"vocabulary_file:={args.vocabulary.resolve()}",
        f"human_slam_config:={args.human_config.resolve()}",
        f"use_human_slam:={bool_text(human)}", f"results_dir:={run_dir}",
        f"left_dir:={args.left_dir}", f"right_dir:={args.right_dir}",
        f"times_file:={args.times_file}", f"playback_rate:={args.playback_rate}",
        f"start_frame:={args.start_frame}", f"end_frame:={args.end_frame}",
        f"semantic_threshold:={args.semantic_threshold}",
        f"candidate_response_count:={args.candidate_count}",
        f"ocr_text_det_limit_side_len:={args.ocr_text_det_limit_side_len}",
        f"use_scene:={bool_text(args.use_scene)}", f"use_object:={bool_text(args.use_object)}",
        f"use_text:={bool_text(args.use_text)}",
    ]
    if human and args.save_semantic_frames:
        command.append(f"debug_output_dir:={run_dir / 'semantic_frames'}")
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    code = run_logged(command, run_dir / "console.log", env=os.environ.copy())
    if code:
        raise SystemExit(f"{name} launch failed with code {code}; see {run_dir/'console.log'}")
    for required in ("trajectory_kitti.txt", "trajectory_frame_ids.csv",
                     "orb_events.csv", "orb_keyframes.csv",
                     "orb_latency.csv", "orb_tracked_features.csv"):
        require(run_dir / required, required)
    evaluate(run_dir, args, human)


def main():
    args = arguments()
    for path, label in ((args.dataset, "dataset"), (args.settings, "settings"),
                        (args.vocabulary, "vocabulary")):
        require(path, label)
    if args.ground_truth:
        require(args.ground_truth, "ground truth")
    modes = (False, True) if args.mode == "both" else (args.mode == "humanslam",)
    for human in modes:
        run_mode(args, human)
    print(f"Completed experiment: {args.output.resolve()}")


if __name__ == "__main__":
    main()
