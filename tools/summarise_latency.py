#!/usr/bin/env python3
"""Summarise HuMemSLAM component and ORB bridge latency CSV files."""

import argparse
from collections import Counter
import csv
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values):
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def numeric_column(path, name, predicate=lambda row: True):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            float(row[name])
            for row in csv.DictReader(stream)
            if row.get(name) not in (None, "") and predicate(row)
        ]


def categorical_counts(path, name, predicate=lambda row: True):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        counts = Counter(
            row[name]
            for row in csv.DictReader(stream)
            if row.get(name) not in (None, "") and predicate(row)
        )
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-csv", required=True, type=Path)
    parser.add_argument("--orb-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "human_all": stats(numeric_column(args.human_csv, "total_ms")),
        "human_recovery_queries": stats(
            numeric_column(
                args.human_csv,
                "total_ms",
                lambda row: row.get("is_keyframe") == "0",
            )
        ),
        "scene": stats(numeric_column(args.human_csv, "scene_ms")),
        "object_ocr": stats(numeric_column(args.human_csv, "object_ocr_ms")),
        "ranking": stats(numeric_column(args.human_csv, "ranking_ms")),
        "orb_to_semantic_response": stats(
            numeric_column(args.orb_csv, "response_latency_ms")
        ),
        "orb_track_stereo": stats(
            numeric_column(args.orb_csv, "track_stereo_ms")
        ),
        "semantic_recovery": stats(
            numeric_column(args.orb_csv, "recovery_latency_ms")
        ),
        "orb_event_counts": categorical_counts(args.orb_csv, "event"),
        "orb_tracking_state_frame_counts": categorical_counts(
            args.orb_csv,
            "tracking_state_name",
            lambda row: row.get("event") == "track",
        ),
        "orb_map_event_counts": categorical_counts(
            args.orb_csv,
            "event",
            lambda row: row.get("event") in {
                "MAP_ACTIVE", "NEW_MAP_OBSERVED", "MAP_SWITCHED"
            },
        ),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
