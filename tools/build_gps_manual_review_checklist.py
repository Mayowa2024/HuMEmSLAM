#!/usr/bin/env python3
"""Build a group-level checklist from the gps_manual frame manifest."""

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("teleopbike_experiments/gps_ground_truth/gps_manual")
rows = list(csv.DictReader((ROOT / "manifest.csv").open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[(row["dataset"], row["method"], row["region"])].append(row)

fields = [
    "dataset", "method", "region", "correspondence_type",
    "lap_1_keyframe_id", "lap_1_frame_id", "lap_2_keyframe_id", "lap_2_frame_id",
    "lap_3_keyframe_id", "lap_3_frame_id", "lap_1_to_2_distance_m",
    "lap_1_to_3_distance_m", "max_heading_difference_deg",
    "worst_gps_quality", "max_gps_bracket_interval_s", "comparison_image",
    "manual_visual_decision", "manual_notes",
]

quality_order = {"rtk_fixed": 3, "rtk_float": 2, "dgps": 1, "gps_fix": 1, "invalid": 0}
output = []
for (dataset, method, region), items in sorted(groups.items()):
    items.sort(key=lambda row: int(row["lap"]))
    base = items[0]
    distances = []
    for item in items[1:]:
        distances.append(math.hypot(
            float(item["east_m"]) - float(base["east_m"]),
            float(item["north_m"]) - float(base["north_m"]),
        ))
    worst = min((item["gps_quality"] for item in items), key=lambda q: quality_order.get(q, -1))
    row = {
        "dataset": dataset,
        "method": method,
        "region": region,
        "correspondence_type": base["correspondence_type"],
        "lap_1_keyframe_id": items[0]["keyframe_id"],
        "lap_1_frame_id": items[0]["frame_id"],
        "lap_2_keyframe_id": items[1]["keyframe_id"],
        "lap_2_frame_id": items[1]["frame_id"],
        "lap_3_keyframe_id": items[2]["keyframe_id"] if len(items) > 2 else "",
        "lap_3_frame_id": items[2]["frame_id"] if len(items) > 2 else "",
        "lap_1_to_2_distance_m": f"{distances[0]:.3f}",
        "lap_1_to_3_distance_m": f"{distances[1]:.3f}" if len(distances) > 1 else "",
        "max_heading_difference_deg": f"{max(float(item['heading_difference_from_lap1_deg']) for item in items):.2f}",
        "worst_gps_quality": worst,
        "max_gps_bracket_interval_s": f"{max(float(item['gps_bracket_interval_s']) for item in items):.3f}",
        "comparison_image": f"exact_correspondences/{dataset}/{method}/{region}_comparison.jpg",
        "manual_visual_decision": "",
        "manual_notes": "",
    }
    output.append(row)

with (ROOT / "review_checklist.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output)

print(f"Wrote {len(output)} correspondence groups to {ROOT / 'review_checklist.csv'}")
