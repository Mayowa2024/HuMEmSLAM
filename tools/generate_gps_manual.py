#!/usr/bin/env python3
"""Generate keyframe-only manual GPS correspondence review assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


LEFT_TOPIC = "/zed/zed_node/left/color/rect/image"
DATASETS = {
    "campus_run_2026-09-10_21-36-25": {
        "bag": "/media/teleopbike/f9c35c30-fa52-4c39-9838-3cf714707170/rosbags/campus_run_2026-09-10_21-36-25",

        "laps": [(26, 104), (104, 176)],
    },
    "campus_run_2026-09-11_09-19-08": {
        "bag": "/media/teleopbike/f9c35c30-fa52-4c39-9838-3cf714707170/rosbags/campus_run_2026-09-11_09-19-08",

        "laps": [(22, 112), (112, 188), (211, 238)],
    },
}


def read_csv(path: Path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def xy(row):
    return float(row["east_m"]), float(row["north_m"])


def distance(a, b):
    ax, ay = xy(a)
    bx, by = xy(b)
    return math.hypot(ax - bx, ay - by)


def heading(rows, index):
    lo, hi = max(0, index - 1), min(len(rows) - 1, index + 1)
    x0, y0 = xy(rows[lo])
    x1, y1 = xy(rows[hi])
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def angle_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def nearest_in_lap(rows, ref_index, bounds):
    ref = rows[ref_index]
    ref_heading = heading(rows, ref_index)
    options = []
    for idx in range(bounds[0], min(bounds[1], len(rows))):
        d = distance(ref, rows[idx])
        hd = angle_difference(ref_heading, heading(rows, idx))


        if hd <= 65.0:
            options.append((d + 0.02 * hd, d, hd, idx))
    return min(options) if options else None


def select_regions(rows, laps, target=6):
    first_start, first_end = laps[0]
    candidates = []
    for ref_idx in range(first_start + 2, min(first_end - 2, len(rows))):
        matches = [(0.0, 0.0, ref_idx)]
        for bounds in laps[1:]:
            found = nearest_in_lap(rows, ref_idx, bounds)
            if found and found[1] <= 10.0:
                matches.append((found[1], found[2], found[3]))
        if len(matches) >= 2:
            candidates.append((max(v[0] for v in matches), ref_idx, matches))



    candidates.sort(key=lambda z: (-len(z[2]), z[0]))
    chosen = []
    for item in candidates:
        ref = rows[item[1]]
        if all(distance(ref, rows[old[1]]) >= 24.0 for old in chosen):
            chosen.append(item)
        if len(chosen) == target:
            break

    if len(chosen) < target:
        for item in candidates:
            if item in chosen:
                continue
            ref = rows[item[1]]
            if all(distance(ref, rows[old[1]]) >= 14.0 for old in chosen):
                chosen.append(item)
            if len(chosen) == target:
                break
    return sorted(chosen, key=lambda z: z[1])


def nearest_keyframe(keyframes, gps_row, timestamp_ns):
    valid = [r for r in keyframes if r.get("gps_valid") == "1"]
    nearby = []
    for row in valid:
        time_gap_s = abs(float(row["dataset_time"]) * 1e9 - timestamp_ns) / 1e9
        spatial_gap_m = math.hypot(
            float(row["east_m"]) - float(gps_row["east_m"]),
            float(row["north_m"]) - float(gps_row["north_m"]),
        )
        if time_gap_s <= 5.0 and spatial_gap_m <= 10.0:
            nearby.append((spatial_gap_m, time_gap_s, row))
    if not nearby:
        return None
    return min(nearby, key=lambda item: (item[0], item[1]))[2]


def bag_files(bag: Path):
    info = yaml.safe_load((bag / "metadata.yaml").read_text(encoding="utf-8"))
    names = info["rosbag2_bagfile_information"]["relative_file_paths"]
    return [bag / name for name in names]


def extract_nearest_images(bag: Path, targets):
    """Return target-key -> (message timestamp, BGR image)."""
    target_ns = {key: round(float(row["dataset_time"]) * 1e9) for key, row in targets.items()}
    best = {key: (10**30, None, None) for key in targets}
    bridge = CvBridge()
    msg_type = None
    for database in bag_files(bag):
        con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        topic = con.execute("SELECT id,type FROM topics WHERE name=?", (LEFT_TOPIC,)).fetchone()
        if topic is None:
            con.close()
            continue
        topic_id, type_name = topic
        msg_type = msg_type or get_message(type_name)
        low, high = min(target_ns.values()) - 100_000_000, max(target_ns.values()) + 100_000_000
        for stamp, payload in con.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (topic_id, low, high),
        ):
            for key, wanted in target_ns.items():
                gap = abs(int(stamp) - wanted)
                if gap < best[key][0]:
                    best[key] = (gap, int(stamp), payload)
        con.close()
    output = {}
    for key, (gap, stamp, payload) in best.items():
        if payload is None or gap > 100_000_000:
            raise RuntimeError(f"No camera frame within 100 ms for {key}")
        msg = deserialize_message(payload, msg_type)
        output[key] = (stamp, bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), gap / 1e6)
    return output


def fit_panel(image, width=560, height=315):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 20, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def orient_for_review(image):
    """Rotate exported copies because the campus ZED was mounted inverted."""
    return cv2.rotate(image, cv2.ROTATE_180)


def labelled_panel(image, label):
    panel = fit_panel(image)
    canvas = np.full((365, 560, 3), 20, np.uint8)
    canvas[:315] = panel
    cv2.putText(canvas, label, (16, 347), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "teleopbike_experiments/gps_ground_truth/gps_manual"))
    args = parser.parse_args()
    root = args.root
    manifest_rows = []

    for dataset, config in DATASETS.items():
        association_root = root / "associations" / dataset
        gps = read_csv(association_root / "baseline" / "gps_enu.csv")
        regions = select_regions(gps, config["laps"])
        dataset_dir = root / "exact_correspondences" / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for method in ("baseline", "humanslam"):
            keyframes = read_csv(association_root / method / "keyframes_with_gps.csv")
            targets = {}
            rows_for_method = []
            for region_number, (_, ref_idx, matches) in enumerate(regions, 1):
                region_targets = {}
                region_rows = []
                for lap_number, (gps_dist, heading_diff, gps_idx) in enumerate(matches, 1):
                    gps_row = gps[gps_idx]
                    gps_ns = int(gps_row["ros_timestamp_ns"])
                    keyframe = nearest_keyframe(keyframes, gps_row, gps_ns)
                    if keyframe is None:
                        region_targets = {}
                        region_rows = []
                        break
                    key = f"region_{region_number:02d}_lap_{lap_number}"
                    region_targets[key] = keyframe
                    region_rows.append({
                        "dataset": dataset,
                        "method": method,
                        "region": f"region_{region_number:02d}",
                        "lap": lap_number,
                        "correspondence_type": f"1-to-1-to-1" if len(matches) == 3 else "1-to-1",
                        "keyframe_id": keyframe["keyframe_id"],
                        "frame_id": keyframe["frame_id"],
                        "dataset_time": keyframe["dataset_time"],
                        "east_m": keyframe["east_m"],
                        "north_m": keyframe["north_m"],
                        "up_m": keyframe["up_m"],
                        "gps_quality": keyframe["gps_quality"],
                        "gps_bracket_interval_s": keyframe["gps_bracket_interval_s"],
                        "distance_from_lap1_gps_m": f"{gps_dist:.3f}",
                        "heading_difference_from_lap1_deg": f"{heading_diff:.2f}",
                    })
                targets.update(region_targets)
                rows_for_method.extend(region_rows)

            images = extract_nearest_images(Path(config["bag"]), targets)
            method_dir = dataset_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            grouped = {}
            for row in rows_for_method:
                key = f"{row['region']}_lap_{row['lap']}"
                stamp, image, gap_ms = images[key]
                image = orient_for_review(image)
                filename = f"{key}_kf_{row['keyframe_id']}_frame_{row['frame_id']}.jpg"
                cv2.imwrite(str(method_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                row["image_file"] = str((method_dir / filename).relative_to(root))
                row["image_timestamp_gap_ms"] = f"{gap_ms:.4f}"
                row["review_rotation_deg"] = "180"
                grouped.setdefault(row["region"], []).append((row, image))
                manifest_rows.append(row)

            contact_rows = []
            for region, items in grouped.items():
                items.sort(key=lambda pair: int(pair[0]["lap"]))
                panels = [labelled_panel(img, f"{region} | lap {row['lap']} | KF {row['keyframe_id']}")
                          for row, img in items]
                while len(panels) < 3:
                    panels.append(np.full_like(panels[0], 20))
                strip = np.hstack(panels)
                cv2.imwrite(str(method_dir / f"{region}_comparison.jpg"), strip,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                contact_rows.append(strip)
            cv2.imwrite(str(method_dir / "contact_sheet.jpg"), np.vstack(contact_rows),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])

    fields = list(manifest_rows[0].keys())
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "experiment": "gps_manual",
        "scope": "keyframes only",
        "stage": "exact_correspondences",
        "datasets": list(DATASETS),
        "selection": "GPS-supported 1-to-1 and, where available, 1-to-1-to-1 lap correspondences",
        "review_image_rotation_deg": 180,
        "rows": len(manifest_rows),
        "ten_metre_context": "deferred until exact correspondences are manually approved",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# gps_manual\n\n"
        "This is the keyframe-only manual audit of GPS correspondence. Open the two "
        "`contact_sheet.jpg` files under `exact_correspondences/<dataset>/<method>/`. "
        "Each row is a qualitative route region; columns are separate laps. GPS "
        "identifies candidates, but visual inspection determines whether a correspondence "
        "is accepted. The ±10 m context stage is intentionally deferred until these exact "
        "matches are approved. See `manifest.csv` for IDs, timestamps, coordinates, GPS "
        "quality, interpolation intervals and spatial/heading differences.\n",
        encoding="utf-8",
    )
    print(f"Wrote gps_manual assets to {root} ({len(manifest_rows)} manifest rows)")


if __name__ == "__main__":
    main()
