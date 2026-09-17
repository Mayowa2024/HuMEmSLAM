#!/usr/bin/env python3
"""Extract synchronized GGA/NavSatFix observations directly from a ROS 2 bag.

The script reads sqlite3 rosbag files without replaying them. Run it from a
shell in which the matching ROS 2 distribution has been sourced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


QUALITY_LABELS = {
    0: "invalid",
    1: "gps_fix",
    2: "dgps",
    4: "rtk_fixed",
    5: "rtk_float",
    6: "estimated",
}


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def bag_files(bag: Path) -> list[Path]:
    metadata = bag / "metadata.yaml"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing rosbag metadata: {metadata}")
    info = yaml.safe_load(metadata.read_text(encoding="utf-8"))["rosbag2_bagfile_information"]
    files = [bag / name for name in info["relative_file_paths"]]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rosbag database file(s): {missing}")
    return files


def read_topic(files: list[Path], topic_name: str):
    records = []
    message_type = None
    for database in files:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        topic = connection.execute(
            "SELECT id, type FROM topics WHERE name = ?", (topic_name,)
        ).fetchone()
        if topic is not None:
            topic_id, type_name = topic
            message_type = message_type or get_message(type_name)
            for bag_timestamp, payload in connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,),
            ):
                records.append((int(bag_timestamp), deserialize_message(payload, message_type)))
        connection.close()
    records.sort(key=lambda item: item[0])
    return records


def nearest(records, timestamp: int):
    if not records:
        return None
    lo, hi = 0, len(records)
    while lo < hi:
        mid = (lo + hi) // 2
        if records[mid][0] < timestamp:
            lo = mid + 1
        else:
            hi = mid
    options = []
    if lo < len(records):
        options.append(records[lo])
    if lo > 0:
        options.append(records[lo - 1])
    return min(options, key=lambda item: abs(item[0] - timestamp))


def write_raw_gga(path: Path, records) -> None:
    fields = [
        "bag_timestamp_ns", "bag_time_s", "ros_timestamp_ns", "ros_time_s",
        "latitude", "longitude", "altitude_m", "gps_quality",
        "quality_label", "num_satellites", "hdop", "differential_age_s",
        "station_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bag_ns, msg in records:
            ros_ns = stamp_ns(msg.header)
            quality = int(msg.gps_qual)
            writer.writerow({
                "bag_timestamp_ns": bag_ns,
                "bag_time_s": f"{bag_ns / 1e9:.9f}",
                "ros_timestamp_ns": ros_ns,
                "ros_time_s": f"{ros_ns / 1e9:.9f}",
                "latitude": f"{float(msg.lat):.10f}",
                "longitude": f"{float(msg.lon):.10f}",
                "altitude_m": f"{float(msg.alt):.4f}",
                "gps_quality": quality,
                "quality_label": QUALITY_LABELS.get(quality, "other"),
                "num_satellites": int(msg.num_sats),
                "hdop": f"{float(msg.hdop):.4f}",
                "differential_age_s": f"{float(msg.diff_age):.4f}",
                "station_id": msg.station_id,
            })


def write_raw_fix(path: Path, records) -> None:
    fields = [
        "bag_timestamp_ns", "bag_time_s", "ros_timestamp_ns", "ros_time_s",
        "latitude", "longitude", "altitude_m", "status", "service",
        "covariance_xx", "covariance_yy", "covariance_zz", "covariance_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bag_ns, msg in records:
            ros_ns = stamp_ns(msg.header)
            covariance = list(msg.position_covariance)
            writer.writerow({
                "bag_timestamp_ns": bag_ns,
                "bag_time_s": f"{bag_ns / 1e9:.9f}",
                "ros_timestamp_ns": ros_ns,
                "ros_time_s": f"{ros_ns / 1e9:.9f}",
                "latitude": f"{float(msg.latitude):.10f}",
                "longitude": f"{float(msg.longitude):.10f}",
                "altitude_m": f"{float(msg.altitude):.4f}",
                "status": int(msg.status.status),
                "service": int(msg.status.service),
                "covariance_xx": f"{float(covariance[0]):.8f}",
                "covariance_yy": f"{float(covariance[4]):.8f}",
                "covariance_zz": f"{float(covariance[8]):.8f}",
                "covariance_type": int(msg.position_covariance_type),
            })


def write_combined(path: Path, gga, fixes) -> list[float]:
    fields = [
        "bag_timestamp_ns", "ros_timestamp_ns", "latitude", "longitude",
        "altitude_m", "gps_quality", "quality_label", "num_satellites",
        "hdop", "fix_status", "fix_covariance_type", "covariance_xx",
        "covariance_yy", "covariance_zz", "nearest_fix_gap_ms",
    ]
    gaps = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bag_ns, msg in gga:
            match = nearest(fixes, bag_ns)
            fix_ns, fix = match if match else (None, None)
            gap_ms = abs(fix_ns - bag_ns) / 1e6 if fix_ns is not None else None
            if gap_ms is not None:
                gaps.append(gap_ms)
            covariance = list(fix.position_covariance) if fix else [""] * 9
            quality = int(msg.gps_qual)
            writer.writerow({
                "bag_timestamp_ns": bag_ns,
                "ros_timestamp_ns": stamp_ns(msg.header),
                "latitude": f"{float(msg.lat):.10f}",
                "longitude": f"{float(msg.lon):.10f}",
                "altitude_m": f"{float(msg.alt):.4f}",
                "gps_quality": quality,
                "quality_label": QUALITY_LABELS.get(quality, "other"),
                "num_satellites": int(msg.num_sats),
                "hdop": f"{float(msg.hdop):.4f}",
                "fix_status": int(fix.status.status) if fix else "",
                "fix_covariance_type": int(fix.position_covariance_type) if fix else "",
                "covariance_xx": f"{float(covariance[0]):.8f}" if fix else "",
                "covariance_yy": f"{float(covariance[4]):.8f}" if fix else "",
                "covariance_zz": f"{float(covariance[8]):.8f}" if fix else "",
                "nearest_fix_gap_ms": f"{gap_ms:.6f}" if gap_ms is not None else "",
            })
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gga-topic", default="/gps/gga")
    parser.add_argument("--fix-topic", default="/gps/fix")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    files = bag_files(args.bag)
    gga = read_topic(files, args.gga_topic)
    fixes = read_topic(files, args.fix_topic)
    if not gga:
        raise RuntimeError(f"No messages found on {args.gga_topic}")

    write_raw_gga(args.output / "gga.csv", gga)
    write_raw_fix(args.output / "navsat_fix.csv", fixes)
    gaps = write_combined(args.output / "gps_combined.csv", gga, fixes)

    quality = Counter(int(msg.gps_qual) for _, msg in gga)
    timestamps = [timestamp for timestamp, _ in gga]
    intervals = [(b - a) / 1e9 for a, b in zip(timestamps, timestamps[1:])]
    summary = {
        "bag": str(args.bag.resolve()),
        "database_files": [str(path.resolve()) for path in files],
        "gga_topic": args.gga_topic,
        "fix_topic": args.fix_topic,
        "gga_messages": len(gga),
        "fix_messages": len(fixes),
        "quality_counts": {
            QUALITY_LABELS.get(code, str(code)): count for code, count in sorted(quality.items())
        },
        "gga_rate_hz_from_timestamps": 1.0 / statistics.mean(intervals) if intervals else None,
        "gga_median_interval_s": statistics.median(intervals) if intervals else None,
        "gga_max_interval_s": max(intervals) if intervals else None,
        "gga_fix_pair_gap_ms_mean": statistics.mean(gaps) if gaps else None,
        "gga_fix_pair_gap_ms_max": max(gaps) if gaps else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
