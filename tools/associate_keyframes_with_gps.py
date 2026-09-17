#!/usr/bin/env python3
"""Convert extracted GNSS to local ENU and optionally associate ORB keyframes."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
QUALITY_ORDER = {"rtk_fixed": 3, "rtk_float": 2, "dgps": 1, "gps_fix": 1, "invalid": 0}


def ecef(latitude_deg: float, longitude_deg: float, altitude_m: float):
    lat, lon = math.radians(latitude_deg), math.radians(longitude_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return (
        (n + altitude_m) * cos_lat * cos_lon,
        (n + altitude_m) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + altitude_m) * sin_lat,
    )


def enu(point, origin_ecef, origin_lat_deg, origin_lon_deg):
    dx, dy, dz = (point[i] - origin_ecef[i] for i in range(3))
    lat, lon = math.radians(origin_lat_deg), math.radians(origin_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def load_gps(path: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    valid = [r for r in rows if r["quality_label"] != "invalid"]
    if not valid:
        raise RuntimeError("No valid GPS observations")
    origin = valid[0]
    origin_lat = float(origin["latitude"])
    origin_lon = float(origin["longitude"])
    origin_alt = float(origin["altitude_m"])
    origin_xyz = ecef(origin_lat, origin_lon, origin_alt)
    for row in rows:
        xyz = ecef(float(row["latitude"]), float(row["longitude"]), float(row["altitude_m"]))
        row["east_m"], row["north_m"], row["up_m"] = enu(
            xyz, origin_xyz, origin_lat, origin_lon
        )
        row["timestamp_ns"] = int(row["ros_timestamp_ns"])
    rows.sort(key=lambda r: r["timestamp_ns"])
    return rows, (origin_lat, origin_lon, origin_alt)


def write_enu(path: Path, rows, origin):
    fields = list(rows[0].keys())
    fields.remove("timestamp_ns")
    fields += ["east_m", "north_m", "up_m"] if "east_m" not in fields else []
    fields = [f for f in fields if f not in ("east_m", "north_m", "up_m")]
    insert_at = fields.index("altitude_m") + 1
    fields[insert_at:insert_at] = ["east_m", "north_m", "up_m"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("east_m", "north_m", "up_m"):
                out[key] = f"{float(row[key]):.4f}"
            writer.writerow(out)
    (path.parent / "enu_origin.txt").write_text(
        f"latitude={origin[0]:.10f}\nlongitude={origin[1]:.10f}\naltitude_m={origin[2]:.4f}\n",
        encoding="utf-8",
    )


def write_route_plot(path: Path, rows) -> None:
    import matplotlib.pyplot as plt

    colours = {"rtk_fixed": "#198754", "rtk_float": "#F9C909", "dgps": "#D97706"}
    east = [float(row["east_m"]) for row in rows]
    north = [float(row["north_m"]) for row in rows]
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    axis.plot(east, north, color="#AAAAAA", linewidth=1.2, zorder=1)
    for quality in ("dgps", "rtk_float", "rtk_fixed"):
        selected = [row for row in rows if row["quality_label"] == quality]
        if selected:
            axis.scatter(
                [float(row["east_m"]) for row in selected],
                [float(row["north_m"]) for row in selected],
                s=24, color=colours[quality], label=quality.replace("_", " ").title(),
                zorder=2,
            )
    axis.scatter(east[0], north[0], marker="o", s=90, color="black", label="Start", zorder=3)
    axis.scatter(east[-1], north[-1], marker="X", s=100, color="#555555", label="End", zorder=3)
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, color="#E5E5E5", linewidth=0.7)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180, transparent=False)
    plt.close(fig)


def combined_quality(a: str, b: str) -> str:
    return a if QUALITY_ORDER.get(a, -1) <= QUALITY_ORDER.get(b, -1) else b


def associate(keyframes_path: Path, gps, output: Path, max_interval_s: float):
    keyframes = list(csv.DictReader(keyframes_path.open(newline="", encoding="utf-8")))
    raw_times = [row["dataset_time"] for row in keyframes]
    if len(set(raw_times)) < max(2, len(raw_times) // 10):
        raise RuntimeError(
            "Keyframe timestamps have insufficient precision. Re-run ORB-SLAM3 "
            "with the corrected EventLogger; association was not generated."
        )
    gps_times = [row["timestamp_ns"] for row in gps]
    fields = list(keyframes[0].keys()) + [
        "gps_valid", "east_m", "north_m", "up_m", "gps_quality",
        "gps_before_gap_s", "gps_after_gap_s", "gps_bracket_interval_s",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for keyframe in keyframes:
            timestamp_ns = round(float(keyframe["dataset_time"]) * 1e9)
            idx = bisect.bisect_right(gps_times, timestamp_ns)
            out = dict(keyframe)
            out.update({k: "" for k in fields if k not in out})
            out["gps_valid"] = 0
            if idx == 0 or idx == len(gps):
                writer.writerow(out)
                continue
            before, after = gps[idx - 1], gps[idx]
            interval_s = (after["timestamp_ns"] - before["timestamp_ns"]) / 1e9
            before_gap = (timestamp_ns - before["timestamp_ns"]) / 1e9
            after_gap = (after["timestamp_ns"] - timestamp_ns) / 1e9
            out["gps_before_gap_s"] = f"{before_gap:.6f}"
            out["gps_after_gap_s"] = f"{after_gap:.6f}"
            out["gps_bracket_interval_s"] = f"{interval_s:.6f}"
            out["gps_quality"] = combined_quality(before["quality_label"], after["quality_label"])
            if interval_s <= max_interval_s and before_gap >= 0 and after_gap >= 0:
                alpha = before_gap / interval_s if interval_s else 0.0
                for coordinate in ("east_m", "north_m", "up_m"):
                    value = float(before[coordinate]) + alpha * (
                        float(after[coordinate]) - float(before[coordinate])
                    )
                    out[coordinate] = f"{value:.4f}"
                out["gps_valid"] = 1
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps", required=True, type=Path, help="gps_combined.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--max-bracket-interval", type=float, default=4.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gps, origin = load_gps(args.gps)
    write_enu(args.output_dir / "gps_enu.csv", gps, origin)
    write_route_plot(args.output_dir / "gps_route_quality.png", gps)
    print(f"Wrote {len(gps)} ENU observations to {args.output_dir / 'gps_enu.csv'}")
    if args.keyframes:
        associate(
            args.keyframes, gps, args.output_dir / "keyframes_with_gps.csv",
            args.max_bracket_interval,
        )
        print(f"Wrote keyframe associations to {args.output_dir / 'keyframes_with_gps.csv'}")


if __name__ == "__main__":
    main()
