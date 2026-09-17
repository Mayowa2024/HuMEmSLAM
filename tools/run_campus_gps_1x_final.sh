#!/usr/bin/env bash

set -eu

ROOT="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
RUNNER="$ROOT/tools/run_campus_bag_comparison.sh"
SSD="/media/teleopbike/f9c35c30-fa52-4c39-9838-3cf714707170/rosbags"
BAG_2136="$SSD/campus_run_2026-09-10_21-36-25"
BAG_0919="$SSD/campus_run_2026-09-11_09-19-08"
PAIR_2136="$ROOT/teleopbike_experiments/offline_results/campus_3x_campaign_2026-09-11/campus_2136_repeat_02"
PAIR_0919="$ROOT/teleopbike_experiments/offline_results/campus_gps_1x_final_2026-09-11/campus_0919"

printf 'Campus GPS 1x final campaign started: %s\n' "$(date --iso-8601=seconds)"

printf '\n[1/3] Running 21:36 HuMemSLAM (existing baseline preserved)\n'
bash "$RUNNER" "$BAG_2136" "$PAIR_2136" humanslam 0.25 0.25

printf '\n[2-3/3] Running 09:19 baseline then HuMemSLAM\n'
bash "$RUNNER" "$BAG_0919" "$PAIR_0919" both 0.25 0.25

printf '\nCampus GPS 1x final campaign complete: %s\n' "$(date --iso-8601=seconds)"
