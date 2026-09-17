#!/usr/bin/env bash

set -u

ROOT="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
RUNNER="$ROOT/tools/run_campus_bag_comparison.sh"
RESULTS="$ROOT/teleopbike_experiments/offline_results/campus_3x_campaign_2026-09-11"
SSD="/media/teleopbike/f9c35c30-fa52-4c39-9838-3cf714707170/rosbags"
BAG_2136="$SSD/campus_run_2026-09-10_21-36-25"
BAG_0919="$SSD/campus_run_2026-09-11_09-19-08"

mkdir -p "$RESULTS"

printf 'Campus repeat queue started: %s\n' "$(date --iso-8601=seconds)"
printf 'Playback rate: 0.25x for both methods\n'

while tmux has-session -t campus_0919_pair 2>/dev/null; do
    printf 'Waiting for campus_0919_pair to finish: %s\n' "$(date --iso-8601=seconds)"
    sleep 60
done

printf 'Rebuilding ORB-SLAM3 timestamp logger: %s\n' "$(date --iso-8601=seconds)"
if ! cmake --build /home/teleopbike/ORB_SLAM3/build --target ORB_SLAM3 -j2; then
    echo "ERROR: ORB-SLAM3 rebuild failed; repeat queue stopped." >&2
    exit 1
fi
printf 'ORB-SLAM3 timestamp logger rebuild complete: %s\n' "$(date --iso-8601=seconds)"

run_pair() {
    local label="$1"
    local bag="$2"
    local out="$RESULTS/$label"
    printf '\n=== %s started: %s ===\n' "$label" "$(date --iso-8601=seconds)"
    bash "$RUNNER" "$bag" "$out" both 0.25 0.25
    local status=$?
    printf '=== %s finished with status %d: %s ===\n' "$label" "$status" "$(date --iso-8601=seconds)"
    return "$status"
}

run_pair campus_2136_repeat_02 "$BAG_2136"
run_pair campus_2136_repeat_03 "$BAG_2136"
run_pair campus_2136_repeat_04_timestamp_replacement "$BAG_2136"
run_pair campus_0919_repeat_02 "$BAG_0919"
run_pair campus_0919_repeat_03 "$BAG_0919"
run_pair campus_0919_repeat_04_timestamp_replacement "$BAG_0919"

printf '\n=== campus_2136_repeat_02 HuMemSLAM recovery started: %s ===\n' "$(date --iso-8601=seconds)"
bash "$RUNNER" "$BAG_2136" "$RESULTS/campus_2136_repeat_02" humanslam 0.25 0.25
printf '=== campus_2136_repeat_02 HuMemSLAM recovery finished: %s ===\n' "$(date --iso-8601=seconds)"

printf '\nCampus GPS-evaluable 3x repeat queue complete: %s\n' "$(date --iso-8601=seconds)"
