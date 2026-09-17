#!/usr/bin/env bash

BAG="${1:-/media/teleopbike/f9c35c30-fa52-4c39-9838-3cf714707170/rosbags/campus_run_2026-09-10_21-36-25}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULT_ROOT="${2:-/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam/teleopbike_experiments/offline_results/campus_${STAMP}}"
VOCAB="/home/teleopbike/ORB_SLAM3/Vocabulary/ORBvoc.txt"
SETTINGS="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam/config/ZED2i_15Hz.yaml"
HUMAN_PARAMS="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/config/human_slam_params.yaml"
MODE="${3:-both}"
BASELINE_RATE="${4:-0.5}"
HUMANSLAM_RATE="${5:-0.25}"

LEFT="/zed/zed_node/left/color/rect/image"
RIGHT="/zed/zed_node/right/color/rect/image"

ORB_PID=""
HUMAN_PID=""
PLAYER_PID=""

source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash

set -u

stop_pid() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
        kill -INT -- "-$pid" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 -- "-$pid" 2>/dev/null || return 0
            sleep 1
        done
        kill -TERM -- "-$pid" 2>/dev/null || true
    fi
}

cleanup() {
    stop_pid "$PLAYER_PID"
    stop_pid "$ORB_PID"
    stop_pid "$HUMAN_PID"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

wait_for_stereo_subscription() {
    for _ in {1..90}; do
        local left_count right_count
        left_count="$(ros2 topic info "$LEFT" 2>/dev/null | awk '/Subscription count/{print $3}')"
        right_count="$(ros2 topic info "$RIGHT" 2>/dev/null | awk '/Subscription count/{print $3}')"
        if [[ "${left_count:-0}" -ge 1 && "${right_count:-0}" -ge 1 ]]; then
            return 0
        fi
        kill -0 "$ORB_PID" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

wait_for_first_processed_frame() {
    local latency_file="$1"
    for _ in {1..300}; do
        if [[ -f "$latency_file" ]] && [[ "$(wc -l < "$latency_file")" -ge 2 ]]; then
            return 0
        fi
        kill -0 "$PLAYER_PID" 2>/dev/null || return 1
        kill -0 "$ORB_PID" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

run_variant() {
    local name="$1"
    local assisted="$2"
    local replay_rate="$3"
    local out="$RESULT_ROOT/$name"
    mkdir -p "$out/logs"

    printf '\n=== Starting %s ===\n' "$name"

    if [[ "$assisted" == "true" ]]; then
        setsid ros2 run slam human_slam_node --ros-args \
            --params-file "$HUMAN_PARAMS" \
            -p latency_output:="$out/human_latency.csv" \
            -p candidate_output:="$out/human_candidates.csv" \
            >"$out/logs/humanslam.log" 2>&1 &
        HUMAN_PID=$!
        sleep 5
        if ! kill -0 "$HUMAN_PID" 2>/dev/null; then
            echo "ERROR: HuMemSLAM failed during startup. See $out/logs/humanslam.log" >&2
            return 1
        fi
    fi

    setsid ros2 run orbslam3_zed_stereo zed_stereo_node --ros-args \
        -p voc_file:="$VOCAB" \
        -p settings_file:="$SETTINGS" \
        -p use_imu:=false \
        -p semantic_assistance_enabled:="$assisted" \
        -p trajectory_output:="$out/trajectory.txt" \
        -p latency_output:="$out/orb_latency.csv" \
        -p tracked_features_output:="$out/tracked_features.csv" \
        -p event_output:="$out/event.csv" \
        -p keyframe_output:="$out/keyframes.csv" \
        >"$out/logs/orbslam3.log" 2>&1 &
    ORB_PID=$!

    if ! wait_for_stereo_subscription; then
        echo "ERROR: ORB-SLAM3 did not subscribe to both stereo topics. See $out/logs/orbslam3.log" >&2
        return 1
    fi

    sleep 3

    setsid ros2 bag play "$BAG" \
        --topics "$LEFT" "$RIGHT" \
        --rate "$replay_rate" \
        --disable-keyboard-controls \
        >"$out/logs/bag_play.log" 2>&1 &
    PLAYER_PID=$!

    if ! wait_for_first_processed_frame "$out/orb_latency.csv"; then
        echo "ERROR: No stereo frame was processed within 300 seconds; aborting this attempt." >&2
        stop_pid "$PLAYER_PID"
        PLAYER_PID=""
        stop_pid "$ORB_PID"
        ORB_PID=""
        stop_pid "$HUMAN_PID"
        HUMAN_PID=""
        return 1
    fi

    wait "$PLAYER_PID"
    PLAYER_PID=""

    sleep 15
    stop_pid "$ORB_PID"
    ORB_PID=""
    stop_pid "$HUMAN_PID"
    HUMAN_PID=""

    printf '=== Finished %s: %s ===\n' "$name" "$out"
}

if [[ ! -f "$BAG/metadata.yaml" ]]; then
    echo "ERROR: Invalid or unfinished rosbag: $BAG" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
printf 'Bag: %s\nResults: %s\n' "$BAG" "$RESULT_ROOT"

case "$MODE" in
    both)
        run_variant baseline false "$BASELINE_RATE" || exit $?
        sleep 5
        run_variant humanslam true "$HUMANSLAM_RATE" || exit $?
        ;;
    baseline)
        run_variant baseline false "$BASELINE_RATE" || exit $?
        ;;
    humanslam)
        run_variant humanslam true "$HUMANSLAM_RATE" || exit $?
        ;;
    *)
        echo "ERROR: mode must be one of: both, baseline, humanslam" >&2
        exit 2
        ;;
esac

trap - INT TERM EXIT
printf '\nComparison complete: %s\n' "$RESULT_ROOT"
