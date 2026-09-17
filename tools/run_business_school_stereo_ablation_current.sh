#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
sequence="${project_dir}/test_scenarios/4seasons_business_school/recording_2021-01-07_13-12-23"
output_root="${project_dir}/test_results/dissertation_final/ablations/business_school_winter_stereo_current_20260821"

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u

export ROS_LOG_DIR=/tmp/humanslam_business_school_stereo_ablation_roslogs
export MPLCONFIGDIR=/tmp/humanslam_matplotlib
mkdir -p "${ROS_LOG_DIR}" "${MPLCONFIGDIR}" "${output_root}"

configs=(
  "scene_only 1 0 0"
  "object_only 0 1 0"
  "scene_object 1 1 0"
  "scene_text 1 0 1"
  "object_text 0 1 1"
  "full 1 1 1"
)

for spec in "${configs[@]}"; do
  read -r name scene object text <<< "${spec}"
  destination="${output_root}/${name}/run_01"
  if [[ -s "${destination}/humanslam/loop_retrieval_metrics.json" ]]; then
    echo "Skipping completed stereo ${name}"
    continue
  fi

  echo "Starting stereo-only Business School winter ${name}"
  python3 tools/run_4seasons_experiment.py \
    --sequence-dir "${sequence}" \
    --mode humanslam \
    --stereo-only \
    --results-root "${output_root}/${name}" \
    --run-name run_01 \
    --playback-rate 1.0 \
    --skip-videos \
    --skip-semantic-frames \
    $( [[ "${scene}" == 1 ]] && echo --use-scene || echo --no-use-scene ) \
    $( [[ "${object}" == 1 ]] && echo --use-object || echo --no-use-object ) \
    $( [[ "${text}" == 1 ]] && echo --use-text || echo --no-use-text )

  python3 tools/evaluate_4seasons_comparison.py \
    --sequence "${sequence}" \
    --results "${destination}" \
    --mode humanslam
done

echo "Current stereo-only Business School ablation complete: ${output_root}"
