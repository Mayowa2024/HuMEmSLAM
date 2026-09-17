#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
sequence="${project_dir}/test_scenarios/4seasons_business_school/recording_2021-01-07_13-12-23"
output_root="${project_dir}/test_results/dissertation_final/ablations/business_school_winter_post_worker_20260820"

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
export ROS_LOG_DIR=/tmp/humanslam_business_school_ablation_post_worker_roslogs
mkdir -p "${ROS_LOG_DIR}" "${output_root}"

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
  for repetition in 1 2 3; do
    run=$(printf 'run_%02d' "${repetition}")
    destination="${output_root}/${name}/${run}"
    if [[ -s "${destination}/humanslam/loop_retrieval_metrics.json" ]]; then
      echo "Skipping completed ${name} ${run}"
      continue
    fi
    echo "Starting ${name} ${run}/run_03"
    python3 tools/run_4seasons_experiment.py \
      --sequence-dir "${sequence}" \
      --mode humanslam \
      --results-root "${output_root}/${name}" \
      --run-name "${run}" \
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
done

echo "Post-worker Business School ablation complete: ${output_root}"
