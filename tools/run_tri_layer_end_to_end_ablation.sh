#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
output_root="${project_dir}/test_results/dissertation_final/ablations/tri_layer_end_to_end_20260816"
kitti_dataset="${project_dir}/test_scenarios/final_kitti06/K06-4_blur15_dark80"
kitti_settings="/home/teleopbike/ORB_SLAM3/Examples/Stereo/KITTI04-12.yaml"
kitti_gt="/home/teleopbike/Documents/slam_experiments/datasets/KITTI/dataset/poses/06.txt"
winter_sequence="${project_dir}/test_scenarios/4seasons_business_school/recording_2021-01-07_13-12-23"

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
unset HUMANSLAM_ALIAS_MICROTEST || true
export ROS_LOG_DIR=/tmp/humanslam_ablation_roslogs
mkdir -p "${ROS_LOG_DIR}" "${output_root}"

echo "Waiting for business_school_10x to finish..."
while tmux has-session -t business_school_10x 2>/dev/null; do
  sleep 60
done
echo "Business School campaign finished; starting ablation."

configs=(
  "scene_only 1 0 0"
  "object_only 0 1 0"
  "scene_object 1 1 0"
  "scene_text 1 0 1"
  "object_text 0 1 1"
  "full 1 1 1"
)

layer_flags() {
  local scene="$1" object="$2" text="$3"
  [[ "${scene}" == 1 ]] && printf '%s\n' --use-scene || printf '%s\n' --no-use-scene
  [[ "${object}" == 1 ]] && printf '%s\n' --use-object || printf '%s\n' --no-use-object
  [[ "${text}" == 1 ]] && printf '%s\n' --use-text || printf '%s\n' --no-use-text
}

for spec in "${configs[@]}"; do
  read -r name scene object text <<< "${spec}"
  mapfile -t flags < <(layer_flags "${scene}" "${object}" "${text}")
  for number in 1 2 3; do
    repetition=$(printf '%02d' "${number}")
    run_dir="${output_root}/kitti06_b15_dark80/${name}/run_${repetition}"
    if [[ -s "${run_dir}/humanslam/run_summary.json" ]]; then
      echo "Skipping completed KITTI ${name} run ${repetition}"
      continue
    fi
    echo "Starting KITTI ${name} run ${repetition}/03"
    python3 tools/run_offline_benchmark.py \
      --dataset "${kitti_dataset}" \
      --settings "${kitti_settings}" \
      --ground-truth "${kitti_gt}" \
      --output "${run_dir}" \
      --mode humanslam \
      --overwrite \
      "${flags[@]}"
  done
done

for spec in "${configs[@]}"; do
  read -r name scene object text <<< "${spec}"
  mapfile -t flags < <(layer_flags "${scene}" "${object}" "${text}")
  run_root="${output_root}/business_school_winter/${name}"
  run_name="run_01"
  run_dir="${run_root}/${run_name}"
  if [[ -s "${run_dir}/metrics/summary.json" ]]; then
    echo "Skipping completed Business School winter ${name}"
    continue
  fi
  echo "Starting Business School winter ${name}"
  python3 tools/run_4seasons_experiment.py \
    --sequence-dir "${winter_sequence}" \
    --mode humanslam \
    --run-name "${run_name}" \
    --results-root "${run_root}" \
    --skip-videos \
    --skip-semantic-frames \
    "${flags[@]}"
  python3 tools/evaluate_4seasons_comparison.py \
    --sequence "${winter_sequence}" \
    --results "${run_dir}" \
    --mode humanslam
done

echo "Tri-layer end-to-end ablation complete."
