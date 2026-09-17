#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
output="${project_dir}/test_results/dissertation_final/ablations/frozen_scene_support_10datasets_20260826"
datasets=(
  kitti06_clean kitti06_b15_d80 kitti06_b35_d80
  business_fall garage_feb2021
  carla_extreme_rain_fog carla_deep_night carla_dense_fog_overcast
  carla_extreme_sunset_glare carla_clear_noon_repeat
)

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
export ROS_LOG_DIR=/tmp/humanslam_frozen_ablation_roslogs
export MPLCONFIGDIR=/tmp/humanslam_frozen_ablation_matplotlib
mkdir -p "${ROS_LOG_DIR}" "${MPLCONFIGDIR}"
mkdir -p "${output}"
printf 'dataset\tstate\treturn_code\n' > "${output}/campaign_status.tsv"
for dataset in "${datasets[@]}"; do
  if [[ -s "${output}/${dataset}/summary.csv" ]]; then
    printf '%s\tskipped\t0\n' "${dataset}" >> "${output}/campaign_status.tsv"
    continue
  fi
  set +e
  python3 tools/run_frozen_scene_support_ablation.py \
    --dataset "${dataset}" --output-root "${output}" \
    --max-queries 100 --latency-repetitions 3
  rc=$?
  set -e
  [[ ${rc} -eq 0 ]] && state=complete || state=failed
  printf '%s\t%s\t%s\n' "${dataset}" "${state}" "${rc}" >> "${output}/campaign_status.tsv"
done
