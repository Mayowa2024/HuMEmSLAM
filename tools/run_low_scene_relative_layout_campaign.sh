#!/usr/bin/env bash
set -eo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
output="${project_dir}/test_results/dissertation_final/ablations/low_scene_relative_layout_20260829"

source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
cd "${project_dir}"
mkdir -p "${output}"

for dataset in business_fall carla_dense_fog_overcast carla_extreme_rain_fog garage_feb2021; do
  python3 tools/run_low_scene_relative_layout_sweep.py \
    --dataset "${dataset}" --output-root "${output}"
done
