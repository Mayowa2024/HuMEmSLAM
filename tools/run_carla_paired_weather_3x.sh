#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash

repository="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
dataset_root="/media/teleopbike/Windows/Users/teleo/OneDrive - Oxford Brookes University/Natalie Mouradian's files - carla_slam_data_retry"
output_root="$repository/test_results/dissertation_final/carla_retry/paired_weather_3x_20260822"
settings="$repository/config/carla_town10hd_stereo.yaml"

cd "$repository"
for dataset in "$dataset_root"/pair_*; do
    name="$(basename "$dataset")"
    python3 tools/run_repeated_offline_benchmark.py \
        --dataset "$dataset" \
        --settings "$settings" \
        --ground-truth "$output_root/ground_truth/$name.txt" \
        --output "$output_root/$name" \
        --runs 3 \
        --perturb-start 1520
done
