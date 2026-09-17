#!/usr/bin/env bash
set -eo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
dataset_root="/home/teleopbike/Documents/slam_experiments/datasets/KITTI/dataset"
results_root="${project_dir}/test_results/dissertation_final/kitti"

source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
cd "${project_dir}"

python3 tools/run_repeated_offline_benchmark.py \
  --dataset test_scenarios/final_kitti00/K00_blur35_dark80_revisits \
  --settings config/KITTI00-02_offline.yaml \
  --ground-truth "${dataset_root}/poses/00.txt" \
  --output "${results_root}/K00-2_blur35_dark80/formal_10x" \
  --perturb-start 1552 \
  --runs 10

python3 tools/run_repeated_offline_benchmark.py \
  --dataset test_scenarios/final_kitti05/K05_blur35_dark80_revisits \
  --settings config/KITTI04-12_offline.yaml \
  --ground-truth "${dataset_root}/poses/05.txt" \
  --output "${results_root}/K05-2_blur35_dark80/formal_10x" \
  --perturb-start 1287 \
  --runs 10
