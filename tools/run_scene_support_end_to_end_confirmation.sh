#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
results_root="${project_dir}/test_results/dissertation_final/scene_support_end_to_end_confirmation_20260823"
kitti_root="/home/teleopbike/Documents/slam_experiments/datasets/KITTI/dataset"
human_config="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/config/human_slam_params.yaml"
orb_settings="/home/teleopbike/ORB_SLAM3/Examples/Stereo/KITTI04-12.yaml"
carla_root="/media/teleopbike/Windows/Users/teleo/OneDrive - Oxford Brookes University/Natalie Mouradian's files - carla_slam_data_retry"

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
unset HUMANSLAM_ALIAS_MICROTEST || true
export ROS_LOG_DIR=/tmp/humanslam_scene_support_confirmation_roslogs
mkdir -p "${ROS_LOG_DIR}" "${results_root}"

run_kitti() {
  local name="$1"
  local dataset="$2"
  local output="${results_root}/${name}"
  if [[ -s "${output}/humanslam/trajectory_kitti.txt" && \
        -s "${output}/humanslam/loop_retrieval_metrics.json" ]]; then
    echo "Skipping completed ${name}"
    return
  fi
  python3 tools/run_offline_benchmark.py \
    --dataset "${dataset}" \
    --settings "${orb_settings}" \
    --ground-truth "${kitti_root}/poses/06.txt" \
    --output "${output}" --mode humanslam --playback-rate 1.0 \
    --semantic-threshold 0.70 --candidate-count 5 --overwrite
}

run_carla() {
  local name="carla_deep_night"
  local output="${results_root}/${name}"
  if [[ -s "${output}/humanslam/trajectory_kitti.txt" && \
        -s "${output}/humanslam/loop_retrieval_metrics.json" ]]; then
    echo "Skipping completed ${name}"
    return
  fi
  python3 tools/run_offline_benchmark.py \
    --dataset "${carla_root}/pair_02_deep_night" \
    --settings "${project_dir}/config/carla_town10hd_stereo.yaml" \
    --ground-truth "${project_dir}/test_results/dissertation_final/carla_retry/paired_weather_3x_20260822/ground_truth/pair_02_deep_night.txt" \
    --output "${output}" --mode humanslam --playback-rate 1.0 \
    --semantic-threshold 0.70 --candidate-count 5 --overwrite
}

run_garage() {
  local name="garage_feb2021"
  local sequence="${project_dir}/test_scenarios/4seasons_multilevel_carpark/recording_2021-02-25_13-39-06"
  local output="${results_root}/${name}"
  if [[ -s "${output}/humanslam/trajectory_kitti.txt" && \
        -s "${output}/humanslam/loop_retrieval_metrics.json" ]]; then
    echo "Skipping completed ${name}"
    return
  fi
  python3 tools/run_4seasons_experiment.py \
    --sequence-dir "${sequence}" --mode humanslam \
    --results-root "${results_root}" --run-name "${name}" \
    --end-frame -1 --playback-rate 1.0 --stereo-only \
    --skip-videos --skip-semantic-frames
}

run_kitti "kitti06_clean" "${kitti_root}/sequences/06"
run_kitti "kitti06_b15_d80" "${project_dir}/test_scenarios/final_kitti06/K06-4_blur15_dark80"
run_carla
run_garage

echo "Scene-support end-to-end confirmation campaign complete: ${results_root}"
