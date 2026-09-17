#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
ros_package="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam"
results_root="${project_dir}/test_results/dissertation_final/ablations/scene_support_cross_dataset_40x_20260826"
kitti_root="/home/teleopbike/Documents/slam_experiments/datasets/KITTI/dataset"
kitti_settings="/home/teleopbike/ORB_SLAM3/Examples/Stereo/KITTI04-12.yaml"
carla_root="/media/teleopbike/Windows/Users/teleo/OneDrive - Oxford Brookes University/Natalie Mouradian's files - carla_slam_data_retry"
carla_gt="${project_dir}/test_results/dissertation_final/carla_retry/paired_weather_3x_20260822/ground_truth"
human_config="${ros_package}/config/human_slam_params.yaml"
status_file="${results_root}/campaign_status.tsv"

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
unset HUMANSLAM_ALIAS_MICROTEST || true
export ROS_LOG_DIR=/tmp/humanslam_scene_support_ablation_40x_roslogs
export MPLCONFIGDIR=/tmp/humanslam_scene_support_ablation_40x_matplotlib
mkdir -p "${ROS_LOG_DIR}" "${MPLCONFIGDIR}" "${results_root}"

if ! grep -Eq 'fusion_mode:[[:space:]]*scene_support' "${human_config}"; then
  echo "ERROR: ${human_config} is not configured for scene_support fusion" >&2
  exit 2
fi

printf 'timestamp\tdataset\tconfiguration\tstate\treturn_code\n' > "${status_file}"

configs=(
  "scene_only 1 0 0"
  "scene_object 1 1 0"
  "scene_text 1 0 1"
  "full 1 1 1"
)

layer_flags() {
  local scene="$1" object="$2" text="$3"
  [[ "${scene}" == 1 ]] && printf '%s\n' --use-scene || printf '%s\n' --no-use-scene
  [[ "${object}" == 1 ]] && printf '%s\n' --use-object || printf '%s\n' --no-use-object
  [[ "${text}" == 1 ]] && printf '%s\n' --use-text || printf '%s\n' --no-use-text
}

record_status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" >> "${status_file}"
}

run_kitti() {
  local dataset_id="$1" dataset_path="$2" config_name="$3"
  shift 3
  local output="${results_root}/${dataset_id}/${config_name}/run_01"
  if [[ -s "${output}/humanslam/run_summary.json" && -s "${output}/humanslam/loop_retrieval_metrics.json" ]]; then
    echo "SKIP complete ${dataset_id}/${config_name}"
    record_status "${dataset_id}" "${config_name}" skipped 0
    return 0
  fi
  echo "START ${dataset_id}/${config_name}"
  python3 tools/run_offline_benchmark.py \
    --dataset "${dataset_path}" --settings "${kitti_settings}" \
    --ground-truth "${kitti_root}/poses/06.txt" \
    --output "${output}" --mode humanslam --playback-rate 1.0 \
    --semantic-threshold 0.70 --candidate-count 5 --overwrite "$@"
  local rc=$?
  [[ ${rc} -eq 0 ]] && record_status "${dataset_id}" "${config_name}" complete 0 || record_status "${dataset_id}" "${config_name}" failed "${rc}"
  return 0
}

run_carla() {
  local dataset_id="$1" folder="$2" config_name="$3"
  shift 3
  local output="${results_root}/${dataset_id}/${config_name}/run_01"
  if [[ -s "${output}/humanslam/run_summary.json" && -s "${output}/humanslam/loop_retrieval_metrics.json" ]]; then
    echo "SKIP complete ${dataset_id}/${config_name}"
    record_status "${dataset_id}" "${config_name}" skipped 0
    return 0
  fi
  echo "START ${dataset_id}/${config_name}"
  python3 tools/run_offline_benchmark.py \
    --dataset "${carla_root}/${folder}" \
    --settings "${project_dir}/config/carla_town10hd_stereo.yaml" \
    --ground-truth "${carla_gt}/${folder}.txt" \
    --output "${output}" --mode humanslam --playback-rate 1.0 \
    --semantic-threshold 0.70 --candidate-count 5 --overwrite "$@"
  local rc=$?
  [[ ${rc} -eq 0 ]] && record_status "${dataset_id}" "${config_name}" complete 0 || record_status "${dataset_id}" "${config_name}" failed "${rc}"
  return 0
}

run_fourseasons() {
  local dataset_id="$1" sequence="$2" config_name="$3"
  shift 3
  local output="${results_root}/${dataset_id}/${config_name}/run_01"
  if [[ -s "${output}/humanslam/loop_retrieval_metrics.json" && -s "${output}/metrics/summary.json" ]]; then
    echo "SKIP complete ${dataset_id}/${config_name}"
    record_status "${dataset_id}" "${config_name}" skipped 0
    return 0
  fi
  echo "START ${dataset_id}/${config_name}"
  python3 tools/run_4seasons_experiment.py \
    --sequence-dir "${sequence}" --mode humanslam --stereo-only \
    --results-root "${results_root}/${dataset_id}/${config_name}" \
    --run-name run_01 --playback-rate 1.0 \
    --skip-videos --skip-semantic-frames "$@"
  local rc=$?
  [[ ${rc} -eq 0 ]] && record_status "${dataset_id}" "${config_name}" complete 0 || record_status "${dataset_id}" "${config_name}" failed "${rc}"
  return 0
}

for spec in "${configs[@]}"; do
  read -r config_name scene object text <<< "${spec}"
  mapfile -t flags < <(layer_flags "${scene}" "${object}" "${text}")

  run_kitti kitti06_clean "${kitti_root}/sequences/06" "${config_name}" "${flags[@]}"
  run_kitti kitti06_b15_d80 "${project_dir}/test_scenarios/final_kitti06/K06-4_blur15_dark80" "${config_name}" "${flags[@]}"
  run_kitti kitti06_b35_d80 "${project_dir}/test_scenarios/final_kitti06/K06-5_blur35_dark80" "${config_name}" "${flags[@]}"

  run_fourseasons business_fall "${project_dir}/test_scenarios/4seasons_business_school/recording_2020-10-08_09-30-57" "${config_name}" "${flags[@]}"
  run_fourseasons garage_feb2021 "${project_dir}/test_scenarios/4seasons_multilevel_carpark/recording_2021-02-25_13-39-06" "${config_name}" "${flags[@]}"

  run_carla carla_extreme_rain_fog pair_01_extreme_rain_fog "${config_name}" "${flags[@]}"
  run_carla carla_deep_night pair_02_deep_night "${config_name}" "${flags[@]}"
  run_carla carla_dense_fog_overcast pair_03_dense_fog_overcast "${config_name}" "${flags[@]}"
  run_carla carla_extreme_sunset_glare pair_04_extreme_sunset_glare "${config_name}" "${flags[@]}"
  run_carla carla_clear_noon_repeat pair_05_clear_noon_repeat "${config_name}" "${flags[@]}"
done

echo "CAMPAIGN COMPLETE: ${results_root}"
