#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
results_root="${project_dir}/test_results/dissertation_final/4seasons/stereo_only_comparison_5x_20260820"
minimum_free_kib=$((8 * 1024 * 1024))

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/orbslam3_ros2_ws/install/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
unset HUMANSLAM_ALIAS_MICROTEST || true
export ROS_LOG_DIR=/tmp/humanslam_4seasons_stereo_only_5x_roslogs
mkdir -p "${ROS_LOG_DIR}" "${results_root}"

run_pair() {
  local dataset_id="$1"
  local sequence_dir="$2"
  local end_frame="$3"
  local repetition="$4"
  local run_name="${dataset_id}_run_${repetition}"
  local run_dir="${results_root}/${run_name}"

  local available_kib
  available_kib=$(df --output=avail "${results_root}" | tail -n 1 | tr -d ' ')
  if (( available_kib < minimum_free_kib )); then
    echo "Stopping before ${run_name}: less than 8 GiB free"
    exit 2
  fi

  local attempt mode baseline_complete humanslam_complete
  for attempt in 1 2 3; do
    baseline_complete=0
    humanslam_complete=0
    [[ -s "${run_dir}/baseline/trajectory_kitti.txt" && \
       -s "${run_dir}/baseline/trajectory_frame_ids.csv" && \
       -s "${run_dir}/baseline/aliasing_evaluation/summary.json" ]] && baseline_complete=1
    [[ -s "${run_dir}/humanslam/trajectory_kitti.txt" && \
       -s "${run_dir}/humanslam/trajectory_frame_ids.csv" && \
       -s "${run_dir}/humanslam/loop_retrieval_metrics.json" ]] && humanslam_complete=1

    if (( baseline_complete && humanslam_complete )); then
      python3 tools/evaluate_4seasons_comparison.py \
        --sequence "${sequence_dir}" --results "${run_dir}"
      echo "Completed ${run_name}"
      return
    elif (( baseline_complete )); then
      mode="humanslam"
    elif (( humanslam_complete )); then
      mode="baseline"
    else
      mode="both"
    fi

    echo "Starting/resuming ${run_name} (${mode}, stereo only), attempt ${attempt}/3"
    if python3 tools/run_4seasons_experiment.py \
      --sequence-dir "${sequence_dir}" \
      --mode "${mode}" \
      --results-root "${results_root}" \
      --run-name "${run_name}" \
      --end-frame "${end_frame}" \
      --playback-rate 1.0 \
      --stereo-only \
      --skip-videos \
      --skip-semantic-frames; then
      if [[ "${mode}" != "both" ]]; then
        python3 tools/evaluate_4seasons_comparison.py \
          --sequence "${sequence_dir}" --results "${run_dir}"
      fi
      echo "Completed ${run_name}"
      return
    fi
    echo "Attempt ${attempt}/3 failed for ${run_name}"
  done
  echo "Failed ${run_name} after three attempts"
  exit 1
}

datasets=(
  "business_fall|${project_dir}/test_scenarios/4seasons_business_school/recording_2020-10-08_09-30-57|10741"
  "business_winter|${project_dir}/test_scenarios/4seasons_business_school/recording_2021-01-07_13-12-23|-1"
  "garage_dec2020|${project_dir}/test_scenarios/4seasons_multilevel_carpark/recording_2020-12-22_12-04-35|-1"
  "garage_feb2021|${project_dir}/test_scenarios/4seasons_multilevel_carpark/recording_2021-02-25_13-39-06|-1"
  "garage_may2021|${project_dir}/test_scenarios/4seasons_multilevel_carpark/recording_2021-05-10_19-15-19|-1"
)

for spec in "${datasets[@]}"; do
  IFS='|' read -r dataset_id sequence_dir end_frame <<< "${spec}"
  for repetition in 01 02 03 04 05; do
    run_pair "${dataset_id}" "${sequence_dir}" "${end_frame}" "${repetition}"
  done
done

echo "4Seasons stereo-only 5x comparison campaign complete: ${results_root}"
