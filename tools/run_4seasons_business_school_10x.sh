#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
results_root="${project_dir}/test_results/dissertation_final/4seasons/business_school_20260816"
minimum_free_kib=$((12 * 1024 * 1024))

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u
unset HUMANSLAM_ALIAS_MICROTEST || true
export ROS_LOG_DIR=/tmp/humanslam_business_school_roslogs
mkdir -p "${ROS_LOG_DIR}" "${results_root}"

run_repetition() {
  local season="$1"
  local sequence_dir="$2"
  local end_frame="$3"
  local repetition="$4"
  local run_name="business_school_${season}_repeat_${repetition}_20260816"
  local run_dir="${results_root}/${run_name}"

  if [[ -s "${run_dir}/metrics/summary.json" ]]; then
    echo "Skipping completed ${run_name}"
    return
  fi

  local available_kib
  available_kib=$(df --output=avail "${results_root}" | tail -n 1 | tr -d ' ')
  if (( available_kib < minimum_free_kib )); then
    echo "Stopping before ${run_name}: less than 12 GiB free."
    exit 2
  fi

  echo "Starting ${run_name}"
  local attempt
  for attempt in 1 2; do
    local mode="both"
    if [[ -s "${run_dir}/baseline/aliasing_evaluation/summary.json" ]]; then
      mode="humanslam"
      echo "Resuming ${run_name} from completed baseline"
    fi
    if [[ -s "${run_dir}/humanslam/aliasing_evaluation/summary.json" ]]; then
      python3 tools/evaluate_4seasons_comparison.py \
        --sequence "${sequence_dir}" --results "${run_dir}"
      echo "Completed ${run_name} from existing method outputs"
      return
    fi
    if python3 tools/run_4seasons_experiment.py \
      --sequence-dir "${sequence_dir}" \
      --mode "${mode}" \
      --end-frame "${end_frame}" \
      --run-name "${run_name}" \
      --results-root "${results_root}" \
      --skip-videos \
      --skip-semantic-frames; then
      if [[ "${mode}" == "humanslam" ]]; then
        python3 tools/evaluate_4seasons_comparison.py \
          --sequence "${sequence_dir}" --results "${run_dir}"
      fi
      echo "Completed ${run_name}"
      return
    fi
    echo "Attempt ${attempt} failed for ${run_name}"
  done
  echo "Failed ${run_name} after two attempts"
  exit 1
}

fall_sequence="${project_dir}/test_scenarios/4seasons_business_school/recording_2020-10-08_09-30-57"
winter_sequence="${project_dir}/test_scenarios/4seasons_business_school/recording_2021-01-07_13-12-23"

for repetition in $(seq -w 1 10); do
  run_repetition "fall" "${fall_sequence}" 10741 "${repetition}"
done

for repetition in $(seq -w 1 10); do
  run_repetition "winter" "${winter_sequence}" -1 "${repetition}"
done

echo "Business School 10x fall/winter campaign complete."
