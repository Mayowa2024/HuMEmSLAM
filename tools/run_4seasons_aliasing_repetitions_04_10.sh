#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
sequence_dir="${project_dir}/test_scenarios/4seasons_multilevel_carpark/recording_2021-05-10_19-15-19"
results_root="${project_dir}/test_results/dissertation_final/4seasons"
minimum_free_kib=$((3 * 1024 * 1024))

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u

for repetition in 04 05 06 07 08 09 10; do
  available_kib=$(df --output=avail "${results_root}" | tail -n 1 | tr -d ' ')
  if (( available_kib < minimum_free_kib )); then
    echo "Stopping before repetition ${repetition}: less than 3 GiB free."
    exit 2
  fi

  run_name="multilevel_carpark_2021-05-10_aliasing_repeat_${repetition}_20260815"
  echo "Starting repetition ${repetition}: ${run_name}"
  python3 tools/run_4seasons_experiment.py \
    --sequence-dir "${sequence_dir}" \
    --mode both \
    --run-name "${run_name}" \
    --results-root "${results_root}"
  echo "Completed repetition ${repetition}"
done

echo "Completed 4Seasons aliasing repetitions 04--10."
