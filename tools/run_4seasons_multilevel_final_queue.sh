#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
scenario_dir="${project_dir}/test_scenarios/4seasons_multilevel_carpark"
results_dir="${project_dir}/test_results/dissertation_final/4seasons"

cd "${project_dir}"
set +u
source /opt/ros/humble/setup.bash
source /home/teleopbike/Documents/Mayowa/ros2_ws/install/setup.bash
set -u

echo "Waiting for k05_final_campaign to finish..."
while tmux has-session -t k05_final_campaign 2>/dev/null; do
  sleep 30
done

mkdir -p "${results_dir}"

python3 tools/run_4seasons_experiment.py \
  --sequence-dir "${scenario_dir}/recording_2020-12-22_12-04-35" \
  --mode both \
  --run-name "multilevel_carpark_recording_2020-12-22_12-04-35" \
  --results-root "${results_dir}"

python3 tools/run_4seasons_experiment.py \
  --sequence-dir "${scenario_dir}/recording_2021-05-10_19-15-19" \
  --mode both \
  --run-name "multilevel_carpark_recording_2021-05-10_19-15-19" \
  --results-root "${results_dir}"

echo "All queued 4Seasons multilevel-carpark runs completed."
