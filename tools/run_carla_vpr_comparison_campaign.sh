#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
output_root="${project_dir}/test_results/dissertation_final/carla_vpr_comparison_20260823"
datasets=(
  carla_extreme_rain_fog
  carla_deep_night
  carla_dense_fog_overcast
  carla_extreme_sunset_glare
  carla_clear_noon_repeat
)

cd "${project_dir}"
for dataset in "${datasets[@]}"; do
  for method in salad megaloc; do
    summary="${output_root}/${dataset}/${method}/summary.json"
    if [[ ! -s "${summary}" ]]; then
      python3 tools/run_external_vpr_comparison.py \
        --model "${method}" --dataset "${dataset}" \
        --output-root "${output_root}" --max-queries 100 \
        --latency-repetitions 3
    fi
  done

  for method in orb_bow humanslam; do
    summary="${output_root}/${dataset}/${method}/summary.json"
    if [[ -s "${summary}" ]]; then
      continue
    fi
    if [[ "${method}" == "orb_bow" ]]; then
      python3 tools/run_orb_bow_vpr_comparison.py \
        --dataset "${dataset}" --output-root "${output_root}" \
        --max-queries 100 --latency-repetitions 3
    else
      python3 tools/run_humanslam_vpr_comparison.py \
        --dataset "${dataset}" --output-root "${output_root}" \
        --max-queries 100 --latency-repetitions 3 --ocr-backend tensorrt
    fi
  done
done

python3 tools/aggregate_external_vpr_comparison.py --results "${output_root}"
