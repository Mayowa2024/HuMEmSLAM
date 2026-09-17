#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
output_root="${project_dir}/test_results/dissertation_final/vpr_external_comparison_20260821"
cd "${project_dir}"

for dataset in \
  kitti06_clean \
  kitti06_b15_d80 \
  kitti06_b35_d80 \
  business_fall \
  garage_feb2021; do
  for model in salad megaloc; do
    summary="${output_root}/${dataset}/${model}/summary.json"
    if [[ -s "${summary}" ]]; then
      echo "Skipping completed ${dataset}/${model}"
      continue
    fi
    echo "Running ${dataset}/${model}"
    python3 tools/run_external_vpr_comparison.py \
      --model "${model}" \
      --dataset "${dataset}" \
      --output-root "${output_root}" \
      --max-queries 100 \
      --latency-repetitions 3
  done
done

echo "Native PyTorch SALAD/MegaLoc campaign complete"
