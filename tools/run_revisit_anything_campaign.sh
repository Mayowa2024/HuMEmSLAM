#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam
RA="$ROOT/external/vpr_comparison/Revisit-Anything"
OUT="$ROOT/test_results/dissertation_final/vpr_external_comparison_20260821"
EXP=exp0_global_SegLoc_VLAD_PCA_o3

cd "$ROOT"
for dataset in kitti06_clean kitti06_b15_d80 kitti06_b35_d80 business_fall garage_feb2021; do
  if [[ -f "$OUT/$dataset/revisit_anything/summary.json" ]]; then
    echo "Skipping completed dataset: $dataset"
    continue
  fi
  stage="$RA/workdir_data/$dataset/out"
  mkdir -p "$stage"
  for method in SAM DINO; do
    for split in ref query; do
      marker="$stage/.efficient_l2_vitb_${method}_${split}.done"
      if [[ ! -f "$marker" ]]; then
        python3 tools/run_revisit_preprocess_split.py \
          --dataset "$dataset" --split "$split" --method "$method"
        touch "$marker"
      fi
    done
  done
  python3 tools/build_revisit_vitb_vocabulary.py --dataset "$dataset"
  (
    cd "$RA"
    python3 place_rec_pca.py --dataset "$dataset" --experiment "$EXP" --vocab-vlad domain
    python3 place_rec_main.py --dataset "$dataset" --experiment "$EXP" \
      --vocab-vlad domain --save-results
  )
  python3 tools/evaluate_revisit_anything_fixed.py \
    --dataset "$dataset" --output-root "$OUT" --max-queries 100
  python3 tools/time_revisit_anything_fixed.py \
    --dataset "$dataset" --output-root "$OUT" --repetitions 3

  find "$stage" -maxdepth 1 -type f \( -name '*.h5' -o -name '*.hdf5' \) -delete
done

python3 tools/aggregate_external_vpr_comparison.py --results "$OUT"
