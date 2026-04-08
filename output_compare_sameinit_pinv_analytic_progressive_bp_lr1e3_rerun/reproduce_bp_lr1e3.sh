#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/changhong/prj/finn_cli_fork"
WORK_DIR="${REPO_DIR}/scripts/DATELBR"
OUT_DIR="${WORK_DIR}/output_compare_sameinit_pinv_analytic_progressive_bp_lr1e3_rerun"

cd "${REPO_DIR}"
./run-docker.sh python scripts/DATELBR/run_compare_reg_vs_bp_same_init.py \
  --data_dir scripts/DATELBR/dataset \
  --scenario all \
  --cycle half \
  --preprocess_mode rms \
  --epochs 120 \
  --batch_size 512 \
  --lr 1e-3 \
  --bp_lr 1e-3 \
  --reg_lambda 1e-2 \
  --reg_method analytic_progressive \
  --reg_progressive_min_frac 0.1 \
  --reg_progressive_anchor prev \
  --reg_progressive_shuffle_seed 42 \
  --w_bit 3 \
  --a_bit 3 \
  --seed 42 \
  --use_cpu \
  --yscale auto \
  --output_dir scripts/DATELBR/output_compare_sameinit_pinv_analytic_progressive_bp_lr1e3_rerun

cd "${WORK_DIR}"
python plot_harmonic_amplitude_bars.py \
  --base_dir "${WORK_DIR}" \
  --output_dir "${OUT_DIR}" \
  --yscale both

python export_matlab_plot_bundle.py \
  --source_dir "${OUT_DIR}" \
  --bundle_dir "${OUT_DIR}/matlab_bundle"

echo "Done. Outputs in: ${OUT_DIR}"
