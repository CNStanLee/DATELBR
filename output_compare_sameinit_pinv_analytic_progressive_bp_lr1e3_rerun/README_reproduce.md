# Reproduce This Version (BP LR = 1e-3)

This folder stores the reproducible result set for:
- same-init comparison
- `reg_method=analytic_progressive`
- BP baseline with `bp_lr=1e-3`

## 1) Re-run comparison (from repo root)

```bash
cd /home/changhong/prj/finn_cli_fork
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
```

## 2) Regenerate harmonic bar stats/figures

```bash
cd /home/changhong/prj/finn_cli_fork/scripts/DATELBR
python plot_harmonic_amplitude_bars.py \
  --base_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR \
  --output_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_progressive_bp_lr1e3_rerun \
  --yscale log
```

## 3) Export MATLAB bundle

```bash
cd /home/changhong/prj/finn_cli_fork/scripts/DATELBR
python export_matlab_plot_bundle.py \
  --source_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_progressive_bp_lr1e3_rerun \
  --bundle_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_progressive_bp_lr1e3_rerun/matlab_bundle
```

## 4) MATLAB plotting

```matlab
cd('/home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_progressive_bp_lr1e3_rerun/matlab_bundle/matlab');
run_all_plots;
```

## Expected outputs
- `compare_summary.csv/json`
- `history_compare_*.json`
- `val_compare_reg_vs_bp_*.png`
- `harmonic_amp_5scenarios_*.png`
- `harmonic_amp_5scenarios_stats.csv/json`
- `harmonic_mae_4methods_log.png` (1-cycle FFT / Pre-finetune / BP finetuned / Closed-form finetuned, with error bars)
- `harmonic_amp_5scenarios_with_4method_errorbars_log.png` (5 grouped scenario panels; each panel shows H3/H5/H7 bars overlaid with 4 method error-bars, with per-panel linear y-range)
- `harmonic_mae_4methods_stats.csv/json`
- `harmonic_mae_4methods_per_scenario.csv`
- `matlab_bundle/`
