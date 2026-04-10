# MATLAB Plot Bundle

This bundle exports plotting metadata and data for:
- `harmonic_amp_5scenarios_log.png`
- `harmonic_mae_4methods_log.png` (if exported in source_dir)
- `harmonic_amp_5scenarios_with_4method_errorbars_log.png` (if exported in source_dir)
- `val_compare_reg_vs_bp_*.png` (v2g/g2v/g2b/b2g)

## Structure
- `data/`
  - `harmonic_amp_stats.csv`
  - `harmonic_mae_4methods_stats.csv` (optional)
  - `harmonic_mae_4methods_per_scenario.csv` (optional)
  - `val_curve_<scenario>.csv`
- `metadata/`
  - `plot_metadata.json`
  - `export_manifest.json`
- `matlab/`
  - `run_all_plots.m`
  - `plot_harmonic_amp_5scenarios_log.m`
  - `plot_harmonic_mae_4methods_log.m`
  - `plot_harmonic_amp_5scenarios_with_4method_errorbars_linear.m`
  - `plot_val_compare_reg_vs_bp_all.m`
- `reference_figures/` (original python figures for visual reference)
- `figures/` (MATLAB output)

## MATLAB usage
1. `cd matlab`
2. run `run_all_plots`

Generated figures will be saved to `../figures`.
