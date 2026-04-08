# Plotting README (Reproduced With Current `run_compare` Code)

本目录下的图是用当前回退后的 `run_compare_reg_vs_bp_same_init.py` 重新生成的。

## 1) 先重跑 `val_compare_reg_vs_bp_xxx.png`

在仓库根目录运行（推荐和你当前环境一致）：

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
  --reg_lambda 1e-2 \
  --reg_method analytic \
  --w_bit 3 --a_bit 3 \
  --seed 42 \
  --use_cpu \
  --yscale auto \
  --output_dir scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp
```

输出：
- `compare_summary.csv/json`
- `history_compare_*.json`
- `val_compare_reg_vs_bp_v2g.png`
- `val_compare_reg_vs_bp_g2v.png`
- `val_compare_reg_vs_bp_g2b.png`
- `val_compare_reg_vs_bp_b2g.png`

## 2) 画谐波幅值柱状图 `harmonic_amp_5scenarios_*.png`

```bash
cd /home/changhong/prj/finn_cli_fork/scripts/DATELBR
python plot_harmonic_amplitude_bars.py \
  --base_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR \
  --output_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp \
  --yscale both
```

输出：
- `harmonic_amp_5scenarios_linear.png`
- `harmonic_amp_5scenarios_log.png`
- `harmonic_amp_5scenarios_stats.csv/json`

## 3) 导出 MATLAB 重绘包（metadata + data + .m 脚本）

```bash
cd /home/changhong/prj/finn_cli_fork/scripts/DATELBR
python export_matlab_plot_bundle.py \
  --source_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp \
  --bundle_dir /home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp/matlab_bundle
```

输出目录：
- `matlab_bundle/data`
- `matlab_bundle/metadata`
- `matlab_bundle/matlab`
- `matlab_bundle/reference_figures`

## 4) 在 MATLAB 里重绘

```matlab
cd('/home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp/matlab_bundle/matlab');
run_all_plots
```

MATLAB 输出会写到：
- `matlab_bundle/figures`
