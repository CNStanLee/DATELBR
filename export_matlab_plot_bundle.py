import argparse
import csv
import json
import os
import shutil
from typing import Dict, List

import numpy as np


BAR_SCENARIO_COLORS = [
    [0.1215686275, 0.4666666667, 0.7058823529],  # tab10 blue
    [1.0, 0.4980392157, 0.0549019608],           # tab10 orange
    [0.1725490196, 0.6274509804, 0.1725490196],  # tab10 green
    [0.8392156863, 0.1529411765, 0.1568627451],  # tab10 red
    [0.5803921569, 0.4039215686, 0.7411764706],  # tab10 purple
]
CHANNEL_COLORS = {
    "A1": [0.1215686275, 0.4666666667, 0.7058823529],  # tab blue
    "A3": [1.0, 0.4980392157, 0.0549019608],           # tab orange
    "A5": [0.1725490196, 0.6274509804, 0.1725490196],  # tab green
    "A7": [0.8392156863, 0.1529411765, 0.1568627451],  # tab red
}
CHANNELS = ["A1", "A3", "A5", "A7"]
OVERLAY_HARMONICS = ["3", "5", "7"]
OVERLAY_FIELDS = ["A3", "A5", "A7"]
OVERLAY_BAR_COLOR = [0.6039215686, 0.6392156863, 0.6784313725]  # #9aa3ad
OVERLAY_METHOD_COLORS = [
    [0.0901960784, 0.7450980392, 0.8117647059],  # tab:cyan
    [1.0, 0.4980392157, 0.0549019608],           # tab:orange
    [0.1215686275, 0.4666666667, 0.7058823529],  # tab:blue
    [0.8901960784, 0.4666666667, 0.7607843137],  # tab:pink
]
OVERLAY_METHOD_CAPSIZE = [7.5, 6.0, 4.5, 3.0]
DEFAULT_OVERLAY_SCENARIO_ORDER = ["caseA1", "v2g", "g2v", "g2b", "b2g"]
SCENARIO_DISPLAY_LABELS = {
    "caseA1": "Case A1",
    "v2g": "V2G",
    "g2v": "G2V",
    "g2b": "G2B",
    "b2g": "B2G",
}
METHOD_KEYS = ["single_cycle_fft", "pre_finetune", "bp_finetuned", "closed_form_finetuned"]
METHOD_LABELS = {
    "single_cycle_fft": "1-cycle FFT",
    "pre_finetune": "Pre-finetune",
    "bp_finetuned": "BP finetuned",
    "closed_form_finetuned": "Closed-form finetuned",
}
METHOD_COLORS = [
    [0.4, 0.7607843137, 0.6470588235],          # Set2[0]
    [0.9882352941, 0.5529411765, 0.3843137255], # Set2[1]
    [0.5529411765, 0.6274509804, 0.7960784314], # Set2[2]
    [0.9058823529, 0.5411764706, 0.7647058824], # Set2[3]
]


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[object]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_history(base_dir: str, row: Dict[str, str], scenario: str) -> Dict:
    local_hist = os.path.join(base_dir, f"history_compare_{scenario}.json")
    if os.path.exists(local_hist):
        hist_path = local_hist
    else:
        hist_path = row.get("history_json", "")
    if not hist_path or not os.path.exists(hist_path):
        raise FileNotFoundError(f"history json missing for scenario={scenario}, tried {local_hist} and {hist_path}")
    with open(hist_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_val_curve_data(base_dir: str, data_dir: str, compare_rows: List[Dict[str, str]]) -> List[Dict]:
    scenario_meta: List[Dict] = []
    for row in compare_rows:
        scenario = row["scenario"]
        hist = load_history(base_dir, row, scenario)
        reg_mean = np.asarray(hist["reg_val_mae_mean_per_epoch"], dtype=np.float64)
        bp_mean = np.asarray(hist["bp_val_mae_mean_per_epoch"], dtype=np.float64)
        reg_ch = np.asarray(hist["reg_val_mae_per_epoch_by_channel"], dtype=np.float64)
        bp_ch = np.asarray(hist["bp_val_mae_per_epoch_by_channel"], dtype=np.float64)
        if reg_ch.shape[1] != 4 or bp_ch.shape[1] != 4:
            raise ValueError(f"Channel dim must be 4 for scenario {scenario}, got reg={reg_ch.shape}, bp={bp_ch.shape}")
        n = reg_mean.shape[0]
        epochs = np.arange(n, dtype=np.int64)
        out_csv = os.path.join(data_dir, f"val_curve_{scenario}.csv")
        rows = []
        for i in range(n):
            rows.append(
                [
                    int(epochs[i]),
                    float(reg_mean[i]),
                    float(bp_mean[i]),
                    float(reg_ch[i, 0]),
                    float(reg_ch[i, 1]),
                    float(reg_ch[i, 2]),
                    float(reg_ch[i, 3]),
                    float(bp_ch[i, 0]),
                    float(bp_ch[i, 1]),
                    float(bp_ch[i, 2]),
                    float(bp_ch[i, 3]),
                ]
            )
        write_csv(
            out_csv,
            [
                "epoch",
                "reg_mean",
                "bp_mean",
                "reg_A1",
                "reg_A3",
                "reg_A5",
                "reg_A7",
                "bp_A1",
                "bp_A3",
                "bp_A5",
                "bp_A7",
            ],
            rows,
        )
        scenario_meta.append(
            {
                "name": scenario,
                "csv": os.path.relpath(out_csv, os.path.dirname(data_dir) + "/..").replace("\\", "/"),
                "yscale": row.get("curve_yscale", "log"),
            }
        )
    return scenario_meta


def write_matlab_scripts(matlab_dir: str, include_method_error_bar: bool, include_overlay_error_bar: bool):
    ensure_dir(matlab_dir)

    run_all = """function run_all_plots()
% One-click plotting for exported metadata/data.
this_dir = fileparts(mfilename('fullpath'));
plot_harmonic_amp_5scenarios_log(this_dir);
plot_harmonic_mae_4methods_log(this_dir);
plot_harmonic_amp_5scenarios_with_4method_errorbars_linear(this_dir);
plot_val_compare_reg_vs_bp_all(this_dir);
disp('All MATLAB plots generated.');
end
"""

    plot_bar = """function plot_harmonic_amp_5scenarios_log(script_dir)
if nargin < 1
    script_dir = fileparts(mfilename('fullpath'));
end
bundle_dir = fileparts(script_dir);
data_dir = fullfile(bundle_dir, 'data');
meta_dir = fullfile(bundle_dir, 'metadata');
fig_dir = fullfile(bundle_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

meta = jsondecode(fileread(fullfile(meta_dir, 'plot_metadata.json')));
t = readtable(fullfile(data_dir, 'harmonic_amp_stats.csv'));

means = [t.mean_A1, t.mean_A3, t.mean_A5, t.mean_A7];   % Nscenario x 4
figure('Color', 'w', 'Position', [100 100 1080 460]);
bh = bar(means', 'grouped', 'LineWidth', 0.5);          % 4 groups x Nscenario bars
for i = 1:numel(bh)
    bh(i).FaceColor = meta.harmonic_bar.scenario_colors(i, :);
end
set(gca, 'XTickLabel', meta.harmonic_bar.harmonics);
set(gca, 'YScale', 'log');
xlabel('Harmonic Order');
ylabel('Amplitude');
title('Harmonic Amplitude by Scenario (log)');
grid on;
legend(meta.harmonic_bar.scenarios, 'Location', 'northoutside', 'Orientation', 'horizontal');
exportgraphics(gcf, fullfile(fig_dir, 'harmonic_amp_5scenarios_log_matlab.png'), 'Resolution', 160);
close(gcf);
end
"""

    plot_method_bar = """function plot_harmonic_mae_4methods_log(script_dir)
if nargin < 1
    script_dir = fileparts(mfilename('fullpath'));
end
bundle_dir = fileparts(script_dir);
meta_dir = fullfile(bundle_dir, 'metadata');
fig_dir = fullfile(bundle_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

meta = jsondecode(fileread(fullfile(meta_dir, 'plot_metadata.json')));
if ~isfield(meta, 'method_error_bar')
    disp('method_error_bar metadata not found, skip plot_harmonic_mae_4methods_log.');
    return;
end

src_csv = fullfile(bundle_dir, meta.method_error_bar.source_csv);
if ~exist(src_csv, 'file')
    disp(['method_error_bar source csv missing: ', src_csv]);
    return;
end

t = readtable(src_csv);
methods = cellstr(meta.method_error_bar.methods);
method_labels = cellstr(meta.method_error_bar.method_labels);
harmonics = cellstr(meta.method_error_bar.harmonics);
colors = meta.method_error_bar.method_colors;
n_method = numel(methods);
n_h = numel(harmonics);

mean_cols = {'mean_A1', 'mean_A3', 'mean_A5', 'mean_A7'};
std_cols = {'std_A1', 'std_A3', 'std_A5', 'std_A7'};
means = zeros(n_method, n_h);
stds = zeros(n_method, n_h);
method_col = string(t.method);
for i = 1:n_method
    idx = find(method_col == string(methods{i}), 1);
    if isempty(idx)
        continue;
    end
    for h = 1:n_h
        means(i, h) = t.(mean_cols{h})(idx);
        stds(i, h) = t.(std_cols{h})(idx);
    end
end

figure('Color', 'w', 'Position', [100 100 1150 500]);
bh = bar(means', 'grouped', 'LineWidth', 0.5);
hold on;
for i = 1:numel(bh)
    bh(i).FaceColor = colors(i, :);
    x = bh(i).XEndPoints;
    errorbar(x, means(i, :), stds(i, :), 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 6);
end
set(gca, 'XTickLabel', harmonics);
set(gca, 'YScale', 'log');
xlabel('Harmonic Order');
ylabel('MAE (vs 10-cycle FFT label)');
title('Harmonic MAE by Method (log, mean\pmstd across scenarios)');
grid on;
legend(method_labels, 'Location', 'northoutside', 'Orientation', 'horizontal');
exportgraphics(gcf, fullfile(fig_dir, 'harmonic_mae_4methods_log_matlab.png'), 'Resolution', 160);
close(gcf);
end
"""

    plot_overlay_bar = """function plot_harmonic_amp_5scenarios_with_4method_errorbars_linear(script_dir)
if nargin < 1
    script_dir = fileparts(mfilename('fullpath'));
end
bundle_dir = fileparts(script_dir);
meta_dir = fullfile(bundle_dir, 'metadata');
fig_dir = fullfile(bundle_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

meta = jsondecode(fileread(fullfile(meta_dir, 'plot_metadata.json')));
if ~isfield(meta, 'overlay_error_bar')
    disp('overlay_error_bar metadata not found, skip plot_harmonic_amp_5scenarios_with_4method_errorbars_linear.');
    return;
end

cfg = meta.overlay_error_bar;
bar_csv = fullfile(bundle_dir, cfg.bar_source_csv);
err_csv = fullfile(bundle_dir, cfg.error_source_csv);
if ~exist(bar_csv, 'file')
    disp(['overlay bar source csv missing: ', bar_csv]);
    return;
end
if ~exist(err_csv, 'file')
    disp(['overlay error source csv missing: ', err_csv]);
    return;
end

tbar = readtable(bar_csv);
terr = readtable(err_csv);

scenarios = cellstr(cfg.scenario_order);
sc_display = cellstr(cfg.scenario_display_labels);
harmonics = cellstr(cfg.harmonics);
h_fields = cellstr(cfg.harmonic_fields);
methods = cellstr(cfg.methods);
method_labels = cellstr(cfg.method_labels);
method_colors = cfg.method_colors;
method_caps = cfg.method_capsize;
bar_color = cfg.bar_color;
bar_width = cfg.bar_width;

n_s = numel(scenarios);
n_h = numel(h_fields);
fig_w = 300 * n_s + 120;
figure('Color', 'w', 'Position', [100 80 fig_w 560]);

for s = 1:n_s
    sc = scenarios{s};
    subplot(1, n_s, s);
    hold on;
    bar_idx = find(string(tbar.scenario) == string(sc), 1);
    if isempty(bar_idx)
        title(sc_display{s});
        grid on;
        continue;
    end

    bar_top = zeros(1, n_h);
    for h = 1:n_h
        col_name = sprintf('mean_%s', h_fields{h});
        bar_top(h) = tbar.(col_name)(bar_idx);
    end

    x = 1:n_h;
    bar(x, bar_top, bar_width, 'FaceColor', bar_color, 'EdgeColor', [0 0 0], 'LineWidth', 0.45);

    y_low = bar_top;
    y_high = bar_top;
    for m = 1:numel(methods)
        m_idx = find(string(terr.scenario) == string(sc) & string(terr.method) == string(methods{m}), 1);
        if isempty(m_idx)
            continue;
        end
        errs = zeros(1, n_h);
        for h = 1:n_h
            errs(h) = terr.(h_fields{h})(m_idx);
        end
        y_low = [y_low, bar_top - errs]; %#ok<AGROW>
        y_high = [y_high, bar_top + errs]; %#ok<AGROW>
        errorbar(
            x, bar_top, errs, 'LineStyle', 'none', 'Color', method_colors(m, :), ...
            'LineWidth', 2.0, 'CapSize', method_caps(m)
        );
    end

    y_low_ref = min(y_low(:));
    y_high_ref = max(y_high(:));
    span = max(y_high_ref - y_low_ref, max(1e-6, 0.08 * max(abs(y_high_ref), 1.0)));
    pad = 0.06 * span;
    y_min = y_low_ref - pad;
    y_max = y_high_ref + pad;
    if y_low_ref >= 0
        y_min = max(0, y_min);
    end
    if y_max <= y_min
        y_max = y_min + span;
    end
    ylim([y_min, y_max]);
    xlim([0.5, n_h + 0.5]);
    set(gca, 'XTick', x, 'XTickLabel', harmonics);
    xlabel('H');
    if s == 1
        ylabel('Amplitude / Error');
    end
    title(sc_display{s});
    grid on;
    hold off;
end

% Shared legend
ax_leg = subplot(1, n_s, 1);
hold(ax_leg, 'on');
legend_handles = gobjects(numel(methods) + 1, 1);
legend_handles(1) = bar(ax_leg, nan, nan, bar_width, 'FaceColor', bar_color, 'EdgeColor', [0 0 0], 'LineWidth', 0.45);
for m = 1:numel(methods)
    legend_handles(m + 1) = plot(ax_leg, nan, nan, '-', 'Color', method_colors(m, :), 'LineWidth', 1.8);
end
legend_labels = [{'Amplitude bar'}; method_labels(:)];
lg = legend(ax_leg, legend_handles, legend_labels, 'Location', 'northoutside', 'Orientation', 'horizontal');
set(lg, 'Interpreter', 'none');
hold(ax_leg, 'off');

sgtitle('Scenario-Grouped Harmonic Amplitude (H3/H5/H7) with Raw Method Error-bars');
exportgraphics(gcf, fullfile(fig_dir, 'harmonic_amp_5scenarios_with_4method_errorbars_linear_matlab.png'), 'Resolution', 160);
close(gcf);
end
"""

    plot_val = """function plot_val_compare_reg_vs_bp_all(script_dir)
if nargin < 1
    script_dir = fileparts(mfilename('fullpath'));
end
bundle_dir = fileparts(script_dir);
data_dir = fullfile(bundle_dir, 'data');
meta_dir = fullfile(bundle_dir, 'metadata');
fig_dir = fullfile(bundle_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

meta = jsondecode(fileread(fullfile(meta_dir, 'plot_metadata.json')));
chs = cellstr(meta.val_compare.harmonics);
colors = meta.val_compare.channel_colors;

for s = 1:numel(meta.val_compare.scenarios)
    sc = meta.val_compare.scenarios(s);
    t = readtable(fullfile(bundle_dir, sc.csv));
    figure('Color', 'w', 'Position', [100 100 1100 470]);
    hold on;
    for c = 1:numel(chs)
        ch = chs{c};
        reg_col = t.(sprintf('reg_%s', ch));
        bp_col = t.(sprintf('bp_%s', ch));
        col = colors(c, :);
        plot(t.epoch, reg_col, '-', 'Color', col, 'LineWidth', 2.0, 'DisplayName', sprintf('%s Reg', ch));
        plot(t.epoch, bp_col, '--', 'Color', col, 'LineWidth', 1.8, 'DisplayName', sprintf('%s BP', ch));
    end
    xline(0, '--', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    if strcmpi(sc.yscale, 'log')
        set(gca, 'YScale', 'log');
    end
    xlabel('Epoch');
    ylabel('Val MAE (original scale)');
    title(sprintf('%s same-init comparison (%s y-scale)', sc.name, sc.yscale));
    grid on;
    legend('Location', 'eastoutside');
    exportgraphics(gcf, fullfile(fig_dir, sprintf('val_compare_reg_vs_bp_%s_matlab.png', sc.name)), 'Resolution', 160);
    close(gcf);
end
end
"""

    with open(os.path.join(matlab_dir, "run_all_plots.m"), "w", encoding="utf-8") as f:
        f.write(run_all)
    with open(os.path.join(matlab_dir, "plot_harmonic_amp_5scenarios_log.m"), "w", encoding="utf-8") as f:
        f.write(plot_bar)
    with open(os.path.join(matlab_dir, "plot_harmonic_mae_4methods_log.m"), "w", encoding="utf-8") as f:
        if include_method_error_bar:
            f.write(plot_method_bar)
        else:
            f.write(
                "function plot_harmonic_mae_4methods_log(~)\n"
                "disp('harmonic_mae_4methods data not found, skip.');\n"
                "end\n"
            )
    with open(
        os.path.join(matlab_dir, "plot_harmonic_amp_5scenarios_with_4method_errorbars_linear.m"),
        "w",
        encoding="utf-8",
    ) as f:
        if include_overlay_error_bar:
            f.write(plot_overlay_bar)
        else:
            f.write(
                "function plot_harmonic_amp_5scenarios_with_4method_errorbars_linear(~)\n"
                "disp('overlay 4-method error-bar data not found, skip.');\n"
                "end\n"
            )
    with open(os.path.join(matlab_dir, "plot_val_compare_reg_vs_bp_all.m"), "w", encoding="utf-8") as f:
        f.write(plot_val)


def write_readme(bundle_dir: str):
    txt = """# MATLAB Plot Bundle

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
"""
    with open(os.path.join(bundle_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    parser = argparse.ArgumentParser(description="Export metadata + MATLAB plotting bundle.")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp",
    )
    parser.add_argument(
        "--bundle_dir",
        type=str,
        default="/home/changhong/prj/finn_cli_fork/scripts/DATELBR/output_compare_sameinit_pinv_analytic_vs_bp/matlab_bundle",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    bundle_dir = args.bundle_dir
    data_dir = os.path.join(bundle_dir, "data")
    meta_dir = os.path.join(bundle_dir, "metadata")
    matlab_dir = os.path.join(bundle_dir, "matlab")
    fig_dir = os.path.join(bundle_dir, "figures")
    ref_dir = os.path.join(bundle_dir, "reference_figures")
    for d in [bundle_dir, data_dir, meta_dir, matlab_dir, fig_dir]:
        ensure_dir(d)
    ensure_dir(ref_dir)

    compare_csv = os.path.join(source_dir, "compare_summary.csv")
    compare_rows = read_csv_rows(compare_csv)
    compare_rows = [r for r in compare_rows if r.get("status") == "ok"]
    scenario_order = [r["scenario"] for r in compare_rows]

    # Harmonic bar stats data
    harmonic_stats_src = os.path.join(source_dir, "harmonic_amp_5scenarios_stats.csv")
    harmonic_stats_dst = os.path.join(data_dir, "harmonic_amp_stats.csv")
    if not os.path.exists(harmonic_stats_src):
        raise FileNotFoundError(f"Missing {harmonic_stats_src}")
    with open(harmonic_stats_src, "r", encoding="utf-8") as f_in, open(harmonic_stats_dst, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.read())
    harmonic_ref_src = os.path.join(source_dir, "harmonic_amp_5scenarios_log.png")
    if os.path.exists(harmonic_ref_src):
        shutil.copy2(harmonic_ref_src, os.path.join(ref_dir, "harmonic_amp_5scenarios_log.png"))

    # Method-error bar data (optional)
    method_stats_src = os.path.join(source_dir, "harmonic_mae_4methods_stats.csv")
    method_per_s_src = os.path.join(source_dir, "harmonic_mae_4methods_per_scenario.csv")
    method_ref_src = os.path.join(source_dir, "harmonic_mae_4methods_log.png")
    overlay_ref_src = os.path.join(source_dir, "harmonic_amp_5scenarios_with_4method_errorbars_log.png")
    has_method_error_bar = os.path.exists(method_stats_src)
    if has_method_error_bar:
        method_stats_dst = os.path.join(data_dir, "harmonic_mae_4methods_stats.csv")
        with open(method_stats_src, "r", encoding="utf-8") as f_in, open(method_stats_dst, "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
        if os.path.exists(method_per_s_src):
            with open(method_per_s_src, "r", encoding="utf-8") as f_in, open(
                os.path.join(data_dir, "harmonic_mae_4methods_per_scenario.csv"), "w", encoding="utf-8"
            ) as f_out:
                f_out.write(f_in.read())
        if os.path.exists(method_ref_src):
            shutil.copy2(method_ref_src, os.path.join(ref_dir, "harmonic_mae_4methods_log.png"))
    has_overlay_error_bar = has_method_error_bar and os.path.exists(method_per_s_src)
    if has_overlay_error_bar and os.path.exists(overlay_ref_src):
        shutil.copy2(
            overlay_ref_src,
            os.path.join(ref_dir, "harmonic_amp_5scenarios_with_4method_errorbars_log.png"),
        )

    # Val curve per scenario
    val_scenarios_meta = []
    for row in compare_rows:
        scenario = row["scenario"]
        hist = load_history(source_dir, row, scenario)
        reg_mean = np.asarray(hist["reg_val_mae_mean_per_epoch"], dtype=np.float64)
        bp_mean = np.asarray(hist["bp_val_mae_mean_per_epoch"], dtype=np.float64)
        reg_ch = np.asarray(hist["reg_val_mae_per_epoch_by_channel"], dtype=np.float64)
        bp_ch = np.asarray(hist["bp_val_mae_per_epoch_by_channel"], dtype=np.float64)
        n = reg_mean.shape[0]
        out_csv = os.path.join(data_dir, f"val_curve_{scenario}.csv")
        rows = []
        for i in range(n):
            rows.append(
                [
                    i,
                    float(reg_mean[i]),
                    float(bp_mean[i]),
                    float(reg_ch[i, 0]),
                    float(reg_ch[i, 1]),
                    float(reg_ch[i, 2]),
                    float(reg_ch[i, 3]),
                    float(bp_ch[i, 0]),
                    float(bp_ch[i, 1]),
                    float(bp_ch[i, 2]),
                    float(bp_ch[i, 3]),
                ]
            )
        write_csv(
            out_csv,
            [
                "epoch",
                "reg_mean",
                "bp_mean",
                "reg_A1",
                "reg_A3",
                "reg_A5",
                "reg_A7",
                "bp_A1",
                "bp_A3",
                "bp_A5",
                "bp_A7",
            ],
            rows,
        )
        val_scenarios_meta.append(
            {
                "name": scenario,
                "csv": f"data/val_curve_{scenario}.csv",
                "yscale": row.get("curve_yscale", "log"),
            }
        )
        ref_curve_src = os.path.join(source_dir, f"val_compare_reg_vs_bp_{scenario}.png")
        if os.path.exists(ref_curve_src):
            shutil.copy2(ref_curve_src, os.path.join(ref_dir, f"val_compare_reg_vs_bp_{scenario}.png"))

    harmonic_rows = read_csv_rows(harmonic_stats_dst)
    harmonic_scenarios = [r["scenario"] for r in harmonic_rows]
    colors = BAR_SCENARIO_COLORS[: len(harmonic_scenarios)]
    metadata = {
        "harmonic_bar": {
            "harmonics": ["1", "3", "5", "7"],
            "scenarios": harmonic_scenarios,
            "scenario_colors": colors,
            "source_csv": "data/harmonic_amp_stats.csv",
            "preferred_yscale": "log",
        },
        "val_compare": {
            "harmonics": CHANNELS,
            "channel_colors": [CHANNEL_COLORS[ch] for ch in CHANNELS],
            "reg_linestyle": "-",
            "bp_linestyle": "--",
            "scenarios": val_scenarios_meta,
            "scenario_order_from_compare_summary": scenario_order,
        },
    }
    if has_method_error_bar:
        metadata["method_error_bar"] = {
            "harmonics": ["1", "3", "5", "7"],
            "methods": METHOD_KEYS,
            "method_labels": [METHOD_LABELS[k] for k in METHOD_KEYS],
            "method_colors": METHOD_COLORS,
            "source_csv": "data/harmonic_mae_4methods_stats.csv",
            "preferred_yscale": "log",
            "errorbar": "std",
        }
    if has_overlay_error_bar:
        per_s_rows = read_csv_rows(os.path.join(data_dir, "harmonic_mae_4methods_per_scenario.csv"))
        per_s_scenarios = sorted(set(r["scenario"] for r in per_s_rows))
        overlay_order = [sc for sc in DEFAULT_OVERLAY_SCENARIO_ORDER if sc in harmonic_scenarios and sc in per_s_scenarios]
        for sc in harmonic_scenarios:
            if sc in per_s_scenarios and sc not in overlay_order:
                overlay_order.append(sc)
        metadata["overlay_error_bar"] = {
            "harmonics": OVERLAY_HARMONICS,
            "harmonic_fields": OVERLAY_FIELDS,
            "methods": METHOD_KEYS,
            "method_labels": [METHOD_LABELS[k] for k in METHOD_KEYS],
            "method_colors": OVERLAY_METHOD_COLORS,
            "method_capsize": OVERLAY_METHOD_CAPSIZE,
            "bar_color": OVERLAY_BAR_COLOR,
            "bar_width": 0.42,
            "bar_source_csv": "data/harmonic_amp_stats.csv",
            "error_source_csv": "data/harmonic_mae_4methods_per_scenario.csv",
            "scenario_order": overlay_order,
            "scenario_display_labels": [SCENARIO_DISPLAY_LABELS.get(sc, sc) for sc in overlay_order],
            "preferred_yscale": "linear",
        }
    with open(os.path.join(meta_dir, "plot_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    data_files = ["data/harmonic_amp_stats.csv"] + [f"data/val_curve_{s}.csv" for s in scenario_order]
    if has_method_error_bar:
        data_files.append("data/harmonic_mae_4methods_stats.csv")
        if os.path.exists(os.path.join(data_dir, "harmonic_mae_4methods_per_scenario.csv")):
            data_files.append("data/harmonic_mae_4methods_per_scenario.csv")

    ref_files = ["reference_figures/harmonic_amp_5scenarios_log.png"] + [
        f"reference_figures/val_compare_reg_vs_bp_{s}.png" for s in scenario_order
    ]
    if has_method_error_bar and os.path.exists(os.path.join(ref_dir, "harmonic_mae_4methods_log.png")):
        ref_files.append("reference_figures/harmonic_mae_4methods_log.png")
    if has_overlay_error_bar and os.path.exists(
        os.path.join(ref_dir, "harmonic_amp_5scenarios_with_4method_errorbars_log.png")
    ):
        ref_files.append("reference_figures/harmonic_amp_5scenarios_with_4method_errorbars_log.png")

    manifest = {
        "source_dir": source_dir,
        "bundle_dir": bundle_dir,
        "generated_files": {
            "metadata": [
                "metadata/plot_metadata.json",
                "metadata/export_manifest.json",
            ],
            "data": data_files,
            "matlab": [
                "matlab/run_all_plots.m",
                "matlab/plot_harmonic_amp_5scenarios_log.m",
                "matlab/plot_harmonic_mae_4methods_log.m",
                "matlab/plot_harmonic_amp_5scenarios_with_4method_errorbars_linear.m",
                "matlab/plot_val_compare_reg_vs_bp_all.m",
            ],
            "reference_figures": ref_files,
        },
    }
    with open(os.path.join(meta_dir, "export_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_matlab_scripts(
        matlab_dir,
        include_method_error_bar=has_method_error_bar,
        include_overlay_error_bar=has_overlay_error_bar,
    )
    write_readme(bundle_dir)
    print(f"MATLAB bundle exported: {bundle_dir}")


if __name__ == "__main__":
    main()
