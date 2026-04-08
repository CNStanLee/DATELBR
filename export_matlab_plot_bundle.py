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


def write_matlab_scripts(matlab_dir: str):
    ensure_dir(matlab_dir)

    run_all = """function run_all_plots()
% One-click plotting for exported metadata/data.
this_dir = fileparts(mfilename('fullpath'));
plot_harmonic_amp_5scenarios_log(this_dir);
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
    with open(os.path.join(matlab_dir, "plot_val_compare_reg_vs_bp_all.m"), "w", encoding="utf-8") as f:
        f.write(plot_val)


def write_readme(bundle_dir: str):
    txt = """# MATLAB Plot Bundle

This bundle exports plotting metadata and data for:
- `harmonic_amp_5scenarios_log.png`
- `val_compare_reg_vs_bp_*.png` (v2g/g2v/g2b/b2g)

## Structure
- `data/`
  - `harmonic_amp_stats.csv`
  - `val_curve_<scenario>.csv`
- `metadata/`
  - `plot_metadata.json`
  - `export_manifest.json`
- `matlab/`
  - `run_all_plots.m`
  - `plot_harmonic_amp_5scenarios_log.m`
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
    with open(os.path.join(meta_dir, "plot_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    manifest = {
        "source_dir": source_dir,
        "bundle_dir": bundle_dir,
        "generated_files": {
            "metadata": [
                "metadata/plot_metadata.json",
                "metadata/export_manifest.json",
            ],
            "data": ["data/harmonic_amp_stats.csv"]
            + [f"data/val_curve_{s}.csv" for s in scenario_order],
            "matlab": [
                "matlab/run_all_plots.m",
                "matlab/plot_harmonic_amp_5scenarios_log.m",
                "matlab/plot_val_compare_reg_vs_bp_all.m",
            ],
            "reference_figures": [
                "reference_figures/harmonic_amp_5scenarios_log.png",
            ]
            + [f"reference_figures/val_compare_reg_vs_bp_{s}.png" for s in scenario_order],
        },
    }
    with open(os.path.join(meta_dir, "export_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_matlab_scripts(matlab_dir)
    write_readme(bundle_dir)
    print(f"MATLAB bundle exported: {bundle_dir}")


if __name__ == "__main__":
    main()
