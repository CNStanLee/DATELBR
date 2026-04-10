function plot_harmonic_mae_4methods_log(script_dir)
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
