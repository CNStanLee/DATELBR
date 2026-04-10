function plot_harmonic_amp_5scenarios_with_4method_errorbars_linear(script_dir)
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
