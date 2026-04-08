function plot_val_compare_reg_vs_bp_all(script_dir)
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
