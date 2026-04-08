function plot_harmonic_amp_5scenarios_log(script_dir)
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
