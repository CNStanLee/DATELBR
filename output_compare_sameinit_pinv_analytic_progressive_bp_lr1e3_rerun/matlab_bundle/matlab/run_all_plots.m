function run_all_plots()
% One-click plotting for exported metadata/data.
this_dir = fileparts(mfilename('fullpath'));
plot_harmonic_amp_5scenarios_log(this_dir);
plot_harmonic_mae_4methods_log(this_dir);
plot_harmonic_amp_5scenarios_with_4method_errorbars_linear(this_dir);
plot_val_compare_reg_vs_bp_all(this_dir);
disp('All MATLAB plots generated.');
end
