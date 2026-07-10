% FIG_ETD1_ERROR_VS_TIME_EPS_SWEEP
%   L2-in-x error vs time for ETD1-ADMM at four epsilon values.
%   Reference: analytical SB for eps <= ANA_EPS, Sinkhorn (Neumann) otherwise.
%
%   Error at each staggered time t_k:
%     e(k) = || rho_etd(k,:) - rho_ref(k,:) ||_{L2(x)}
%           = sqrt( dx * sum( (rho_etd(k,:) - rho_ref(k,:)).^2 ) )
%
%   Output (results/paper/):
%     sb_etd1_error_vs_time_nt<NT>_nx<NX>.pdf

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
eps_vals = [1e-4, 1e-2, 1, 100];
ANA_EPS  = 1e-3;

NT       = 256;
NX       = 256;
GAMMA    = 100;
TAU      = 101;
MAX_ITER = 10000;
TOL      = 1e-10;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Solve and compute error for each eps
%% -----------------------------------------------------------------------
ne       = numel(eps_vals);
prob_def = prob_gaussian();

cfg_base          = cfg_ladmm_gaussian_expsemi();
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.gamma    = GAMMA;
cfg_base.tau      = TAU;
cfg_base.max_iter = MAX_ITER;
cfg_base.tol      = TOL;

err_cell   = cell(ne, 1);
t_cell     = cell(ne, 1);
leg_labels = cell(ne, 1);

for i = 1:ne
    eps_i        = eps_vals(i);
    cfg_i        = cfg_base;
    cfg_i.vareps = eps_i;
    prob_i       = setup_problem(cfg_i, prob_def);

    fprintf('Running ETD1-ADMM  eps=%.1e ... ', eps_i);
    res = discretize_then_optimize(cfg_i, prob_i);
    fprintf('iters=%d  conv=%d  wall=%.1fs\n', res.iters, res.converged, res.walltime);

    ntm_i = prob_i.nt - 1;
    dx_i  = prob_i.dx;
    dt_i  = prob_i.dt;

    if eps_i <= ANA_EPS
        [rho_ref, ~] = analytical_sb_gaussian(prob_i, eps_i);
        ref_str      = 'Ana';
    else
        fprintf('  Sinkhorn reference    eps=%.1e ... ', eps_i);
        cfg_sk.vareps       = eps_i;
        cfg_sk.max_iter     = 500;
        cfg_sk.tol          = 1e-10;
        cfg_sk.precomp_heat = @precomp_heat_neumann;
        res_sk   = sinkhorn_hopf_cole(prob_i, cfg_sk);
        rho_ref  = res_sk.rho(2:prob_i.nt, :);
        ref_str  = 'Sink';
        fprintf('iters=%d  wall=%.1fs\n', res_sk.iters, res_sk.walltime);
    end

    err_cell{i} = sqrt(dx_i * sum((res.rho_stag - rho_ref).^2, 2));   % (ntm x 1)
    t_cell{i}   = (1:ntm_i)' * dt_i;

    e = round(log10(eps_i));
    if abs(log10(eps_i) - e) < 0.01
        leg_labels{i} = sprintf('$\\varepsilon=10^{%d}$ (%s)', e, ref_str);
    else
        leg_labels{i} = sprintf('$\\varepsilon=%g$ (%s)', eps_i, ref_str);
    end

    fprintf('  max err = %.2e\n', max(err_cell{i}));
end

%% -----------------------------------------------------------------------
%% Figure
%% -----------------------------------------------------------------------
FS  = 11;   LW  = 1.8;   MS  = 5;
FW  = 16.0; FH  = 10.5;

cols = lines(ne);

fig = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
             'PaperUnits','centimeters','PaperSize',[FW, FH]);
hold on;

for i = 1:ne
    semilogy(t_cell{i}, err_cell{i}, '-', ...
        'Color', cols(i,:), 'LineWidth', LW, 'DisplayName', leg_labels{i});
end

set(gca, 'YScale', 'log', 'FontSize', FS, 'Box', 'on', 'TickDir', 'out');
xlabel('$t$',                                                     'FontSize', FS);
ylabel('$\|\tilde{\rho}_\mathrm{ETD1}(t,\cdot) - \rho_\mathrm{ref}(t,\cdot)\|_{L^2(x)}$', ...
    'FontSize', FS);
title(sprintf('ETD1 error vs time  ($N_T=N_x=%d$)', NT),         'FontSize', FS);
legend('Location', 'best', 'FontSize', FS, 'Box', 'off');
xlim([0, 1]);
grid on;

fname = sprintf('sb_etd1_error_vs_time_nt%d_nx%d.pdf', NT, NX);
save_fig(fig, fullfile(out_dir, fname));

%% -----------------------------------------------------------------------
%% Local helpers
%% -----------------------------------------------------------------------
function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    catch
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end
