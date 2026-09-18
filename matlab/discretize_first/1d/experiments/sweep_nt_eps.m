% SWEEP_NT_EPS  Verify that max L2 error (ADMM vs Sinkhorn) scales as eps/NT.
%
%   For each (eps, NT) pair:
%     - Run CN-ADMM and Sinkhorn on the Gaussian SB problem
%     - Compute max_t ||rho_ADMM(t) - rho_Sink(t)||_{L2(x)}
%
%   Figures (saved to results/figures/):
%     sweep_nt_eps_raw.png    -- max error vs NT, one line per eps
%     sweep_nt_eps_scaled.png -- max error * NT / eps vs NT (should collapse)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% --- Sweep parameters ---
eps_vals = [0.1, 0.5, 1.0, 2.0, 4.0];
nt_vals  = [32, 64, 128, 256, 512];
NX       = 128;   % fixed spatial grid (spatial error constant across sweep)

ne  = numel(eps_vals);
nnt = numel(nt_vals);

%% --- Base configs ---
cfg_base            = cfg_ladmm_gaussian();
cfg_base.nx         = NX;
cfg_base.max_iter   = 20000;
cfg_base.tol        = 1e-9;

cfg_sink_base.max_iter     = 500;
cfg_sink_base.tol          = 1e-12;
cfg_sink_base.precomp_heat = @precomp_heat_neumann;

prob_def = prob_gaussian();

%% --- Run ---
max_err  = nan(ne, nnt);   % max_t L2 error vs Sinkhorn
max_berr = nan(ne, nnt);   % error at first interior time (t=dt, boundary layer)

total = ne * nnt;
run_idx = 0;

for i = 1:ne
    for j = 1:nnt
        run_idx = run_idx + 1;
        eps = eps_vals(i);
        NT  = nt_vals(j);

        fprintf('[%d/%d]  eps=%-5g  NT=%-4d  NX=%d  ...', ...
            run_idx, total, eps, NT, NX);

        cfg        = cfg_base;
        cfg.vareps = eps;
        cfg.nt     = NT;

        prob = setup_problem(cfg, prob_def);
        nt   = prob.nt;   ntm = nt - 1;
        dx   = prob.dx;

        % Sinkhorn reference
        cfg_sink        = cfg_sink_base;
        cfg_sink.vareps = eps;
        res_sink = sinkhorn_hopf_cole(prob, cfg_sink);

        % CN-ADMM
        res_admm = discretize_then_optimize(cfg, prob);

        % Grid-aligned comparison: rho_stag (ntm x NX) vs rho_sink(2:nt,:)
        rho_s = res_sink.rho(2:nt, :);
        diff  = res_admm.rho_stag - rho_s;
        err_t = sqrt(dx * sum(diff.^2, 2));   % (ntm x 1)

        max_err(i, j)  = max(err_t);
        max_berr(i, j) = max(err_t(1), err_t(end));   % boundary times only

        fprintf('  max_err=%.2e  iters=%d  conv=%d\n', ...
            max_err(i,j), res_admm.iters, res_admm.converged);
    end
end

%% --- Figure 1: max error vs NT, one line per eps ---
FS = 11;   LW = 1.8;
cmap = lines(ne);

fig1 = figure('Units','centimeters','Position',[2 2 16 11]);
hold on;
set(gca, 'XScale', 'log', 'YScale', 'log');

for i = 1:ne
    plot(nt_vals, max_err(i,:), 'o-', 'Color', cmap(i,:), 'LineWidth', LW, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$\\varepsilon=%g$', eps_vals(i)));
end

% O(1/NT) reference anchored at (eps=1, NT=64)
ref_i   = find(eps_vals == 1.0);
ref_j   = find(nt_vals  == 64);
ref_val = max_err(ref_i, ref_j);
plot(nt_vals, ref_val * nt_vals(ref_j) ./ nt_vals, 'k--', 'LineWidth', 1.2, ...
    'DisplayName', '$\mathcal{O}(1/N_T)$');

xlabel('$N_T$', 'FontSize', FS);
ylabel('$\max_t \|\tilde{\rho} - \rho_\mathrm{Sink}\|_{L^2(x)}$', 'FontSize', FS);
title('Max $L^2$ error vs $N_T$  (CN-ADMM vs Sinkhorn)', 'FontSize', FS);
legend('Location','southwest', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig1, fullfile(fig_dir, 'sweep_nt_eps_raw.png'));

%% --- Figure 2: scaled error * NT / eps vs NT (should collapse to constant) ---
fig2 = figure('Units','centimeters','Position',[2 2 16 11]);
hold on;
set(gca, 'XScale', 'log');

for i = 1:ne
    scaled = max_err(i,:) .* nt_vals / eps_vals(i);
    plot(nt_vals, scaled, 'o-', 'Color', cmap(i,:), 'LineWidth', LW, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$\\varepsilon=%g$', eps_vals(i)));
end

xlabel('$N_T$', 'FontSize', FS);
ylabel('$\max_t \|\cdot\|_{L^2} \times N_T \,/\, \varepsilon$', 'FontSize', FS);
title('Scaled error (collapses to constant if error $\sim \varepsilon/N_T$)', 'FontSize', FS);
legend('Location','best', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig2, fullfile(fig_dir, 'sweep_nt_eps_scaled.png'));

%% --- Summary table ---
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('max L2 error (ADMM vs Sinkhorn)\n');
fprintf('%-8s', 'eps\NT');
fprintf('  NT=%-5d', nt_vals);
fprintf('\n%s\n', repmat('-', 1, 60));
for i = 1:ne
    fprintf('eps=%-5g', eps_vals(i));
    fprintf('  %-9.2e', max_err(i,:));
    fprintf('\n');
end
fprintf('%s\n', repmat('=', 1, 60));

fprintf('\nScaled error (max_err * NT / eps)  -- should be ~constant:\n');
fprintf('%-8s', 'eps\NT');
fprintf('  NT=%-5d', nt_vals);
fprintf('\n%s\n', repmat('-', 1, 60));
for i = 1:ne
    fprintf('eps=%-5g', eps_vals(i));
    scaled = max_err(i,:) .* nt_vals ./ eps_vals(i);
    fprintf('  %-9.2e', scaled);
    fprintf('\n');
end
