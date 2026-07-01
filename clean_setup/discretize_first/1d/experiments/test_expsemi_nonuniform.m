% TEST_EXPSEMI_NONUNIFORM  Non-uniform ETD-ADMM vs Sinkhorn — grid comparison.
%
%   Runs Sinkhorn (reference) and three ETD-ADMM variants:
%     (1) Uniform grid
%     (2) Non-uniform power law  (pow_scale = 1, default)
%     (3) Non-uniform power law  (pow_scale = 2, steeper near t=1)
%
%   Figures (saved to results/figures/):
%     expsemi_nu_density_<tag>.png   -- density evolution (default non-uniform)
%     expsemi_nu_l2err_<tag>.png     -- relative L2 error vs time, all three grids
%     expsemi_nu_eps_sweep.png       -- max interior error vs eps (default grid)

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% --- Config ---
VAREPS = 100;
NT     = 512;
NX     = 256;

cfg_base        = cfg_ladmm_gaussian_expsemi_nonuniform();
cfg_base.vareps = VAREPS;
cfg_base.nt     = NT;
cfg_base.nx     = NX;

cfg_sink.vareps       = VAREPS;
cfg_sink.max_iter     = 50000;
cfg_sink.tol          = -1;
cfg_sink.precomp_heat = @precomp_heat_neumann;

prob_def = prob_gaussian();
problem  = setup_problem(cfg_base, prob_def);

nt  = problem.nt;   ntm = nt - 1;
dx  = problem.dx;
xx  = problem.xx;   nx  = problem.nx;
lam = problem.lambda_x(:);   % (nx x 1)

ftag = sprintf('nt%d_nx%d_eps%g', NT, NX, VAREPS);
FS = 11;   LW = 1.5;   MS = 4;

%% --- Sinkhorn reference ---
fprintf('Running Sinkhorn (eps=%.4g)...\n', VAREPS);
res_sink  = sinkhorn_hopf_cole(problem, cfg_sink);
fprintf('  iters=%d  converged=%d  wall=%.1fs\n', ...
    res_sink.iters, res_sink.converged, res_sink.walltime);

phi_0_hat = dct(res_sink.phi(1,:)');
psi_T_hat = dct(res_sink.psi(end,:)');

%% --- Grid variants ---
grids(1).label    = 'Uniform';
grids(1).tgrid_fn = @make_uniform_tgrid;
grids(1).color    = [0.80 0.15 0.15];
grids(1).marker   = 's';

grids(2).label    = 'Non-uniform ($p_0$)';
grids(2).tgrid_fn = @make_nonuniform_tgrid;
grids(2).color    = [0.10 0.55 0.10];
grids(2).marker   = 'o';

grids(3).label    = 'Non-uniform ($2p_0$)';
grids(3).tgrid_fn = @(n, e) make_nonuniform_tgrid(n, e, 2);
grids(3).color    = [0.10 0.25 0.80];
grids(3).marker   = '^';

NG = numel(grids);

%% --- Run each variant and compute errors ---
for g = 1:NG
    cfg_g          = cfg_base;
    cfg_g.tgrid_fn = grids(g).tgrid_fn;

    fprintf('Running ETD-ADMM [%s] (nt=%d)...\n', grids(g).label, NT);
    res_g = discretize_then_optimize_nonuniform(cfg_g, problem);
    fprintf('  iters=%d  converged=%d  wall=%.1fs  res=%.2e\n', ...
        res_g.iters, res_g.converged, res_g.walltime, res_g.error);

    t_stag = res_g.t_grid(2:nt);                        % (ntm x 1)
    dfwd   = exp(-VAREPS * lam * t_stag');               % (nx x ntm)
    dbwd   = exp(-VAREPS * lam * (1 - t_stag)');        % (nx x ntm)
    ref_g  = (max(idct(phi_0_hat .* dfwd), 0) .* ...
               max(idct(psi_T_hat .* dbwd), 0))';       % (ntm x nx)

    err_g  = sqrt(dx * sum((res_g.rho_stag - ref_g).^2, 2));
    nrm_g  = sqrt(dx * sum(ref_g.^2, 2));

    grids(g).res    = res_g;
    grids(g).t_stag = t_stag;
    grids(g).err    = err_g;
    grids(g).rel    = err_g ./ nrm_g;
end

% Convenience aliases for the default non-uniform grid (used in Fig 1 & diagnostics)
res_nu    = grids(2).res;
t_stag_nu = grids(2).t_stag;
err_nu    = grids(2).err;
rel_nu    = grids(2).rel;

%% --- Figure 1: density evolution (default non-uniform grid) ---
t_fracs = [0.1, 0.25, 0.5, 0.75, 0.9];
n_t     = numel(t_fracs);
cmap    = lines(n_t);
stride  = max(1, floor(nx / 60));

fig1 = figure('Units','centimeters','Position',[2 2 18 11]);
hold on;

for p = 1:n_t
    t_target = t_fracs(p);
    [~, k_nu] = min(abs(t_stag_nu - t_target));
    t_k   = t_stag_nu(k_nu);
    phi_k = max(idct(exp(-VAREPS * lam * t_k)     .* phi_0_hat), 0);
    psi_k = max(idct(exp(-VAREPS * lam * (1-t_k)) .* psi_T_hat), 0);
    rho_sink_k = (phi_k .* psi_k)';
    idx  = 1:stride:nx;
    col  = cmap(p,:);
    plot(xx, rho_sink_k,                              '-',  'Color', col, 'LineWidth', LW, ...
        'HandleVisibility', 'off');
    plot(xx(idx), res_nu.rho_stag(k_nu, idx), 'o', 'Color', col, ...
        'MarkerSize', MS, 'MarkerFaceColor', 'none', 'LineWidth', 0.9, ...
        'HandleVisibility', 'off');
end

h1 = plot(nan,nan, 'k-',  'LineWidth', LW,  'DisplayName', 'Sinkhorn');
h2 = plot(nan,nan, 'ko',  'MarkerSize', MS, 'MarkerFaceColor','none', ...
    'LineWidth', 0.9, 'DisplayName', 'Non-uniform ETD ($p_0$)');
h_t = gobjects(n_t, 1);
for p = 1:n_t
    h_t(p) = plot(nan,nan, '-', 'Color', cmap(p,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$t\\approx%.2f$', t_fracs(p)));
end

xlabel('$x$', 'FontSize', FS);
ylabel('$\tilde{\rho}(t,x)$', 'FontSize', FS);
title(sprintf('Density: Non-uniform ETD-ADMM vs Sinkhorn  ($\\varepsilon=%.4g$)', VAREPS), ...
    'FontSize', FS);
legend([h1; h2; h_t], 'Location','best', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;
saveas(fig1, fullfile(fig_dir, sprintf('expsemi_nu_density_%s.png', ftag)));

%% --- Figure 2: relative L2 error vs time, all three grid variants ---
fig2 = figure('Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1); hold on;
for g = 1:NG
    semilogy(grids(g).t_stag, grids(g).err, '.', ...
        'Color', grids(g).color, 'MarkerSize', 3, ...
        'DisplayName', grids(g).label);
end
xlabel('$t$', 'FontSize', FS);
ylabel('$\|\tilde{\rho} - \rho_\mathrm{Sink}\|_{L^2(x)}$', 'FontSize', FS);
title(sprintf('Absolute $L^2$ error  ($\\varepsilon=%.4g$, $N_T=%d$)', VAREPS, NT), 'FontSize', FS);
legend('Location','northwest', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

subplot(1,2,2); hold on;
for g = 1:NG
    semilogy(grids(g).t_stag, grids(g).rel, '.', ...
        'Color', grids(g).color, 'MarkerSize', 3, ...
        'DisplayName', grids(g).label);
end
xlabel('$t$', 'FontSize', FS);
ylabel('$\|\tilde{\rho} - \rho_\mathrm{Sink}\|_{L^2} / \|\rho_\mathrm{Sink}\|_{L^2}$', 'FontSize', FS);
title(sprintf('Relative $L^2$ error  ($\\varepsilon=%.4g$, $N_T=%d$)', VAREPS, NT), 'FontSize', FS);
legend('Location','northwest', 'FontSize', FS-1, 'Box','off');
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig2, fullfile(fig_dir, sprintf('expsemi_nu_l2err_%s.png', ftag)));

%% --- Summary (default non-uniform grid) ---
err_int = err_nu(1:ntm-1);
rel_int = rel_nu(1:ntm-1);

fprintf('\n--- Summary (eps=%.4g, nt=%d, nx=%d) ---\n', VAREPS, NT, NX);
fprintf('  Sinkhorn:        wall=%.2fs  iters=%d\n', res_sink.walltime, res_sink.iters);
for g = 1:NG
    r = grids(g).res;
    fprintf('  %-26s  wall=%.2fs  iters=%d  conv=%d  res=%.2e\n', ...
        grids(g).label, r.walltime, r.iters, r.converged, r.error);
end
fprintf('  Default grid (p0): max_abs=%.3e  max_rel=%.3e  (interior: %.3e / %.3e)\n', ...
    max(err_nu), max(rel_nu), max(err_int), max(rel_int));

fprintf('\n--- Last-stagger diagnostics for default grid (last 5 points) ---\n');
fprintf('  %-12s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
    't', 'eps*dt_n', 'abs_err', 'nrm_ref', 'rel_err', 'dt_n');
nrm_nu = grids(2).err ./ grids(2).rel;
for kk = max(1, ntm-4):ntm
    t_k  = t_stag_nu(kk);
    dt_k = res_nu.dt_vec(kk+1);
    fprintf('  %-12.6f  %-10.2e  %-10.2e  %-10.2e  %-10.2e  %-10.2e\n', ...
        t_k, VAREPS*dt_k, err_nu(kk), nrm_nu(kk), rel_nu(kk), dt_k);
end
fprintf('  (threshold dt_n < sigma^2/eps = %.2e for <5%% error)\n', 0.15^2 / VAREPS);

%% --- Figure 3: max interior error vs eps sweep (default non-uniform grid) ---
eps_vals    = [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0, 50.0, 100.0];
ne          = numel(eps_vals);
max_abs_sw  = nan(1, ne);
max_rel_sw  = nan(1, ne);

fprintf('\nEps sweep [Non-uniform p0] (NT=%d, NX=%d):\n', NT, NX);
fprintf('  %-8s  %-14s  %-14s  %-12s  %-6s  %-8s  %-10s  %-12s\n', ...
    'eps', 'max_abs_int', 'max_rel_int', 'last_rel', 'conv', 'iters', 'final_res', 'eps*dt_min');

for i = 1:ne
    eps_i = eps_vals(i);

    cfg_sw_i        = cfg_base;
    cfg_sw_i.vareps = eps_i;

    cfg_sk_i.vareps       = eps_i;
    cfg_sk_i.max_iter     = 500;
    cfg_sk_i.tol          = 1e-10;
    cfg_sk_i.precomp_heat = @precomp_heat_neumann;

    prob_i   = setup_problem(cfg_sw_i, prob_def);
    res_sk_i = sinkhorn_hopf_cole(prob_i, cfg_sk_i);
    res_sw_i = discretize_then_optimize_nonuniform(cfg_sw_i, prob_i);

    nt_i   = prob_i.nt;   dx_i = prob_i.dx;   ntm_i = nt_i - 1;
    lam_i  = prob_i.lambda_x(:);
    t_si   = res_sw_i.t_grid(2:nt_i);

    ph_hat_i = dct(res_sk_i.phi(1,:)');
    ps_hat_i = dct(res_sk_i.psi(end,:)');
    ref_i    = (max(idct(ph_hat_i .* exp(-eps_i * lam_i * t_si')),     0) .* ...
                max(idct(ps_hat_i .* exp(-eps_i * lam_i * (1-t_si)')), 0))';

    e_i      = sqrt(dx_i * sum((res_sw_i.rho_stag - ref_i).^2, 2));
    nrm_i    = sqrt(dx_i * sum(ref_i.^2, 2));
    rel_i    = e_i ./ nrm_i;

    max_abs_sw(i) = max(e_i(1:ntm_i-1));
    max_rel_sw(i) = max(rel_i(1:ntm_i-1));
    last_rel_i    = rel_i(ntm_i);
    dt_min_i      = min(res_sw_i.dt_vec);

    fprintf('  %-8g  %-14.2e  %-14.2e  last_rel=%.2e  conv=%d  iters=%-5d  res=%.1e  eps*dtmin=%.2e\n', ...
        eps_i, max_abs_sw(i), max_rel_sw(i), last_rel_i, res_sw_i.converged, res_sw_i.iters, ...
        res_sw_i.error, eps_i * dt_min_i);
end

fig3 = figure('Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1);
loglog(eps_vals, max_abs_sw, 'r^-', 'LineWidth', LW, 'MarkerSize', 6);
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_{t<1} \|\tilde{\rho} - \rho_\mathrm{Sink}\|_{L^2(x)}$', 'FontSize', FS);
title(sprintf('Max interior abs error vs $\\varepsilon$  ($N_T=%d$, $N_x=%d$)', NT, NX), 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

subplot(1,2,2);
loglog(eps_vals, max_rel_sw, 'b^-', 'LineWidth', LW, 'MarkerSize', 6);
xlabel('$\varepsilon$', 'FontSize', FS);
ylabel('$\max_{t<1} \|\tilde{\rho} - \rho_\mathrm{Sink}\|_{L^2} / \|\rho_\mathrm{Sink}\|_{L^2}$', 'FontSize', FS);
title(sprintf('Max interior rel error vs $\\varepsilon$  ($N_T=%d$, $N_x=%d$)', NT, NX), 'FontSize', FS);
set(gca, 'FontSize', FS, 'Box','on', 'TickDir','out');
grid on;

saveas(fig3, fullfile(fig_dir, 'expsemi_nu_eps_sweep.png'));
