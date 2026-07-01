% TEST_SINKHORN_LOGDOMAIN_COMPARE  Vanilla vs log-domain Sinkhorn across eps.
%
%   Runs both solvers on the Gaussian SB problem for a sweep of epsilon
%   values.  Vanilla Sinkhorn suffers overflow/underflow for small epsilon;
%   log-domain is stable for all epsilon.
%
%   Figures (saved to results/figures/):
%     sink_logdom_density_eps<e>.png   -- density + convergence at each eps
%     sink_logdom_conv_sweep.png       -- convergence curves, all eps
%     sink_logdom_summary.png          -- iters and max error vs eps

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% --- Shared config ---
NT  = 256;
NX  = 256;
FS  = 11;   LW = 1.5;

EPS_VALS = [1e-5 1e-4 0.001, 0.01, 0.1, 1.0, 10.0];
NE = numel(EPS_VALS);

% Seed from cfg_ladmm_gaussian so setup_problem gets cfg.name and cfg.disc
cfg_base          = cfg_ladmm_gaussian();
cfg_base.nt       = NT;
cfg_base.nx       = NX;
cfg_base.max_iter = 2000;
cfg_base.tol      = 1e-10;

prob_def = prob_gaussian();

%% --- Run sweep ---
res_van = cell(NE, 1);
res_log = cell(NE, 1);

for i = 1:NE
    eps_i = EPS_VALS(i);
    fprintf('eps = %-8g  ', eps_i);

    cfg_i        = cfg_base;
    cfg_i.vareps = eps_i;
    problem_i    = setup_problem(cfg_i, prob_def);

    % Vanilla
    cfg_v = cfg_i;
    cfg_v.precomp_heat = @precomp_heat_neumann;
    rv = sinkhorn_hopf_cole(problem_i, cfg_v);
    res_van{i} = rv;
    fprintf('vanilla: iters=%-5d  conv=%d  err=%.1e  wall=%.2fs  |  ', ...
        rv.iters, rv.converged, rv.error, rv.walltime);

    % Log-domain with epsilon-annealing (eps_init=1 ensures overlap at start)
    cfg_l = cfg_i;
    cfg_l.precomp_heat  = @precomp_heat_neumann_log;
    cfg_l.eps_init      = max(1.0, eps_i);   % no-op when eps_i >= 1
    cfg_l.anneal_factor = 2;
    rl = sinkhorn_hopf_cole_logdomain(problem_i, cfg_l);
    res_log{i} = rl;
    fprintf('logdom:  iters=%-5d  conv=%d  err=%.1e  wall=%.2fs\n', ...
        rl.iters, rl.converged, rl.error, rl.walltime);
end

%% --- Figure 1: density + convergence at each eps ---
for i = 1:NE
    eps_i    = EPS_VALS(i);
    cfg_i    = cfg_base;   cfg_i.vareps = eps_i;
    problem_i = setup_problem(cfg_i, prob_def);
    rv = res_van{i};
    rl = res_log{i};

    nt_i = problem_i.nt;   dx_i = problem_i.dx;   nx_i = problem_i.nx;
    xx_i = problem_i.xx;   dt_i = problem_i.dt;

    % Analytical solution on edge grid
    mu0   = problem_i.mu0;   mu1   = problem_i.mu1;
    sigma = problem_i.sigma;
    alpha = sqrt(sigma^4 + eps_i^2) - sigma^2;
    Gauss = @(x, mu, sig) exp(-0.5*((x-mu)/sig).^2) / (sqrt(2*pi)*sig);
    t_edge = rv.t_grid;
    rho_ana = zeros(nt_i+1, nx_i);
    for k = 1:(nt_i+1)
        t_k    = t_edge(k);
        mu_k   = (1-t_k)*mu0 + t_k*mu1;
        sig_k  = sqrt(sigma^2 + 2*alpha*t_k*(1-t_k));
        row    = Gauss(xx_i, mu_k, sig_k);
        rho_ana(k,:) = row / sum(row);
    end

    stride = max(1, floor(nx_i / 60));
    n_t    = 5;
    t_fracs = linspace(0.1, 0.9, n_t);
    cmap   = lines(n_t);

    fig = figure('Units','centimeters','Position',[2 2 26 10]);
    tiledlayout(1, 3, 'TileSpacing','compact','Padding','compact');

    % Left: density — vanilla vs analytical
    nexttile; hold on;
    for p = 1:n_t
        [~, k] = min(abs(t_edge - t_fracs(p)));
        idx = 1:stride:nx_i;
        col = cmap(p,:);
        plot(xx_i, rho_ana(k,:), '-', 'Color', col, 'LineWidth', LW, 'HandleVisibility','off');
        plot(xx_i(idx), rv.rho(k,idx), 's', 'Color', col, 'MarkerSize', 3, ...
            'LineWidth', 0.8, 'HandleVisibility','off');
    end
    h1 = plot(nan,nan,'k-','LineWidth',LW,'DisplayName','Analytical');
    h2 = plot(nan,nan,'ks','MarkerSize',3,'DisplayName','Vanilla');
    legend([h1 h2],'Location','best','FontSize',FS-2,'Box','off');
    xlabel('$x$','FontSize',FS); ylabel('$\rho$','FontSize',FS);
    title(sprintf('Vanilla  ($\\varepsilon=%.3g$)', eps_i),'FontSize',FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    % Middle: density — log-domain vs analytical
    nexttile; hold on;
    for p = 1:n_t
        [~, k] = min(abs(t_edge - t_fracs(p)));
        idx = 1:stride:nx_i;
        col = cmap(p,:);
        plot(xx_i, rho_ana(k,:), '-', 'Color', col, 'LineWidth', LW, 'HandleVisibility','off');
        plot(xx_i(idx), rl.rho(k,idx), 'o', 'Color', col, 'MarkerSize', 3, ...
            'LineWidth', 0.8, 'HandleVisibility','off');
    end
    h3 = plot(nan,nan,'k-','LineWidth',LW,'DisplayName','Analytical');
    h4 = plot(nan,nan,'ko','MarkerSize',3,'DisplayName','Log-domain');
    legend([h3 h4],'Location','best','FontSize',FS-2,'Box','off');
    xlabel('$x$','FontSize',FS); ylabel('$\rho$','FontSize',FS);
    title(sprintf('Log-domain  ($\\varepsilon=%.3g$)', eps_i),'FontSize',FS);
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    % Right: convergence curves
    nexttile; hold on;
    if ~isempty(rv.errors)
        semilogy(rv.errors, 'r-', 'LineWidth', LW, 'DisplayName', 'Vanilla');
    end
    semilogy(rl.errors, 'b-', 'LineWidth', LW, 'DisplayName', 'Log-domain');
    if cfg_base.tol > 0
        yline(cfg_base.tol, 'k--', 'LineWidth', 0.8, 'HandleVisibility','off');
    end
    xlabel('Iteration','FontSize',FS);
    ylabel('Left-marginal $L^2$ error','FontSize',FS);
    title('Convergence','FontSize',FS);
    legend('Location','best','FontSize',FS-2,'Box','off');
    set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

    etag = strrep(sprintf('%.3g', eps_i), '.', 'p');
    saveas(fig, fullfile(fig_dir, sprintf('sink_logdom_density_eps%s.png', etag)));
    close(fig);
end

%% --- Figure 2: convergence curves overlaid, all eps ---
fig2 = figure('Units','centimeters','Position',[2 2 22 9]);
cmap2 = parula(NE);

subplot(1,2,1); hold on;
title('Vanilla — convergence','FontSize',FS);
for i = 1:NE
    rv = res_van{i};
    if ~isempty(rv.errors)
        semilogy(rv.errors, '-', 'Color', cmap2(i,:), 'LineWidth', LW, ...
            'DisplayName', sprintf('$\\varepsilon=%.3g$', EPS_VALS(i)));
    end
end
xlabel('Iteration','FontSize',FS);
ylabel('Left-marginal $L^2$ error','FontSize',FS);
legend('Location','best','FontSize',FS-2,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

subplot(1,2,2); hold on;
title('Log-domain — convergence','FontSize',FS);
for i = 1:NE
    rl = res_log{i};
    semilogy(rl.errors, '-', 'Color', cmap2(i,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('$\\varepsilon=%.3g$', EPS_VALS(i)));
end
xlabel('Iteration','FontSize',FS);
ylabel('Left-marginal $L^2$ error','FontSize',FS);
legend('Location','best','FontSize',FS-2,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

saveas(fig2, fullfile(fig_dir, 'sink_logdom_conv_sweep.png'));

%% --- Figure 3: iters and max error vs eps ---
iters_van = cellfun(@(r) r.iters,     res_van);
conv_van  = cellfun(@(r) r.converged, res_van);
err_van   = cellfun(@(r) r.error,     res_van);
iters_log = cellfun(@(r) r.iters,     res_log);
conv_log  = cellfun(@(r) r.converged, res_log);
err_log   = cellfun(@(r) r.error,     res_log);

fig3 = figure('Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1); hold on;
loglog(EPS_VALS, iters_van, 'rs-', 'LineWidth', LW, 'MarkerSize', 6, 'DisplayName', 'Vanilla');
loglog(EPS_VALS, iters_log, 'bo-', 'LineWidth', LW, 'MarkerSize', 6, 'DisplayName', 'Log-domain');
% Mark non-converged with open marker
nc_v = ~conv_van;   nc_l = ~conv_log;
if any(nc_v), loglog(EPS_VALS(nc_v), iters_van(nc_v), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility','off'); end
if any(nc_l), loglog(EPS_VALS(nc_l), iters_log(nc_l), 'bx', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility','off'); end
xlabel('$\varepsilon$','FontSize',FS);
ylabel('Sinkhorn iterations','FontSize',FS);
title(sprintf('Iterations to converge  ($N_T=%d$, $N_x=%d$)', NT, NX),'FontSize',FS);
legend('Location','best','FontSize',FS-1,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;
text(0.05, 0.08, '{\bf x} = did not converge', 'Units','normalized','FontSize',FS-2,'Interpreter','tex');

subplot(1,2,2); hold on;
loglog(EPS_VALS, err_van, 'rs-', 'LineWidth', LW, 'MarkerSize', 6, 'DisplayName', 'Vanilla');
loglog(EPS_VALS, err_log, 'bo-', 'LineWidth', LW, 'MarkerSize', 6, 'DisplayName', 'Log-domain');
xlabel('$\varepsilon$','FontSize',FS);
ylabel('Final left-marginal $L^2$ error','FontSize',FS);
title('Final residual','FontSize',FS);
legend('Location','best','FontSize',FS-1,'Box','off');
set(gca,'FontSize',FS,'Box','on','TickDir','out'); grid on;

saveas(fig3, fullfile(fig_dir, 'sink_logdom_summary.png'));

%% --- Text summary ---
fprintf('\n--- Summary (NT=%d, NX=%d, tol=%.1e) ---\n', NT, NX, cfg_base.tol);
fprintf('  %-10s  %-25s  %-25s\n', 'eps', 'Vanilla (iters/conv/err)', 'Log-domain (iters/conv/err)');
for i = 1:NE
    rv = res_van{i};   rl = res_log{i};
    fprintf('  %-10g  %5d / %d / %.1e              %5d / %d / %.1e\n', ...
        EPS_VALS(i), rv.iters, rv.converged, rv.error, ...
        rl.iters, rl.converged, rl.error);
end
