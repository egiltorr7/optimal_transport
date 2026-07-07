% TEST_EXPSEMI_2D  ETD exact-semigroup projection in 2D vs Sinkhorn.
%
%   Runs the ETD exact-semigroup ADMM (proj_fokker_planck_expsemi_gpu) on
%   two test problems and compares against a Sinkhorn reference across an
%   epsilon sweep.
%
%   Figures (saved to results/figures/):
%     expsemi2d_density_<prob>_eps<e>.pdf   -- density snapshots at t=0.1,0.5,0.9
%     expsemi2d_l2err_<prob>_eps<e>.pdf     -- L2 error vs time
%     expsemi2d_eps_sweep_<prob>.pdf        -- max relative error vs eps

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));
clear functions

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
fig_dir = fullfile(res_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultFigureVisible',            'off');
set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% -------------------------------------------------------------------------
%  Grid and solver settings
% -------------------------------------------------------------------------
NT = 256;
NX = 128;
NY = 128;

EPS_SWEEP = [0.01, 0.1, 1.0, 10.0, 100.0];
NE        = numel(EPS_SWEEP);

TOL_WORK = 1e-8;


cfg_es              = cfg_ladmm_gaussian_expsemi();
cfg_es.nt           = NT;
cfg_es.nx           = NX;
cfg_es.ny           = NY;
cfg_es.max_iter     = 10000;
cfg_es.tol          = TOL_WORK;
cfg_es.use_gpu      = true;
cfg_es.gpu_device   = 1;
cfg_es.print_every  = 2000;
cfg_es.projection   = @proj_fokker_planck_expsemi_gpu;

% Graphics constants
FS  = 11;   LW  = 1.6;   MS  = 5;
col_es   = [0.13 0.47 0.71];   % blue -> expsemi
col_sink = [0.50 0.50 0.50];   % grey -> sinkhorn

%% -------------------------------------------------------------------------
%  Run both test problems
% -------------------------------------------------------------------------
PROBLEMS = { prob_gaussian(),    'gaussian' ; ...
             prob_four_peaks(),  'four\_peaks' };

for p_idx = 1:size(PROBLEMS, 1)
    prob_def  = PROBLEMS{p_idx, 1};
    prob_name = PROBLEMS{p_idx, 2};
    prob_tag  = prob_def.name;

    fprintf('\n=== Problem: %s  (NT=%d, NX=%d, NY=%d) ===\n', ...
        prob_name, NT, NX, NY);

    max_err_es   = nan(NE, 1);
    wall_es      = nan(NE, 1);
    wall_sink    = nan(NE, 1);
    iters_es     = nan(NE, 1);

    for i = 1:NE
        eps_i = EPS_SWEEP(i);
        fprintf('\n  eps = %.4g\n', eps_i);

        cfg_es.vareps = eps_i;
        problem_i     = setup_problem(cfg_es, prob_def);

        nt = problem_i.nt;
        nx = problem_i.nx;
        ny = problem_i.ny;
        dx = problem_i.dx;
        dy = problem_i.dy;
        dt = problem_i.dt;

        % ---- Sinkhorn reference ----
        cfg_sk.vareps            = eps_i;
        cfg_sk.max_iter          = 5000;
        cfg_sk.tol               = 1e-10;
        cfg_sk.precomp_heat      = @precomp_heat_neumann_2d;
        cfg_sk.use_pdf_marginals = true;

        fprintf('    Sinkhorn  '); t0 = tic;
        res_sk       = sinkhorn_hopf_cole(problem_i, cfg_sk);
        wall_sink(i) = toc(t0);
        fprintf('iters=%d  conv=%d  wall=%.2fs\n', ...
            res_sk.iters, res_sk.converged, wall_sink(i));

        rho_sk_cc = 0.5 * (res_sk.rho(1:nt,:,:) + res_sk.rho(2:nt+1,:,:));

        % ---- ExpSemi ADMM ----
        fprintf('    ExpSemi   '); t0 = tic;
        res_es      = discretize_then_optimize(cfg_es, problem_i);
        wall_es(i)  = toc(t0);
        iters_es(i) = res_es.iters;
        fprintf('iters=%d  conv=%d  wall=%.2fs\n', ...
            res_es.iters, res_es.converged, wall_es(i));

        % ---- Errors vs Sinkhorn ----
        dV       = dx * dy;
        err_es_t = sqrt(dV * sum(sum( (res_es.rho_cc - rho_sk_cc).^2, 2), 3));
        nrm_sk_t = sqrt(dV * sum(sum( rho_sk_cc.^2, 2), 3));

        max_err_es(i) = max(err_es_t ./ nrm_sk_t);
        fprintf('    max rel L2 err:  expsemi=%.3e\n', max_err_es(i));

        % ---- Density snapshots at eps=0.1 ----
        if i == 2
            t_fracs = [0.1, 0.5, 0.9];
            n_snap  = numel(t_fracs);

            fig_d = figure('Units', 'centimeters', 'Position', [2 2 16 9]);
            tl_d  = tiledlayout(2, n_snap, 'TileSpacing', 'compact', 'Padding', 'compact');

            clim_max = max(rho_sk_cc(:));

            for s = 1:n_snap
                k   = max(1, round(t_fracs(s) * nt));
                t_k = (k - 0.5) * dt;

                ax = nexttile(tl_d, s);
                imagesc(problem_i.xx, problem_i.yy, squeeze(rho_sk_cc(k,:,:))');
                axis xy; colorbar; clim([0, clim_max]);
                title(sprintf('Sinkhorn,  $t=%.2f$', t_k), 'FontSize', FS);
                if s == 1, ylabel('$y$', 'FontSize', FS); end
                set(ax, 'FontSize', FS-1, 'TickDir', 'out');

                ax = nexttile(tl_d, n_snap + s);
                imagesc(problem_i.xx, problem_i.yy, squeeze(res_es.rho_cc(k,:,:))');
                axis xy; colorbar; clim([0, clim_max]);
                title(sprintf('ExpSemi,  $t=%.2f$', t_k), 'FontSize', FS);
                xlabel('$x$', 'FontSize', FS);
                if s == 1, ylabel('$y$', 'FontSize', FS); end
                set(ax, 'FontSize', FS-1, 'TickDir', 'out');
            end
            sgtitle(sprintf('%s  ($\\varepsilon = %.4g$,  $N_T=%d$,  $N_x=N_y=%d$)', ...
                strrep(prob_name, '_', '\_'), eps_i, NT, NX), 'FontSize', FS+1);

            fname = sprintf('expsemi2d_density_%s_eps%g', prob_tag, eps_i);
            print(fig_d, fullfile(fig_dir, fname), '-dpdf', '-painters');
            saveas(fig_d, fullfile(fig_dir, [fname '.png']));
            close(fig_d);

            % ---- L2 error vs time ----
            t_cc = ((1:nt)' - 0.5) * dt;

            fig_e = figure('Units', 'centimeters', 'Position', [2 2 16 7]);
            tl_e  = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

            nexttile;
            semilogy(t_cc, err_es_t, '-', 'Color', col_es, 'LineWidth', LW);
            xlabel('$t$', 'FontSize', FS);
            ylabel('$\|\rho - \rho_{\rm Sink}\|_{L^2(x,y)}$', 'FontSize', FS);
            title('(a) Absolute $L^2$ error', 'FontSize', FS);
            set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out'); grid on;

            nexttile;
            semilogy(t_cc, err_es_t ./ nrm_sk_t, '-', 'Color', col_es, 'LineWidth', LW);
            xlabel('$t$', 'FontSize', FS);
            ylabel('$\|\rho - \rho_{\rm Sink}\|_{L^2} / \|\rho_{\rm Sink}\|_{L^2}$', 'FontSize', FS);
            title('(b) Relative $L^2$ error', 'FontSize', FS);
            set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out'); grid on;

            sgtitle(sprintf('%s  ($\\varepsilon = %.4g$)', strrep(prob_name,'_','\_'), eps_i), ...
                'FontSize', FS+1);

            fname = sprintf('expsemi2d_l2err_%s_eps%g', prob_tag, eps_i);
            print(fig_e, fullfile(fig_dir, fname), '-dpdf', '-painters');
            saveas(fig_e, fullfile(fig_dir, [fname '.png']));
            close(fig_e);
        end
    end   % eps sweep

    % ---- Eps sweep figure ----
    fig_s = figure('Units', 'centimeters', 'Position', [2 2 16 7]);
    tl_s  = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    loglog(EPS_SWEEP, max_err_es, '-^', 'Color', col_es, 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', col_es, 'DisplayName', 'ExpSemi (ETD)');
    xlabel('$\varepsilon$', 'FontSize', FS);
    ylabel('$\max_t \|\rho - \rho_{\rm Sink}\|_{L^2} / \|\rho_{\rm Sink}\|_{L^2}$', 'FontSize', FS);
    title('(a) Max relative $L^2$ error', 'FontSize', FS);
    set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out'); grid on;

    nexttile;
    loglog(EPS_SWEEP, wall_es,   '-^', 'Color', col_es,   'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', col_es,   'DisplayName', 'ExpSemi (ETD)');
    hold on;
    loglog(EPS_SWEEP, wall_sink, '-s', 'Color', col_sink, 'LineWidth', LW, ...
        'MarkerSize', MS, 'MarkerFaceColor', col_sink, 'DisplayName', 'Sinkhorn');
    xlabel('$\varepsilon$', 'FontSize', FS);
    ylabel('Wall time (s)', 'FontSize', FS);
    title('(b) Computational cost', 'FontSize', FS);
    legend('Location', 'best', 'FontSize', FS-1, 'Box', 'off');
    set(gca, 'FontSize', FS, 'Box', 'on', 'TickDir', 'out'); grid on;

    sgtitle(sprintf('%s  ($N_T=%d$, $N_x=N_y=%d$)', ...
        strrep(prob_name, '_', '\_'), NT, NX), 'FontSize', FS+1);

    fname = sprintf('expsemi2d_eps_sweep_%s_nt%d_nx%d', prob_tag, NT, NX);
    print(fig_s, fullfile(fig_dir, fname), '-dpdf', '-painters');
    saveas(fig_s, fullfile(fig_dir, [fname '.png']));
    close(fig_s);
    fprintf('\nSweep figure saved: %s\n', fname);

    save(fullfile(res_dir, sprintf('expsemi2d_sweep_%s_nt%d_nx%d.mat', prob_tag, NT, NX)), ...
        'EPS_SWEEP', 'NT', 'NX', 'NY', 'max_err_es', 'wall_es', 'wall_sink', 'iters_es');
end   % problem loop
