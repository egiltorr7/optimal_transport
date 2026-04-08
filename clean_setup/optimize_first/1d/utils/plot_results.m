function plot_results(result, problem, cfg)
% PLOT_RESULTS  Standard diagnostic plots for an OT experiment result.
%
%   plot_results(result, problem, cfg)
%
%   Produces:
%     Figure 1 -- convergence (residual vs iteration)
%     Figure 2 -- final density field (rho as heatmap)
%     Figure 3 -- snapshots of rho at several time slices

    rho     = result.rho;
    residual = result.residual;
    iters   = result.iters;
    xx      = problem.xx;
    nt      = problem.nt;

    % --- Convergence ---
    figure;
    semilogy(2:iters, residual(2:iters));
    xlabel('Iteration'); ylabel('Residual');
    title(sprintf('Convergence: %s', cfg.name));
    grid on;

    % --- Density heatmap ---
    figure;
    imagesc(rho);
    xlabel('Space index'); ylabel('Time index');
    title(sprintf('\\rho(t,x): %s', cfg.name));
    colorbar;

    % --- Snapshots ---
    figure;
    t_idx = round(linspace(1, size(rho,1), min(6, size(rho,1))));
    hold on;
    for i = 1:length(t_idx)
        plot(xx, rho(t_idx(i),:), 'DisplayName', sprintf('t=%d', t_idx(i)));
    end
    legend('show');
    xlabel('x'); ylabel('\rho');
    title(sprintf('Density snapshots: %s', cfg.name));
    grid on;
end
