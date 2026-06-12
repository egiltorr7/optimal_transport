% POSTPROC_OT_GRID_REFINEMENT  Publication-quality grid refinement study.
%
%  Problem  : Gaussian N(1/3, 0.05^2) -> N(2/3, 0.05^2),  eps = 0 (pure OT)
%  Reference: analytical_gaussian  (exact displacement interpolation)
%
%  Three sweeps:
%    1. Spatial   -- nx in [32,64,128,256,512],  nt = NT_FIX (256)
%    2. Temporal  -- nt in [16,32,64,128,256],   nx = NX_FIX (256)
%    3. Joint     -- nt = nx in [16,32,64,128,256]
%
%  Output (results/paper/):
%    ot_grid_convergence.pdf   1x3 log-log L2-error panel   [main figure]
%    ot_density_finest.pdf     density snapshots at nx=512, nt=256
%    LaTeX convergence table printed to terminal

clear; close all;
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup_paths.m'));

%% -----------------------------------------------------------------------
%% Global rendering: all text via LaTeX engine
%% -----------------------------------------------------------------------
set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultAxesTickLabelInterpreter',     'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

%% -----------------------------------------------------------------------
%% Parameters
%% -----------------------------------------------------------------------
NT_FIX = 256;   % fixed nt when sweeping nx
NX_FIX = 256;   % fixed nx when sweeping nt

nx_vals = [32,  64,  128, 256, 512];   % spatial sweep
nt_vals = [16,  32,  64,  128, 256];   % temporal sweep
nj_vals = [16,  32,  64,  128, 256];   % joint sweep  (nt = nx)

cfg_base        = cfg_ladmm_gaussian();
cfg_base.vareps = 0;   % pure OT
prob_def        = prob_gaussian();

KE_EXACT = (2/3 - 1/3)^2;   % W_2^2 = 1/9 (equal-sigma Gaussians)

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'paper');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%% Sweep 1: nx,  nt fixed
%% -----------------------------------------------------------------------
fprintf('\n=== Sweep 1: nx  (nt=%d fixed) ===\n', NT_FIX);
n1 = numel(nx_vals);
[err1, ke1] = deal(zeros(n1, 1));
[res1, prob1] = deal(cell(n1, 1));
for k = 1:n1
    cfg_k = cfg_base;  cfg_k.nt = NT_FIX;  cfg_k.nx = nx_vals(k);
    [err1(k), ke1(k), res1{k}, prob1{k}] = run_level(cfg_k, prob_def);
end

%% -----------------------------------------------------------------------
%% Sweep 2: nt,  nx fixed
%% -----------------------------------------------------------------------
fprintf('\n=== Sweep 2: nt  (nx=%d fixed) ===\n', NX_FIX);
n2 = numel(nt_vals);
[err2, ke2] = deal(zeros(n2, 1));
[res2, prob2] = deal(cell(n2, 1));
for k = 1:n2
    cfg_k = cfg_base;  cfg_k.nt = nt_vals(k);  cfg_k.nx = NX_FIX;
    [err2(k), ke2(k), res2{k}, prob2{k}] = run_level(cfg_k, prob_def);
end

%% -----------------------------------------------------------------------
%% Sweep 3: joint  nt = nx
%% -----------------------------------------------------------------------
fprintf('\n=== Sweep 3: joint  (nt = nx) ===\n');
n3 = numel(nj_vals);
[err3, ke3] = deal(zeros(n3, 1));
[res3, prob3] = deal(cell(n3, 1));
for k = 1:n3
    cfg_k = cfg_base;  cfg_k.nt = nj_vals(k);  cfg_k.nx = nj_vals(k);
    [err3(k), ke3(k), res3{k}, prob3{k}] = run_level(cfg_k, prob_def);
end

%% -----------------------------------------------------------------------
%% Figure 1: 1x3 convergence panel
%% -----------------------------------------------------------------------
% Style parameters — tweak these to taste
C_DATA = [0.00, 0.45, 0.70];   % blue       – numerical data
C_H2   = [0.40, 0.40, 0.40];   % grey       – 2nd-order reference slope
C_H1   = [0.10, 0.10, 0.10];   % near-black – 1st-order reference slope

FS   = 11;    % base font size [pt]
LW_D = 2.0;   % data line width
LW_R = 1.4;   % reference slope line width
MS   = 7;     % marker size
FW   = 24.0;  % figure width  [cm]  (scale down with \includegraphics in paper)
FH   =  8.0;  % figure height [cm]  (~4:3 per panel at 24cm/3 wide)

fig1 = figure('Units','centimeters','Position',[2, 2, FW, FH], ...
              'PaperUnits','centimeters','PaperSize',[FW, FH]);
tl   = tiledlayout(1, 3, 'TileSpacing','compact', 'Padding','loose');

% ---- Panel (a): spatial refinement ------------------------------------
nexttile;
loglog(nx_vals, err1, 'o-', 'Color',C_DATA, 'LineWidth',LW_D, ...
       'MarkerFaceColor',C_DATA, 'MarkerSize',MS);
hold on;
loglog(nx_vals, err1(1)*(nx_vals(1)./nx_vals).^2, '--','Color',C_H2,'LineWidth',LW_R);
loglog(nx_vals, err1(1)*(nx_vals(1)./nx_vals).^1, ':' ,'Color',C_H1,'LineWidth',LW_R);
set(gca,'FontSize',FS,'Box','on','TickDir','out','XMinorTick','off');
xlabel(sprintf('$n_x$  ($n_t{=}%d$)', NT_FIX), 'FontSize',FS);
ylabel('$\|(\tilde{\rho}_h,b_h) - (\tilde{\rho}^*\!,b^*)\|_{L^2}$', 'FontSize',FS);
legend('LADMM','$\mathcal{O}(n_x^{-2})$','$\mathcal{O}(n_x^{-1})$', ...
       'FontSize',FS,'Location','southwest','Box','off');
grid on;
text(0.05,0.95,'(a)','Units','normalized','FontSize',FS+1,'FontWeight','bold', ...
     'VerticalAlignment','top');

% ---- Panel (b): temporal refinement -----------------------------------
nexttile;
loglog(nt_vals, err2, 's-', 'Color',C_DATA, 'LineWidth',LW_D, ...
       'MarkerFaceColor',C_DATA, 'MarkerSize',MS);
hold on;
loglog(nt_vals, err2(1)*(nt_vals(1)./nt_vals).^2, '--','Color',C_H2,'LineWidth',LW_R);
loglog(nt_vals, err2(1)*(nt_vals(1)./nt_vals).^1, ':' ,'Color',C_H1,'LineWidth',LW_R);
set(gca,'FontSize',FS,'Box','on','TickDir','out','XMinorTick','off');
xlabel(sprintf('$n_t$  ($n_x{=}%d$)', NX_FIX), 'FontSize',FS);
legend('LADMM','$\mathcal{O}(n_t^{-2})$','$\mathcal{O}(n_t^{-1})$', ...
       'FontSize',FS,'Location','southwest','Box','off');
grid on;
text(0.05,0.95,'(b)','Units','normalized','FontSize',FS+1,'FontWeight','bold', ...
     'VerticalAlignment','top');

% ---- Panel (c): joint refinement (x-axis = n = nt = nx) --------------
N_vals = nj_vals .* nj_vals;   % kept for table rate computation
nexttile;
loglog(nj_vals, err3, '^-', 'Color',C_DATA, 'LineWidth',LW_D, ...
       'MarkerFaceColor',C_DATA, 'MarkerSize',MS);
hold on;
loglog(nj_vals, err3(1)*(nj_vals(1)./nj_vals).^2, '--','Color',C_H2,'LineWidth',LW_R);
loglog(nj_vals, err3(1)*(nj_vals(1)./nj_vals).^1, ':' ,'Color',C_H1,'LineWidth',LW_R);
set(gca,'FontSize',FS,'Box','on','TickDir','out','XMinorTick','off');
xlabel('$n = n_t = n_x$', 'FontSize',FS);
legend('LADMM','$\mathcal{O}(n^{-2})$','$\mathcal{O}(n^{-1})$', ...
       'FontSize',FS,'Location','southwest','Box','off');
grid on;
text(0.05,0.95,'(c)','Units','normalized','FontSize',FS+1,'FontWeight','bold', ...
     'VerticalAlignment','top');

save_fig(fig1, fullfile(out_dir, 'ot_grid_convergence.pdf'));

%% -----------------------------------------------------------------------
%% Figure 2: density snapshots at finest resolution (spatial sweep, nx=512)
%%
%% Exact solution : solid coloured lines
%% LADMM          : open coloured markers (no connecting line)
%% rho stored as probability mass (sums to 1); divide by dx for density
%% -----------------------------------------------------------------------
prob_f = prob1{end};   % nx=512, nt=NT_FIX
res_f  = res1{end};
[rho_a_f, ~] = analytical_gaussian(prob_f);

t_show = [0.10, 0.25, 0.50, 0.75, 0.90];
nshow  = numel(t_show);
cmap   = lines(nshow);

ntm_f  = prob_f.nt - 1;  dt_f = prob_f.dt;  dx_f = prob_f.dx;
nx_f   = prob_f.nx;       xx_f = prob_f.xx;
stride = max(1, floor(nx_f / 60));   % ~60 candidate marker locations per curve
THRESH = 0.0;                        % no threshold — show all markers

FW2 = 16.0;  % width  [cm]
FH2 = 10.5;  % height [cm]

fig2 = figure('Units','centimeters','Position',[2,10,FW2,FH2], ...
              'PaperUnits','centimeters','PaperSize',[FW2,FH2]);
hold on;

for j = 1:nshow
    ti = max(1, min(ntm_f, round(t_show(j) / dt_f)));
    rho_exact = rho_a_f(ti,:) / dx_f;
    rho_num   = res_f.rho_stag(ti,:) / dx_f;

    % Exact: solid line
    plot(xx_f, rho_exact, '-', 'Color',cmap(j,:), ...
        'LineWidth',LW_D, 'HandleVisibility','off');

    % LADMM: open markers only where the density is non-trivial
    idx_cand = 1:stride:nx_f;
    idx      = idx_cand(rho_exact(idx_cand) > THRESH * max(rho_exact));
    plot(xx_f(idx), rho_num(idx), 'o', 'Color',cmap(j,:), ...
        'MarkerSize',MS-1, 'MarkerFaceColor','none', 'LineWidth',0.9, ...
        'HandleVisibility','off');
end

% --- Legend: style row (exact vs LADMM) + time row ---
h_exact = plot(nan,nan, 'k-',  'LineWidth',LW_D);
h_num   = plot(nan,nan, 'ko',  'MarkerSize',MS-1, 'MarkerFaceColor','none', ...
               'LineWidth',0.9, 'LineStyle','none');
h_time  = gobjects(nshow,1);
for j = 1:nshow
    h_time(j) = plot(nan,nan, '-', 'Color',cmap(j,:), 'LineWidth',LW_D);
end

legend([h_exact; h_num; h_time], ...
    [{'Exact', 'LADMM'}, ...
     arrayfun(@(t) sprintf('$t=%.2f$',t), t_show, 'UniformOutput',false)], ...
    'Location','northwest', 'FontSize',FS, 'Box','off');

xlabel('$x$',         'FontSize',FS);
ylabel('$\tilde{\rho}(t,x)$', 'FontSize',FS);
title('Dynamics of the Density', 'FontSize',FS+1, 'Interpreter','latex');
set(gca, 'FontSize',FS, 'Box','on', 'TickDir','out');
grid on;
den_fname = sprintf('ot_density_finest_nt%d_nx%d.pdf', NT_FIX, nx_f);
save_fig(fig2, fullfile(out_dir, den_fname));

%% -----------------------------------------------------------------------
%% LaTeX convergence table
%% -----------------------------------------------------------------------
rate_fn = @(e, n) log(e(1:end-1)./e(2:end)) ./ log(n(2:end)./n(1:end-1));
r1 = rate_fn(err1, nx_vals);
r2 = rate_fn(err2, nt_vals);
r3 = rate_fn(err3, nj_vals);  % rate w.r.t. n = nt = nx

fprintf('\n%%%% ===== Copy into paper =====\n');
fprintf('\\begin{table}[htbp]\n');
fprintf('\\centering\n');
fprintf(['\\caption{Grid convergence for pure OT ($\\varepsilon=0$), ' ...
         'Gaussian $N(\\tfrac{1}{3},0.05^2)\\to N(\\tfrac{2}{3},0.05^2)$. ' ...
         '$\\|\\mathbf{e}\\|$ is the discrete $L^2$ norm of the ' ...
         'error in $(\\tilde{\\rho},b)$.}\n']);
fprintf('\\label{tab:ot_grid_convergence}\n');
fprintf('\\begin{tabular}{r c r | r c r | r c r}\n');
fprintf('\\toprule\n');
fprintf(['\\multicolumn{3}{c|}{Spatial ($n_t{=}%d$)} & ' ...
         '\\multicolumn{3}{c|}{Temporal ($n_x{=}%d$)} & ' ...
         '\\multicolumn{3}{c}{Joint ($n_t{=}n_x$)} \\\\\n'], NT_FIX, NX_FIX);
fprintf(['\\cmidrule(lr){1-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n']);
fprintf('$n_x$ & $\\|\\mathbf{e}\\|$ & rate & ');
fprintf('$n_t$ & $\\|\\mathbf{e}\\|$ & rate & ');
fprintf('$n{=}n_t{=}n_x$ & $\\|\\mathbf{e}\\|$ & rate \\\\\n');
fprintf('\\midrule\n');

for k = 1:max([n1, n2, n3])
    if k <= n1
        if k == 1, r1s = '---'; else, r1s = sprintf('%.2f', r1(k-1)); end
        col1 = sprintf('%4d & %.2e & %s', nx_vals(k), err1(k), r1s);
    else
        col1 = '     &          &    ';
    end
    if k <= n2
        if k == 1, r2s = '---'; else, r2s = sprintf('%.2f', r2(k-1)); end
        col2 = sprintf('%4d & %.2e & %s', nt_vals(k), err2(k), r2s);
    else
        col2 = '     &          &    ';
    end
    if k <= n3
        if k == 1, r3s = '---'; else, r3s = sprintf('%.2f', r3(k-1)); end
        col3 = sprintf('%4d & %.2e & %s', nj_vals(k), err3(k), r3s);
    else
        col3 = '     &          &    ';
    end
    fprintf('%s & %s & %s \\\\\n', col1, col2, col3);
end

fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n');

fprintf('\n%%%% KE_exact = 1/9 = %.8f\n', KE_EXACT);
fprintf('%%%% Spatial  finest (nx=%d): KE = %.6f\n', nx_vals(end), ke1(end));
fprintf('%%%% Temporal finest (nt=%d): KE = %.6f\n', nt_vals(end), ke2(end));
fprintf('%%%% Joint    finest (n=%d):   KE = %.6f\n', nj_vals(end), ke3(end));

%% -----------------------------------------------------------------------
%% Local functions
%% -----------------------------------------------------------------------
function [err_tot, ke_num, r, prob] = run_level(cfg, prob_def)
    prob          = setup_problem(cfg, prob_def);
    r             = cfg.pipeline(cfg, prob);
    [rho_a, mx_a] = analytical_gaussian(prob);
    dt = prob.dt;   dx = prob.dx;
    err_rho = norm(r.rho_stag(:) - rho_a(:)) * sqrt(dt * dx);
    err_mx  = norm(r.mx_stag(:)  - mx_a(:))  * sqrt(dt * dx);
    err_tot = sqrt(err_rho^2 + err_mx^2);
    ke_num  = compute_objective(r.rho_stag, r.mx_stag, prob) / dx;
    fprintf('  nt=%3d  nx=%3d  err=%.2e  KE=%.6f  iters=%d  %.1fs\n', ...
        cfg.nt, cfg.nx, err_tot, ke_num, r.iters, r.walltime);
end

function save_fig(fig, fpath)
    try
        exportgraphics(fig, fpath, 'ContentType','vector', 'BackgroundColor','none');
    catch
        % exportgraphics requires R2020a; fall back to print
        print(fig, fpath, '-dpdf', '-painters', '-r0');
    end
    fprintf('Saved: %s\n', fpath);
end

