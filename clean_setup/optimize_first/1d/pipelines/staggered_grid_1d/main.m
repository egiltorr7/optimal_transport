%% Main Script the runs different algorithms for 1D OT Dynamic Formulation

clear

%% Configuration
% Test case name
test_case = 'gaussian';

% Grid in Space and Time (Staggered)
nt = 64; ntp = nt+1; ntm = nt-1;
dt = 1/nt;
t = linspace(0,1,ntp)';
nx = 128;
nxp = nx+1; nxm = nx-1;
dx = 1/nx;
x = linspace(0,1,nxp);
% xx = x(2:end-1);
xx = (x(2:end) + x(1:end-1))/2;

% Step Size
% Just doing ADMM so no restriction on step size
gamma = 10;
maxIter = 5000;
vareps = 1e-5;   % Parameter for Schrodinger Bridge

% Reference solution parameters
compute_reference = true;  % Set to true to compute and save reference solution
use_reference = true;       % Set to true to load and use reference solution for error tracking
maxIter_ref = 10000;       % Number of iterations for reference solution

%% Time Boundary Conditions for rho
% Gaussian to Gaussian
meanvalue0 = 1/3; sigma0 = 0.05;
meanvalue1 = 2/3; sigma1 = 0.05;
Normal = @(x,meanvalue,sigma) 1/(sqrt(2*pi)*sigma)*exp(-0.5*((x-meanvalue)/sigma).^2);
rho0 = Normal(xx,meanvalue0,sigma0);
rho1 = Normal(xx,meanvalue1,sigma1);

rho0 = rho0/sum(sum(rho0));
rho1 = rho1/sum(sum(rho1));

% Bimodal to Gaussian
% meanvalue1 = 0.5; sigma1 = 0.1;
% meanvalue0a = 0.1; meanvalue0b = 0.9;
% sigma0 = 0.05;
% Normal = @(x,meanvalue,sigma) 1/(sqrt(2*pi)*sigma)*exp(-0.5*((x-meanvalue)/sigma).^2);
% rho0 = Normal(x,meanvalue0a,sigma0) + Normal(x,meanvalue0b,sigma0) + 0.1;
% rho1 = Normal(x,meanvalue1,sigma1)+0.1;

%% Create data folder if it doesn't exist
data_folder = 'data';
if ~exist(data_folder, 'dir')
    mkdir(data_folder);
    fprintf('Created data folder: %s\n', data_folder);
end

%% Generate filename for reference solution
refsoln_filename = fullfile(data_folder, ...
                            sprintf('refsoln_%s_gamma%.1f_nx%d_nt%d.mat', ...
                            test_case, gamma, nx, nt));

%% Compute or load reference solution
if compute_reference
    fprintf('Computing reference solution with %d iterations...\n', maxIter_ref);

    % Set up options for reference solution
    opts_ref.nt = nt;
    opts_ref.nx = nx;
    opts_ref.maxIter = maxIter_ref;
    opts_ref.gamma = gamma;
    opts_ref.vareps = vareps;
    % opts_ref.tau = tau;

    % Compute reference solution
    [rho_star, mx_star, outs_ref] = sb1d_admm(rho0, rho1, opts_ref);

    % Save reference solution
    save(refsoln_filename, 'rho_star', 'mx_star', 'outs_ref', ...
         'gamma', 'nx', 'nt', 'test_case', 'rho0', 'rho1');
    fprintf('Reference solution saved to %s\n', refsoln_filename);

    % Plot convergence of reference solution
    figure;
    semilogy(2:maxIter_ref, outs_ref.residual_diff(2:end));
    xlabel('Iteration');
    ylabel('Residual');
    title(sprintf('Reference Solution Convergence (maxIter=%d)', maxIter_ref));
    grid on;
end

%% Run main optimization
opts.nt = nt;
opts.nx = nx;
opts.maxIter = maxIter;
opts.gamma = gamma;
opts.vareps = vareps;
% opts.tau = tau;

% Load reference solution if requested
if use_reference
    if ~exist(refsoln_filename, 'file')
        error('Reference solution file %s not found. Set compute_reference=true first.', refsoln_filename);
    end
    fprintf('Loading reference solution from %s...\n', refsoln_filename);
    ref_data = load(refsoln_filename);
    opts.rho_star = ref_data.rho_star;
    opts.mx_star = ref_data.mx_star;
end

[rho_admm, mx_admm, outs_admm] = sb1d_admm(rho0, rho1, opts);

%% Post Process

% Create a figure
figure;
h = plot(xx, rho0, 'LineWidth', 2);
ylim([0, max(rho_admm(:))*1.1]);  % set y-limits so they don't jump
xlabel('x'); ylabel('\rho');
title('Density transport');

% Optional: create a video writer
v = VideoWriter('rho_transport.mp4', 'MPEG-4');
v.FrameRate = 1;  % adjust as needed
open(v);

for i = 1:nt-1
    set(h, 'YData', rho_admm(i,:));  % update plot
    title(sprintf('Density transport, t = %d/%d', i, nt));
    drawnow;

    % write frame to video
    frame = getframe(gcf);
    writeVideo(v, frame);

    % optional: pause for visualization without saving
    % pause(0.05)
end

close(v);

%% Plot convergence
figure;
subplot(2,1,1);
semilogy(2:maxIter, outs_admm.residual_diff(2:end));
xlabel('Iteration');
ylabel('Residual Difference');
title('Convergence: Residual');
grid on;

if use_reference && isfield(outs_admm, 'true_error')
    subplot(2,1,2);
    semilogy(1:maxIter, outs_admm.true_error);
    xlabel('Iteration');
    ylabel('||u_k - u_*||');
    title('Convergence: True Error');
    grid on;
end
