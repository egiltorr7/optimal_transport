%% Main script for Dynamic Formulation of Optimal Transport (FEA Formulation)
% Collocated Grid
% 2D

clear
post_process = true;
tau = 0.1;
GPUcomputing = true;  
gpu_device = 1;
folder = "./data/";
case_name = "gaussian_";

nt = 63;
ntp = nt+1;
dt = 1/ntp;
% t = linspace(0,1-dt,ntp);
% t = t';
maxIter = 10000;


% nx = nt;
nx = 127;
nxp = nx+1;
dx = 1/nxp;
x = linspace(0,1-dx,nxp); 
x = x';

% ny = 127;
ny = nx;
nyp = ny+1;
dy = 1/nyp;
y = linspace(0,1-dy,nyp);

grid_name = sprintf('tau%0.2f_nt%d_nx%d_ny%d', tau, nt, nx, ny);
file_name = folder + case_name + grid_name;

% Initial Distributions
rho0 = zeros(nxp,nyp);
rho1 = zeros(nxp,nyp);
rho0((x-0.25).^2 + (y-0.25).^2 < 0.01) = 1;
rho1((x-0.75).^2 + (y-0.75).^2 < 0.01) = 1;

% Quadrature Vector
rho0_mass = sum(rho0,'all')*dx*dy;
rho1_mass = sum(rho1,'all')*dx*dy;
%rho0 = rho0/rho0_mass;
%rho1 = rho1/rho1_mass;
rho1  = rho1*(rho0_mass/rho1_mass);

% mass_check = abs((rho0-rho1)*quadrature_x);

% if (mass_check > 1e-13)
%     disp("WARNING: Mass does not match in terms of Quadrature chosen")
% end

% Options
opts.GPUcomputing = GPUcomputing;
opts.tol = 1e-10;
opts.nt = nt;
opts.nx = nx;
opts.ny = ny;
opts.maxIter = maxIter;
% opts.sub_maxit = 1;
opts.tau = tau;
% opts.rho_analytical = rho_analytical;
% opts.mx_analytical = mx_analytical;
opts.rho_numerical = zeros(nt,nxp,nyp);
opts.mx_numerical = zeros(ntp,nx,nyp);
opts.my_numerical = zeros(ntp,nxp,ny);

if post_process
    s = load(file_name+".mat");
    opts.rho_numerical = s.rho_admm;
    opts.mx_numerical = s.mx_admm;
    opts.my_numerical = s.my_admm;
    maxIter = s.maxIter;
end

outs = 1;

if (GPUcomputing)
    fprintf('GPU loading matrices')
    Device = gpuDevice(gpu_device);
    rho0 = gpuArray(rho0);
    rho1 = gpuArray(rho1);
    opts.rho_numerical = gpuArray(opts.rho_numerical);
    opts.mx_numerical = gpuArray(opts.mx_numerical);
    opts.my_numerical = gpuArray(opts.my_numerical);
    opts.Device = Device;
end

% OT1D: ADMM on primal formulation
[rho_admm, mx_admm, my_admm, outs_admm] = ot2d_admm_primal(rho0, rho1, opts);
iterations = (1:opts.maxIter);
maxIter = opts.maxIter;
if post_process == false
    save(file_name+".mat", "rho_admm", "mx_admm", "my_admm", "maxIter")
else
    save(file_name+"_post.mat", "iterations", "outs_admm")
end

%% Post Processing
% figure (1);
% iterations = (1:opts.maxIter);
% semilogy(iterations,outs_admm.residual_diff,'LineWidth',2,...
%     'DisplayName', sprintf('nt = %i, nx = %i',nt,nx))
% xlabel('Iterations'); 
% ylabel('$\| (\rho,m)^k - (\rho,m)^{k-1}\|_2$','Interpreter', 'latex');
% legend show;
% % 
% % figure (2);
% % semilogy(iterations,outs_admm_primal.residual_analytical,'LineWidth',2,...
% %     'DisplayName', sprintf('nt = %i, nx = %i',nt,nx))
% % xlabel('Iterations')
% % ylabel('$\| (\rho,m)^k - (\rho,m)^{*}\|_2$','Interpreter', 'latex')
% % title("Analytical Optimizer")
% % 
% figure (3);
% semilogy(iterations,outs_admm.residual_numerical,'LineWidth',2,...
%     'DisplayName', sprintf('nt = %i, nx = %i',nt,nx))
% xlabel('Iterations')
% ylabel('$\| (\rho,m)^k - (\rho,m)^{*}\|_2$','Interpreter', 'latex')
% title("Numerical Optimizer")
% 
% 
