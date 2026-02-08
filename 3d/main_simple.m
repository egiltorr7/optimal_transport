%% Main script for Dynamic Formulation of Optimal Transport (FEA Formulation)
% Collocated Grid
% 3D

clear
post_process = true;
tau = 0.1;
GPUcomputing = true;
gpu_device = 1;
folder = "./data/";
case_name = "sphere_";

nt = 31;
ntp = nt+1;
dt = 1/ntp;
maxIter = 10000;

% Spatial grid
nx = 31;
nxp = nx+1;
dx = 1/nxp;
x = linspace(0,1-dx,nxp);
x = x';

ny = nx;
nyp = ny+1;
dy = 1/nyp;
y = linspace(0,1-dy,nyp);

nz = nx;
nzp = nz+1;
dz = 1/nzp;
z = linspace(0,1-dz,nzp);

grid_name = sprintf('tau%0.2f_nt%d_nx%d_ny%d_nz%d', tau, nt, nx, ny, nz);
file_name = folder + case_name + grid_name;

% Initial Distributions (3D Gaussians)
rho0 = zeros(nxp,nyp,nzp);
rho1 = zeros(nxp,nyp,nzp);
% Sphere at (0.25, 0.25, 0.25) and (0.75, 0.75, 0.75)
rho0((x-0.25).^2 + (y-0.25).^2 + (reshape(z,1,1,nzp)-0.25).^2 < 0.01) = 1;
rho1((x-0.75).^2 + (y-0.75).^2 + (reshape(z,1,1,nzp)-0.75).^2 < 0.01) = 1;

% Mass normalization
rho0_mass = sum(rho0,'all')*dx*dy*dz;
rho1_mass = sum(rho1,'all')*dx*dy*dz;
rho1  = rho1*(rho0_mass/rho1_mass);

% Options
opts.GPUcomputing = GPUcomputing;
opts.tol = 1e-10;
opts.nt = nt;
opts.nx = nx;
opts.ny = ny;
opts.nz = nz;
opts.maxIter = maxIter;
opts.tau = tau;
opts.rho_numerical = zeros(nt,nxp,nyp,nzp);
opts.mx_numerical = zeros(ntp,nx,nyp,nzp);
opts.my_numerical = zeros(ntp,nxp,ny,nzp);
opts.mz_numerical = zeros(ntp,nxp,nyp,nz);

if post_process
    s = load(file_name+".mat");
    opts.rho_numerical = s.rho_admm;
    opts.mx_numerical = s.mx_admm;
    opts.my_numerical = s.my_admm;
    opts.mz_numerical = s.mz_admm;
    maxIter = s.maxIter;
end

outs = 1;

if (GPUcomputing)
    fprintf('GPU loading matrices\n')
    Device = gpuDevice(gpu_device);
    rho0 = gpuArray(rho0);
    rho1 = gpuArray(rho1);
    opts.rho_numerical = gpuArray(opts.rho_numerical);
    opts.mx_numerical = gpuArray(opts.mx_numerical);
    opts.my_numerical = gpuArray(opts.my_numerical);
    opts.mz_numerical = gpuArray(opts.mz_numerical);
    opts.Device = Device;
end

% OT3D: ADMM on primal formulation
[rho_admm, mx_admm, my_admm, mz_admm, outs_admm] = ot3d_admm_primal(rho0, rho1, opts);
iterations = (1:opts.maxIter);
maxIter = opts.maxIter;
if post_process == false
    save(file_name+".mat", "rho_admm", "mx_admm", "my_admm", "mz_admm", "maxIter")
else
    save(file_name+"_post.mat", "iterations", "outs_admm")
end

fprintf('Simulation complete!\n');
