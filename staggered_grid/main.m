%% Main Script the runs different algorithms for 1D OT Dynamic Formulation

clear
% % Grid in Space and Time 

% (Staggered)
nt = 4; ntp = nt+1; ntm = nt-1;
dt = 1/nt;
t = linspace(0,1,ntp)';
nx = 4;
nxp = nx+1; nxm = nx-1;
dx = 1/nx;
x = linspace(0,1,nxp);
xx = (x(1:end-1)+x(2:end))/2;

% Step Size
gamma = 0.1;
tau   = 0.1;
maxIter = 1000;

% Time Boundary Conditions for rho
% Gaussian to Gaussian
eg = 'gaussian1d';
meanvalue0 = 1/3; sigma0 = 0.1;
meanvalue1 = 2/3; sigma1 = 0.1;
Normal = @(x,meanvalue,sigma) 1/(sqrt(2*pi)*sigma)*exp(-0.5*((x-meanvalue)/sigma).^2);
rho0 = Normal(xx,meanvalue0,sigma0);
rho1 = Normal(xx,meanvalue1,sigma1);

% Bimodal to Gaussian
% meanvalue1 = 0.5; sigma1 = 0.1;
% meanvalue0a = 0.1; meanvalue0b = 0.9;
% sigma0 = 0.05;
% Normal = @(x,meanvalue,sigma) 1/(sqrt(2*pi)*sigma)*exp(-0.5*((x-meanvalue)/sigma).^2);
% rho0 = Normal(x,meanvalue0a,sigma0) + Normal(x,meanvalue0b,sigma0) + 0.1;
% rho1 = Normal(x,meanvalue1,sigma1)+0.1;

opts.tol = 1e-10;
opts.nt = nt;
opts.nx = nx;
opts.maxIter = maxIter;
opts.sub_maxit = 5;
opts.gamma = gamma;
opts.tau = tau;
% opts.rho_analytical = rho_analytical;
% opts.mx_analytical = mx_analytical;

[rho_admm, mx_admm, outs_admm] = ot1d_admm(rho0,rho1,opts);
