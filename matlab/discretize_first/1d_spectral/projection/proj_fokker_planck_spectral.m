function x_out = proj_fokker_planck_spectral(x_in, problem, cfg)
% PROJ_FOKKER_PLANCK_SPECTRAL  Project onto the discretised FP constraint.
%
%   x_out = proj_fokker_planck_spectral(x_in, problem, cfg)
%
%   Variable layout (interior grid):
%     x_in.rho  (nt-1 x nx)   interior density at t_1,...,t_{nt-1}
%     x_in.mx   ( nt  x nx)   momentum (LEFT edges FE/IMEX; RIGHT edges BE)
%
%   After FFT in x, each mode k is projected independently:
%     u_k_out = u_in - Lp_k * (L_k * u_in - f_k)
%   where  u_k = [rho_hat_k(1:nt-1); mx_hat_k(1:nt)]  (size 2*nt-1)
%   and the RHS f_k encodes the BC densities:
%     f_k(1)  = f_scale_0(k) * rho0_hat(k)
%     f_k(nt) += f_scale_T(k) * rho1_hat(k)   ("+=" handles nt=1 correctly)

    nt = problem.nt;
    nx = problem.nx;
    sp = problem.spectral_proj;

    rho_hat = fft(x_in.rho, [], 2);   % (nt-1 x nx)
    mx_hat  = fft(x_in.mx,  [], 2);   % (nt   x nx)

    rho0_hat = fft(problem.rho0);      % (1 x nx)
    rho1_hat = fft(problem.rho1);

    rho_out_hat = zeros(nt-1, nx);
    mx_out_hat  = zeros(nt,   nx);

    for k = 1 : nx
        % Variable vector: [rho_hat(1:nt-1); mx_hat(1:nt)]
        u_in_k = [rho_hat(:, k); mx_hat(:, k)];   % (2*nt-1 x 1)

        % RHS: BC contributions
        f_k     = zeros(nt, 1);
        f_k(1)  = sp.f_scale_0(k) * rho0_hat(k);
        f_k(nt) = f_k(nt) + sp.f_scale_T(k) * rho1_hat(k);

        % Orthogonal projection onto nullspace complement of L_k
        residual_k = sp.L{k} * u_in_k - f_k;
        u_out_k    = u_in_k - sp.Lp{k} * residual_k;

        rho_out_hat(:, k) = u_out_k(1:nt-1);
        mx_out_hat(:,  k) = u_out_k(nt:end);
    end

    x_out.rho = real(ifft(rho_out_hat, [], 2));
    x_out.mx  = real(ifft(mx_out_hat,  [], 2));
end
