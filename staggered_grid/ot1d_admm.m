function [rho,m,outs] = ot1d_admm(rho0,rho1,opts)
    
    % Grid Parameters
    nt = opts.nt;
    ntm = nt-1;
    nx = length(rho0);
    nxm = nx-1;
    dx = 1/nx;
    dt = 1/nt;

    % More parameters
    gamma = opts.gamma; % ADMM Time Step
    tol = opts.tol;
    tau = opts.tau; % DR timestep (inner loop)
    maxIter = opts.maxIter;
    sub_maxit = opts.sub_maxit;
    residual_diff = zeros(maxIter,1);
    residual_analytical  = zeros(maxIter,1);

    % BC
    dirichlet_bc_space  = zeros(nt,1);

    rho_new = ones(ntm,nx); rho = ones(ntm,nx);
    mx_new  = ones(nt,nxm); mx  = ones(nt,nxm);
    rho_tilde = interp_t_at_phi(rho,rho0,rho1);
    mx_tilde = interp_x_at_phi(rho,dirichlet_bc_space,dirichlet_bc_space);
    delta_rho = zeros(nt,nx); delta_mx = zeros(nt,nx);

    % Eigenvalues for the Laplacian
    lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nx-1)))/dx/dx;
    lambda_t = (2*ones(nt,1) - 2*cos(pi*dt.*(0:nt-1)'))/dt/dt;
    lambda_lap = lambda_x + lambda_t;

    for iter = 1:maxIter

        

    
    end



    %% Helper Functions
    function out = interp_t_at_phi(in,bc_start,bc_end)
        c = [1 1 zeros(1,ntm-1)];
        r = [1 zeros(1,ntm-1)];
        It = toeplitz(c,r);
        It = 0.5*It;
        out = It*in;

        % Boundary Conditions
        out(1,:) = out(1,:) + 0.5*bc_start;
        out(end,:) = out(end,:) + 0.5*bc_end;
    end

    function out = interp_t_at_rho(in)
        c = [1 zeros(1,ntm-1)];
        r = [1 1 zeros(1,ntm-1)];
        It = toeplitz(c,r);
        It = 0.5*It;
        out = It*in;
    end

    function out = interp_x_at_phi(in,bc_in,bc_out)
        c = [1 1 zeros(1,nxm-1)];
        r = [1 zeros(1,nxm-1)];
        Ix = toeplitz(c,r);
        Ix = 0.5*Ix;
        out = in*Ix';

        % Boundary Conditions
        out(:,1) = out(:,1) + 0.5*bc_in;
        out(:,end) = out(:,end) + 0.5*bc_out;
    end

    function out = interp_x_at_m(in)
        c = [1 zeros(1,nxm-1)];
        r = [1 1 zeros(1,nxm-1)];
        Ix = toeplitz(c,r);
        Ix = 0.5*Ix;
        out = in*Ix';

        % Boundary Conditions
        out(:,1) = out(:,1);
        out(:,end) = out(:,end);
    end

    function out = deriv_t_at_phi(in,bc_start,bc_end)
        c = [1 -1 zeros(1,ntm-1)];
        r = [1 zeros(1,ntm-1)];
        Dt = toeplitz(c,r);
        Dt = Dt./dt;
        out = Dt*in;

        % Boundary Conditions
        out(1,:) = out(1,:) - bc_start./dt;
        out(end,:) = out(end,:) + bc_end./dt;
    end

    function out = deriv_t_at_rho(in)
        c = [-1 zeros(1,ntm-1)];
        r = [-1 1 zeros(1,ntm-1)];
        Dt = toeplitz(c,r);
        Dt = Dt./dt;
        out = Dt*in;
    end

    function out = deriv_x_at_phi(in,bc_in,bc_out)
        c = [1 -1 zeros(1,nxm-1)];
        r = [1 zeros(1,nxm-1)];
        Dx = toeplitz(c,r);
        Dx = Dx./dx;
        out = in*Dx';

        % Boundary Conditions
        out(:,1) = out(:,1) - bc_in./dx;
        out(:,end) = out(:,end) + bc_out./dx;
    end

    function out = deriv_x_at_m(in)
        c = [-1 zeros(1,nxm-1)];
        r = [-1 1 zeros(1,nxm-1)];
        Dx = toeplitz(c,r);
        Dx = Dx./dx;
        out = in*Dx';
    end

    % Calculating the derivative of the objective
    % f = \int \int ||m||^2/rho dt dx
    function [df_drho, df_dm] = calc_gradf(rho_in,m_in)

        

        dirichlet_bc  = zeros(nt,1);
        rho_at_phi = interp_t_at_phi(rho_in,rho0,rho1);
        m_at_phi   = interp_x_at_phi(m_in,dirichlet_bc,dirichlet_bc);

        % Should this be the cutoff?
        ind = rho_at_phi > 1e-8;

        df_dm = zeros(size(rho_at_phi));
        df_drho = zeros(size(rho_at_phi));

        df_dm(ind)   = m_at_phi(ind)./rho_at_phi(ind);
        df_drho(ind) = -0.5*(df_dm(ind)).^2;  

        df_drho = interp_t_at_rho(df_drho);
        df_dm   = interp_x_at_m(df_dm);
    end

    % Calculate the objective function
    function obj = calc_objective(rho_in,m_in)
        dirichlet_bc  = zeros(nt,1);
        rho_at_phi = interp_t_at_phi(rho_in,rho0,rho1);
        m_at_phi   = interp_x_at_phi(m_in,dirichlet_bc,dirichlet_bc);

        % Should this be the cutoff?
        ind = rho_at_phi > 1e-8;

        obj = zeros(size(rho_at_phi));
        obj = zeros(size(rho_at_phi));

        obj(ind) = (m_at_phi(ind).^2)./rho_at_phi(ind);
        obj = sum(sum(obj))*dt*dx;
    end

    % Invert the negative Laplacian with Homogeneous Neumann BC's
    % (-\Delta \phi = f)
    function phi = invert_neg_laplacian(f)
        phi_hat = mirt_dctn(f);
        phi_hat = phi_hat./lambda_lap;
        phi_hat(1,1) = 0.0;  % Setting 0-frequency to zero
        phi = mirt_idctn(phi_hat);
    end

    % Project to the divergence free set
    % mu is at the rho-location, psi is at the m-location
    % Uses the fact that the boundary conditions in time are  
    % mu(0,x) = rho0 and mu(1,x) = rho1
    % and psi \cdot n = 0 for x on the boundary
    function [rho_out, m_out] = proj_div_free(mu, psi)
        dirichlet_bc  = zeros(nt,1);
        dmu_dt  = deriv_t_at_phi(mu,rho0,rho1);
        dpsi_dx = deriv_x_at_phi(psi,dirichlet_bc,dirichlet_bc);

        phi_temp = invert_neg_laplacian(dmu_dt + dpsi_dx);
        rho_out = mu + deriv_t_at_rho(phi_temp);
        m_out   = psi + deriv_x_at_m(phi_temp);
        
    end
    


    
end