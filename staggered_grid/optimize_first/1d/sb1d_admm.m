function [rho,mx,outs] = sb1d_admm(rho0,rho1,opts)
    
    % Grid Parameters
    nt = opts.nt;
    ntm = nt-1;
    nx = length(rho0);
    nxm = nx-1;
    dx = 1/nx;
    dt = 1/nt;

    % More parameters
    gamma = opts.gamma; % ADMM Time Step
    % tau = opts.tau;   % Second Time Step for PDHG
    % tol = opts.tol;
    maxIter = opts.maxIter;
    vareps  = opts.vareps;
    residual_diff = zeros(maxIter,1);
    residual_analytical  = zeros(maxIter,1);


    % Check if reference solution is provided for true error computation
    compute_true_error = false;
    if isfield(opts, 'rho_star') && isfield(opts, 'mx_star')
        compute_true_error = true;
        rho_star = opts.rho_star;
        mx_star = opts.mx_star;
        true_error = zeros(maxIter,1);
    end

    % BC
    zeros_x  = zeros(nt,1);
    zeros_t  = zeros(1,nx);
    
    t = linspace(0,1,nt+1)';
    tt = t(2:end-1);
    % Initial Guess (Linear Interpolation btwn rho0 & rho1)
    % Primal Vars
    rho = (1-tt).*rho0 + tt.*rho1; rho_new = rho;
    mx  = zeros(nt,nxm);  mx_new = mx;

    % Dual Vars
    rho_tilde = rho; rho_tilde_new = rho;
    bx = mx; bx_new = mx;
    
    % Extra Gradient Step
    delta_rho = zeros(size(rho)); delta_mx = zeros(size(mx));

    % Eigenvalues for the Laplacian
    lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
    lambda_t = (2*ones(nt,1) - 2*cos(pi*dt.*(0:ntm)'))/dt/dt;
    % lambda_lap = lambda_x + lambda_t;

    lambda_lap_biharmonic = lambda_x + lambda_t - (vareps^2).*lambda_x.^2;

    % Better guess for velocity
    % potential = ones(nt,1)*(rho1 - rho0);
    % potential = mirt_dctn(potential);
    % potential = potential./lambda_x;
    % potential(1) = 0.0;
    % potential = mirt_idctn(potential);
    % potential = interp_x_at_m(potential);
    % mx = deriv_x_at_phi(potential,dirichlet_bc_space,dirichlet_bc_space);
    % mx = rho.*mx;
    

    for iter = 1:maxIter


        % Proximal Step for the Kinetic Energy Term
        % Version 1: Interpolate to cell centers. Compute prox,
        % then interpolate back

        tmp_rho = rho_tilde + delta_rho./gamma;
        tmp_mx  = bx  + delta_mx./gamma;

        % Interpolate to cell-centers
        tmp_rho = interp_t_at_phi(tmp_rho,rho0,rho1);
        tmp_mx  = interp_x_at_phi(tmp_mx,zeros_x,zeros_x);

        % Proximal Operator on Cell-Centers
        sigma = 1/gamma; % Just to make code cleaner
        rho_new = solve_cubic(1,2*sigma-tmp_rho,sigma^2-2*sigma*tmp_rho,...
                                -sigma*(sigma*tmp_rho + 0.5.*tmp_mx.^2));

        mx_new = rho_new.*tmp_mx./(rho_new + sigma);

        % Truncate negative rho's
        neg_ind = (rho_new <= 1e-12);
        rho_new(neg_ind) = 0.0;
        mx_new(neg_ind) = 0.0;

        % Now interpolate back to rho and m locations
        rho_new = interp_t_at_rho(rho_new);
        mx_new  = interp_x_at_m(mx_new);
        
        % Projection to divergence free set

        % (rho,m) - delta/gamma
        tmp_rho = rho_new - delta_rho./gamma;
        tmp_mx  = mx_new  - delta_mx./gamma;

        [rho_tilde_new, bx_new] = proj_div_biharmonic(tmp_rho, tmp_mx);

        % Extra-Gradient step
        delta_rho_new = delta_rho - gamma*(rho_new - rho_tilde_new);
        delta_mx_new  = delta_mx  - gamma*(mx_new - bx_new);
        
        % Running Error
        drho = (rho_tilde_new - rho_tilde).^2;
        dmx  = (bx_new - bx).^2;

        residual_diff(iter) = sqrt(dt*dx*(sum(drho(:)) + sum(dmx(:))));

        % Compute true error if reference solution is provided
        if compute_true_error
            true_error(iter) = sqrt(sum((rho(:) - rho_star(:)).^2) + sum((mx(:) - mx_star(:)).^2)) * sqrt(dt*dx);
        end

         % Update
        rho_tilde = rho_tilde_new;
        bx  = bx_new;
        rho = rho_new; mx = mx_new;
        delta_rho = delta_rho_new;
        delta_mx  = delta_mx_new;

    end

    outs.residual_diff = residual_diff;
    if compute_true_error
        outs.true_error = true_error;
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
    % function phi = invert_neg_laplacian(f)
    %     phi_hat = mirt_dctn(f);
    %     phi_hat = phi_hat./lambda_lap;
    %     phi_hat(1,1) = 0.0;  % Setting 0-frequency to zero
    %     phi = mirt_idctn(phi_hat);
    % end

    % Invert the negative laplacian minus eps^2*biharmonic
    function phi = invert_neg_laplacian_biharmonic(f)

        phi_hat = mirt_dctn(f);
        phi_hat = phi_hat./(lambda_lap_biharmonic);
        phi_hat(1,1) = 0.0;
        phi = mirt_idctn(phi_hat);

    end

    % Project to the focker-planck equation
    % mu is at the rho-location, psi is at the m-location
    % Uses the fact that the boundary conditions in time are  
    % mu(0,x) = rho0 and mu(1,x) = rho1
    % and psi \cdot n = 0 for x on the boundary
    function [rho_out, m_out] = proj_div_biharmonic(mu, psi)

        zeros_xx = zeros_x(2:end);

        dirichlet_bc  = zeros(nt,1);
        dmu_dt  = deriv_t_at_phi(mu,rho0,rho1);
        nablax_mu = deriv_x_at_m(mu);
        nablax_mu = deriv_x_at_phi(nablax_mu,zeros_xx,zeros_xx);

        nablax_rho0 = deriv_x_at_m(rho0);
        nablax_rho0 = deriv_x_at_phi(nablax_rho0,0,0);
        nablax_rho1 = deriv_x_at_m(rho1);
        nablax_rho1 = deriv_x_at_phi(nablax_rho1,0,0);

        % Interpolate back to cell-center
        nablax_mu  = interp_t_at_phi(nablax_mu,nablax_rho0,nablax_rho1);

        dpsi_dx = deriv_x_at_phi(psi,dirichlet_bc,dirichlet_bc);

        phi_temp = invert_neg_laplacian_biharmonic(-dmu_dt - dpsi_dx + vareps.*nablax_mu);
        
        dphi_dx = deriv_x_at_m(phi_temp);
        nablax_phi = deriv_x_at_phi(dphi_dx,zeros_x,zeros_x);
        nablax_phi = interp_t_at_rho(nablax_phi);
        rho_out = mu - deriv_t_at_rho(phi_temp) - vareps.*nablax_phi;
        m_out   = psi - dphi_dx; 
        % max(max(abs(deriv_t_at_phi(rho_out,rho0,rho1) + deriv_x_at_phi(m_out,dirichlet_bc,dirichlet_bc))))

    end
    


    
end