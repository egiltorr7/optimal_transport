function [rho,mx,outs] = ot1d_admm(rho0,rho1,opts)
    
    % Grid Parameters
    nt = opts.nt;
    ntm = nt-1;
    nx = length(rho0);
    nxm = nx-1;
    dx = 1/nx;
    dt = 1/nt;

    % More parameters
    gamma = opts.gamma; % ADMM Time Step
    tau = opts.tau;     % Second Time Step for PDHG
    % tol = opts.tol;
    maxIter = opts.maxIter;
    residual_primal_diff = zeros(maxIter,1);
    residual_aux_diff = zeros(maxIter,1);
    residual_dual_diff = zeros(maxIter,1);
    % residual_analytical  = zeros(maxIter,1);

    % BC
    zeros_x  = zeros(nt,1);
    zeros_t  = zeros(1,nx);
    
    t = linspace(0,1,nt+1)';
    tt = (t(2:end)+t(1:end-1))/2;
    % Initial Guess (Linear Interpolation btwn rho0 & rho1)
    % Primal Vars
    rho = (1-tt).*rho0 + tt.*rho1; rho_new = zeros(size(rho));
    mx  = zeros(nt,nx); mx_new = zeros(size(mx));

    % Dual Vars
    rho_tilde = interp_t_at_rho(rho); rho_tilde_new = zeros(size(rho_tilde));
    bx = interp_x_at_m(mx); bx_new = zeros(size(bx));

    rho = -1.*interp_t_at_phi(rho_tilde,zeros_t,zeros_t);
    mx  = -1.*interp_x_at_phi(bx,zeros_x,zeros_x);
    
    % Extra Gradient Step
    delta_rho = zeros(size(rho_tilde)); delta_mx = zeros(size(bx));
    delta_rho_new = delta_rho; delta_mx_new = delta_mx;

    % Eigenvalues for the Laplacian
    lambda_x = (2*ones(1,nx) - 2*cos(pi*dx.*(0:nxm)))/dx/dx;
    lambda_t = (2*ones(nt,1) - 2*cos(pi*dt.*(0:ntm)'))/dt/dt;
    lambda_lap = lambda_x + lambda_t;

    % Better guess for velocity
    % potential = ones(nt,1)*(rho1 - rho0);
    % potential = mirt_dctn(potential);
    % potential = potential./lambda_x;
    % potential(1) = 0.0;
    % potential = mirt_idctn(potential);
    % potential = interp_x_at_m(potential);
    % mx = deriv_x_at_phi(potential,zeros_x,zeros_x);
    % mx = rho.*mx;
    

    for iter = 1:maxIter
        
        % Set up for first Proximal Operator
           
        % A(rho,m) + (rho_tilde,b) - \delta/gamma
        tmp_rho = interp_t_at_rho(rho) + rho_tilde - delta_rho./gamma;
        tmp_mx  = interp_x_at_m(mx) + bx - delta_mx./gamma;
        
        % (rho,m) - (gamma/tau)*A^T*[(tmp_rho,tmp,mx)]
        tmp_rho = rho - (gamma/tau)*interp_t_at_phi(tmp_rho,zeros_t,zeros_t);
        tmp_mx  = mx - (gamma/tau)*interp_x_at_phi(tmp_mx,zeros_x,zeros_x);

        % (rho,m) - d/tau - (gamma/tau)*A^T*[(tmp_rho,tmp,mx)]
        % where d are the temporal BC's for the interpolation in time
        tmp_rho(1,:)   = tmp_rho(1,:) + rho0./(2*tau);
        tmp_rho(end,:) = tmp_rho(end,:) + rho1./(2*tau);

        % Prox of J^*
        [rho_new, mx_new] = project_parabolic(tmp_rho,tmp_mx);

        % % Set-up for Moreau
        % tmp_rho = tau.*tmp_rho;
        % tmp_mx  = tau.*tmp_mx;
        % 
        % % Proximal Step for the Kinetic Energy Term
        % % sigma = 1/tau; % Just to make code cleaner
        % rho_new = solve_cubic(1,2*tau-tmp_rho,tau^2-2*tau*tmp_rho,...
        %                         -tau*(tau*tmp_rho + 0.5.*tmp_mx.^2));
        % mx_new  = (rho_new.*tmp_mx)./(tau+rho_new); 
        % 
        % % Truncate negative rho's
        % neg_ind = (rho_new <= 1e-12);
        % rho_new(neg_ind) = 0.0;
        % mx_new(neg_ind) = 0.0;
        % 
        % % Now use Moreau Decomposition
        % rho_new = (tmp_rho - rho_new)./tau;
        % mx_new  = (tmp_mx-mx_new)./tau;
        

        % Set up the second Proximal operator
        tmp_rho = delta_rho./gamma - interp_t_at_rho(rho_new);
        tmp_mx  = delta_mx./gamma  - interp_x_at_m(mx_new);

        [rho_tilde_new, bx_new] = proj_div_free(gamma.*tmp_rho, gamma.*tmp_mx);
        
        % Use Moreau
        rho_tilde_new = tmp_rho - rho_tilde_new./gamma;
        bx_new        = tmp_mx - bx_new./gamma;

        % Extra-Gradient step
        delta_rho_new = delta_rho - gamma*(interp_t_at_rho(rho_new) + rho_tilde_new);
        delta_mx_new  = delta_mx  - gamma*(interp_x_at_m(mx_new) + bx_new);
        
        % Running Error
        drho = (rho_new - rho).^2;
        dmx  = (mx_new - mx).^2;
        drho_tilde = (rho_tilde_new - rho_tilde).^2;
        dbx  = (bx_new - bx).^2;
        ddelta_rho = (delta_rho_new - delta_rho).^2;
        ddelta_mx  = (delta_mx_new - delta_mx).^2;

        residual_primal_diff(iter) = sqrt(dt*dx*(sum(drho(:)) + sum(dmx(:))));
        residual_aux_diff(iter)    = sqrt(dt*dx*(sum(drho_tilde(:)) + sum(dbx(:))));
        residual_dual_diff(iter)   = sqrt(dt*dx*(sum(ddelta_rho(:)) + sum(ddelta_mx(:))));

         % Update
        rho = rho_new; rho_tilde = rho_tilde_new; delta_rho = delta_rho_new;
        mx  = mx_new;  bx = bx_new; delta_mx = delta_mx_new;
    end
    
    % DOUBLE CHECK: This should be the primal vars
    rho = delta_rho;
    mx = delta_mx;

    outs.residual_primal_diff = residual_primal_diff;
    outs.residual_aux_diff = residual_aux_diff;
    outs.residual_dual_diff = residual_dual_diff;
    % outs.rho_tilde = rho_tilde;

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
        % max(max(abs(deriv_t_at_phi(rho_out,rho0,rho1) + deriv_x_at_phi(m_out,dirichlet_bc,dirichlet_bc))))

    end
    


    
end