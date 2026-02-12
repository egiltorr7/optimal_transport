% ADMM on the primal problem of the dynamic formulation of OT
function [rho,mx,my,outs] = ot2d_admm_primal(rho0, rho1, opts)
    
    % Grid Parameters
    nt = opts.nt;
    ntp = nt+1;

    nx = opts.nx;
    nxp = nx+1;

    ny = opts.ny;
    nyp = ny+1;

    dx = 1/nxp;
    dy = 1/nyp;
    dt = 1/ntp;

    ex = ones(nxp,1);
    ey = ones(nyp,1);
    et = ones(ntp,1);

    % More parameters
    tol = opts.tol;
    maxIter = opts.maxIter;
    tau = opts.tau;
    sigma = 1/tau;
    residual_diff = zeros(maxIter,1);
    % residual_analytical  = zeros(maxIter,1);
    residual_numerical = zeros(maxIter,1);
    gpu = opts.GPUcomputing;

    % Initialize rho as linear interpolation in time from rho0 to rho1
    t_vec = reshape((0:nt-1)' * dt, nt, 1, 1);  % Shape (nt, 1, 1)
    rho = (1 - t_vec) .* reshape(rho0, 1, nxp, nyp) + t_vec .* reshape(rho1, 1, nxp, nyp);
    rho_new = rho;
    mx  = zeros(ntp,nx,nyp); mx_new  = zeros(ntp,nx,nyp);
    my  = zeros(ntp,nxp,ny); my_new  = zeros(ntp,nxp,ny);
    rho_tilde = rho; a = ones(nt,nxp,nyp);
    mx_tilde = zeros(ntp,nx,nyp);  my_tilde = zeros(ntp,nxp,ny);
    bx = zeros(ntp,nx,nyp); by = zeros(ntp,nxp,ny);
    bx_interior = ones(nt,nx,ny); by_interior = ones(nt,nx,ny);
    b_interior_sqrd = ones(nt,nx,ny); a_interior = ones(nt,nx,ny);

    phit = zeros(nt,nxp,nyp);
    phix = zeros(ntp,nx,nyp);
    phiy = zeros(ntp,nxp,ny);

    % Create Derivative Matrix 
    Dy = create_div_1d(ny);
    Dx = create_div_1d(nx);
    Dt = create_div_1d(nt);

    % Eigenvalues for laplacian
    [lambda_x, Vx] = Bmtx_eig(nx);
    % lambda_x = lambda_x';
    iVx = Vx';
    [lambda_t, Vt] = Bmtx_eig(nt);
    iVt = Vt';
    [lambda_y, Vy] = Bmtx_eig(ny);
    iVy = Vy';
    
    % Boundary Conditions
    bc0 = rho0./dt;
    bc1 = -rho1./dt;

    if (opts.GPUcomputing)
        ex = gpuArray(ex);
        ey = gpuArray(ey);
        et = gpuArray(et);
        residual_diff = gpuArray(residual_diff);
        residual_numerical = gpuArray(residual_numerical);
        rho = gpuArray(rho);
        rho_new = gpuArray(rho_new);
        mx  = gpuArray(mx); mx_new  = gpuArray(mx_new);
        my  = gpuArray(my); my_new  = gpuArray(my_new);
        rho_tilde = gpuArray(rho_tilde); mx_tilde = gpuArray(mx_tilde);
        my_tilde = gpuArray(my_tilde);
        phix = gpuArray(phix); phiy = gpuArray(phiy);
        phit = gpuArray(phit);
        Dx = gpuArray(Dx); Dy = gpuArray(Dy);
        Dt = gpuArray(Dt); lambda_x = gpuArray(lambda_x);
        Vx = gpuArray(Vx); lambda_y = gpuArray(lambda_y);
        Vy = gpuArray(Vy); lambda_t = gpuArray(lambda_t);
        Vt = gpuArray(Vt);
        bc0 = gpuArray(bc0); bc1 = gpuArray(bc1);
        bx = gpuArray(bx); by = gpuArray(by);
        bx_interior = gpuArray(bx_interior); by_interior = gpuArray(by_interior);
        b_interior_sqrd = gpuArray(b_interior_sqrd); a_interior = gpuArray(a_interior);
        Device = opts.Device;
    end
    
    if (opts.GPUcomputing)
        lambda_lap = gpuArray(squeeze(tensorprod(lambda_t, ex*ey'))+squeeze(tensorprod(et,lambda_x*ey'))+squeeze(tensorprod(et,ex*lambda_y')));
    else
        lambda_lap = squeeze(tensorprod(lambda_t, ex*ey'))+squeeze(tensorprod(et,lambda_x*ey'))+squeeze(tensorprod(et,ex*lambda_y'));
    end
    % lambda_lap(1,1,1) = 1;

    % Boundary Conditions
    % helper_t0 = zeros(ntp,nxp); helper_t0(1,1) = 1;
    % helper_t1 = zeros(ntp,nxp); helper_t1(end,end) = 1;

    % bc = [1 zeros(1,nt)]'*rho0./dt - [zeros(1,nt) 1]'*rho1./dt;
    

    tic
    for iter = 1:maxIter

        % Prox for quadratic cost
        a = rho_tilde + sigma*phit;
        a_interior = a(:,2:nxp,2:nyp);
        bx = mx_tilde + sigma*phix;
        by = my_tilde + sigma*phiy;
        bx_interior = bx(2:ntp,:,2:nyp);
        by_interior = by(2:ntp,2:nxp,:);
        b_interior_sqrd  = bx_interior.^2 + by_interior.^2;

        % First take care of the interior nodes
        rho_new(:,2:nxp,2:nyp) = solve_cubic(1,2*sigma - a_interior, sigma^2 - 2*a_interior*sigma, -sigma*(sigma*a_interior+(b_interior_sqrd./2)),gpu);
        mx_new(2:ntp,:,2:nyp)  = (rho_new(:,2:nxp,2:nyp).*bx_interior)./(rho_new(:,2:nxp,2:nyp)+sigma);
        my_new(2:ntp,2:nxp,:)  = (rho_new(:,2:nxp,2:nyp).*by_interior)./(rho_new(:,2:nxp,2:nyp)+sigma);

        % Now do the x=0,y=0 boundary
        rho_new(:,1,1) = a(:,1,1);

        % Now the t=0,y=0 boundary
        mx_new(1,:,1) = (rho0(2:nxp,1)'.*squeeze(bx(1,:,1)))./(rho0(2:nxp,1)'+sigma);
        
        % t=0,x=0 boundary
        my_new(1,1,:) = (rho0(1,2:nyp)'.*squeeze(by(1,1,:)))./(rho0(1,2:nyp)'+sigma);

        % t=0 boundary
        mx_new(1,:,2:nyp) = (rho0(2:nxp,2:nyp).*squeeze(bx(1,:,2:nyp)))./(rho0(2:nxp,2:nyp)+sigma);
        my_new(1,2:nxp,:) = (rho0(2:nxp,2:nyp).*squeeze(by(1,2:nxp,:)))./(rho0(2:nxp,2:nyp)+sigma);

        % x=0 boundary
        a_interior = a(:,1,2:nyp);
        b_interior_sqrd = by(2:ntp,1,:).^2;
        rho_new(:,1,2:nyp) = solve_cubic(1,2*sigma - a_interior, sigma^2 - 2*a_interior*sigma, -sigma*(sigma*a_interior+(b_interior_sqrd./2)),gpu);
        my_new(2:ntp,1,:) = (squeeze(rho_new(:,1,2:nyp)).*squeeze(by(2:ntp,1,:)))./(squeeze(rho_new(:,1,2:nyp))+sigma);

        % y=0 boundary 
        a_interior = a(:,2:nxp,1);
        b_interior_sqrd = bx(2:ntp,:,1).^2;
        rho_new(:,2:nxp,1) = solve_cubic(1,2*sigma - a_interior, sigma^2 - 2*a_interior*sigma, -sigma*(sigma*a_interior+(b_interior_sqrd./2)),gpu);
        mx_new(2:ntp,:,1) = (rho_new(:,2:nxp,1).*squeeze(bx(2:ntp,:,1)))./(rho_new(:,2:nxp,1)+sigma);

        % Need to figure out this indexing... it's very tedious
        neg_ind = (rho_new <= 1e-12);
        neg_ind_rho0 = (rho0 <= 1e-12);
        rho_new(neg_ind) = 0.0;
        mx_new(2:ntp,:,:) = mx_new(2:ntp,:,:) .* ~neg_ind(:,2:nxp,:);
        mx_new(1,:,:) = squeeze(mx_new(1,:,:)) .* ~neg_ind_rho0(2:nxp,:);

        my_new(2:ntp,:,:) = my_new(2:ntp,:,:) .* ~neg_ind(:,:,2:nyp);
        my_new(1,:,:) = squeeze(my_new(1,:,:)) .* ~neg_ind_rho0(:,2:nyp);

        % Think this^ solved the problem

        % Done with the proximal operator for quadratic cost

        % Projection to the set Au=b
        a = rho_new - sigma*phit;
        bx = mx_new - sigma*phix;
        by = my_new - sigma*phiy;
        [rho_tilde, mx_tilde, my_tilde] = proj_div_free(a,bx,by,bc0,bc1);

        % Last Step of ADMM
        phit = phit - tau*(rho_new-rho_tilde);
        phix = phix - tau*(mx_new-mx_tilde);
        phiy = phiy - tau*(my_new-my_tilde);
        
        % Calculate residual
        drho = rho_new - rho;
        dmx  = mx_new - mx;
        dmy  = my_new - my;

        drho_ref = rho_new - opts.rho_numerical;
        dmx_ref  = mx_new - opts.mx_numerical;
        dmy_ref  = my_new - opts.my_numerical;
        residual_diff(iter) = sqrt(dt*dx*dy*(norm(drho,'fro')^2 + norm(dmx,'fro')^2 + norm(dmy,'fro')^2));
        residual_numerical(iter) = sqrt(dt*dx*dy*(norm(drho_ref,'fro')^2 ...
            + norm(dmx_ref,'fro')^2 + norm(dmy_ref,'fro')^2));

        if residual_diff(iter) < 1e-14
            opts.maxIter = iter;
            break

        end
        
        % update
        rho = rho_new;
        mx  = mx_new;
        my  = my_new;

        % GPU Clearing memory
        if gpu && (mod(iter,50)==0) && (Device.AvailableMemory < 0.25*Device.TotalMemory)
            reset(Device)
        end
            

    end
    toc

    % fprintf('Final residual (difference) is: %.2e\n', residual_diff(end));
    % fprintf('Final residual (analytical) is: %.2e\n', residual_analytical(end));
    % fprintf('Final residual (numerical) is: %.2e\n', residual_numerical(end));
    

    % Data to be saved for output
    rho = gather(rho);
    mx = gather(mx);
    my = gather(my);

    outs.residual_diff = gather(residual_diff);
    % outs.residual_analytical = residual_analytical;
    outs.residual_numerical = gather(residual_numerical);

    %% Auxiliary Functions
    function D = create_div_1d(n)
        h = 1/(n+1);
        e=ones(n,1);
        D=spdiags([-e 1*e], [-1 0], n+1, n);
        D = full(D);
        D=D/h;
    end

    function [lambda,V] = Bmtx_eig(n)
        np = n+1;
        h = 1/np;
        xx = [0.5*h:h:1-h/2]';
        lambda=(2*ones(np,1)-2*cos([0:n]'*pi*h))/h^2;
        V=cos(xx*pi*[0:n]);
        V=V*diag(sqrt(1./diag(V'*V))); 
    end

    function [rho_out,mx_out,my_out] = proj_div_free(rho_in, mx_in, my_in, bc0, bc1)

        % tmp1 = -(Dt*rho_in + mx_in*Dx') + b_vec;
        % b - Au
        tmp1 =  squeeze(-tensorprod(Dt,rho_in,2,1)); % t
        tmp1 = tmp1 - pagetranspose(squeeze(tensorprod(Dx,mx_in,2,2)));   % x
        tmp1 = tmp1 - tensorprod(my_in,Dy',3,1);  % y
        tmp1(1,:,:) = squeeze(tmp1(1,:,:)) + bc0;
        tmp1(ntp,:,:) = squeeze(tmp1(ntp,:,:)) + bc1;

        % Inverse Transform
        tmp2 = squeeze(tensorprod(iVt,tmp1,2,1));  % t
        tmp2 = pagetranspose(squeeze(tensorprod(iVx,tmp2,2,2)));  % x
        tmp2 = squeeze(tensorprod(tmp2,iVy',3,1)); % y
        
        % Divide by laplacian
        tmp2 = tmp2./lambda_lap;
        tmp2(1,1,1) = 0.0;

        % Forward transform
        tmp2 = squeeze(tensorprod(Vt,tmp2,2,1));  % t
        tmp2 = pagetranspose(squeeze(tensorprod(Vx,tmp2,2,2)));  % x
        tmp2 = squeeze(tensorprod(tmp2,Vy',3,1)); % y
        
        rho_out = rho_in + squeeze(tensorprod(Dt,tmp2,1,1));
        mx_out  = mx_in  + pagetranspose(squeeze(tensorprod(Dx,tmp2,1,2)));
        my_out  = my_in  + squeeze(tensorprod(tmp2,Dy',3,2));

        % Error
        % err_vec = - tensorprod(Dt,rho_out,2,1) - ...
        %             pagetranspose(tensorprod(Dx,mx_out,2,2)) - tensorprod(my_out,Dy',3,1);
        % err_vec(1,:,:) = squeeze(err_vec(1,:,:)) + bc0;
        % err_vec(ntp,:,:) = squeeze(err_vec(ntp,:,:)) + bc1;
        % err = max(max(max(abs(err_vec))));
        % disp(err)
    end

    
end