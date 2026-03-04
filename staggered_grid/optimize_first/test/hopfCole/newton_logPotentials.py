def newton_log_potentials(rho0, rho1, epsilon, N_steps=100, tol=1e-8, max_iter=50):
    """
    Newton solver on log-potentials u=log(phi0), w=log(hat_phiN) (Method B)
    """
    N = len(rho0)
    dt = 1.0 / N_steps
    
    # Initialize logs
    u = np.zeros(N)
    w = np.zeros(N)
    
    for it in range(max_iter):
        # Forward and backward
        phi0 = np.exp(u)
        hat_phiN = np.exp(w)
        P_phi0 = heat_kernel_operator(phi0, epsilon, dt*N_steps)
        P_hat_phiN = heat_kernel_operator(hat_phiN, epsilon, dt*N_steps)
        
        # Residuals
        F_u = phi0 * P_hat_phiN - rho0
        F_w = P_phi0 * hat_phiN - rho1
        
        # Simple Newton step (diagonal approximation)
        du = -F_u / (P_hat_phiN * phi0 + 1e-12)
        dw = -F_w / (P_phi0 * hat_phiN + 1e-12)
        
        u += du
        w += dw
        
        err = np.linalg.norm(F_u) + np.linalg.norm(F_w)
        if err < tol:
            print(f"Newton converged in {it+1} iterations, error={err:.2e}")
            break
    
    # Compute full trajectory
    phi_traj = np.zeros((N_steps+1, N))
    hat_phi_traj = np.zeros((N_steps+1, N))
    for k in range(N_steps+1):
        phi_traj[k] = heat_kernel_operator(np.exp(u), epsilon, dt*k)
        hat_phi_traj[k] = heat_kernel_operator(np.exp(w), epsilon, dt*(N_steps - k))
    
    rho_traj = phi_traj * hat_phi_traj
    return rho_traj, phi_traj, hat_phi_traj

# Example usage
rho_traj2, phi_traj2, hat_phi_traj2 = newton_log_potentials(rho0, rho1, epsilon)