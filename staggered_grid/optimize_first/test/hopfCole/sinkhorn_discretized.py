import numpy as np
from scipy.fft import fft, ifft, fftfreq
from pdb import set_trace as keyboard

def heat_kernel_operator(u, epsilon, dt, L=1.0):
    """
    Apply the 1D heat semigroup to vector u using FFT (periodic BCs)
    """
    N = len(u)
    k = fftfreq(N, d=L/N) * 2 * np.pi  # angular frequencies
    u_hat = fft(u)
    u_hat = u_hat * np.exp(-0.5 * epsilon * dt * k**2)
    return np.real(ifft(u_hat))

def sinkhorn_time_discrete(rho0, rho1, epsilon, N_steps=100, tol=1e-6, max_iter=500):
    """
    Dynamic Schrödinger Bridge via iterative scaling (Method A)
    """
    # Initialize potentials
    N = len(rho0)
    phi0 = np.ones(N)
    hat_phiN = np.ones(N)
    
    dt = 1.0 / N_steps
    
    for it in range(max_iter):
        # Forward propagation
        phi0_new = rho0 / heat_kernel_operator(hat_phiN, epsilon, dt*N_steps)
        # Backward propagation
        hat_phiN_new = rho1 / heat_kernel_operator(phi0_new, epsilon, dt*N_steps)
        
        err = np.linalg.norm(phi0_new - phi0) + np.linalg.norm(hat_phiN_new - hat_phiN)
        phi0, hat_phiN = phi0_new, hat_phiN_new
        
        if err < tol:
            print(f"Converged in {it+1} iterations, error={err:.2e}")
            break
    
    # Compute full trajectory
    phi_traj = np.zeros((N_steps+1, N))
    hat_phi_traj = np.zeros((N_steps+1, N))
    for k in range(N_steps+1):
        phi_traj[k] = heat_kernel_operator(phi0, epsilon, dt*k)
        hat_phi_traj[k] = heat_kernel_operator(hat_phiN, epsilon, dt*(N_steps - k))
    
    rho_traj = phi_traj * hat_phi_traj
    return rho_traj, phi_traj, hat_phi_traj

# Example usage
N_grid = 128
x = np.linspace(0,1,N_grid, endpoint=False)
rho0 = np.exp(-100*(x-0.3)**2)
rho0 /= rho0.sum()
rho1 = np.exp(-100*(x-0.7)**2)
rho1 /= rho1.sum()

epsilon = 0.01
rho_traj, phi_traj, hat_phi_traj = sinkhorn_time_discrete(rho0, rho1, epsilon)



