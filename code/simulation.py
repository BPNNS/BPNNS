#!/usr/bin/env python3
"""
Bio-Physical Neural Network Simulation
======================================
Integrates the Navier–Stokes–Madhava–Leibniz–Golden Ratio coupled PDEs.
Output: biophysical_ai_results.pdf (3‑panel figure).

Author: Mohsen Mostafa
ORCID:  0009-0004-4478-0317
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===== Domain & Parameters =====
N = 128
L = 2 * np.pi
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Physical constants
nu = 0.01            # viscosity
D_sigma = 0.005      # synaptic diffusivity
lam = 0.2            # geometric forcing strength
gamma = 1.0          # growth rate
phi_golden = (1 + np.sqrt(5)) / 2   # Golden Ratio
eta = 0.05           # spectral nudging rate (gentle)

# Time
dt = 0.005
t_final = 50.0
n_steps = int(t_final / dt)
plot_interval = 500

# ===== Fourier Setup =====
kx = fftfreq(N, d=dx) * 2 * np.pi
ky = fftfreq(N, d=dx) * 2 * np.pi
Kx, Ky = np.meshgrid(kx, ky)
K2 = Kx**2 + Ky**2
K2[0, 0] = 1e-16          # avoid division by zero
inv_K2 = 1.0 / K2
inv_K2[0, 0] = 0.0

# Dealiasing mask (2/3 rule)
kmax = N // 3
mask = (np.abs(Kx) < kmax * 2 * np.pi / L) & (np.abs(Ky) < kmax * 2 * np.pi / L)
mask = mask.astype(float)

# Target vorticity amplitudes for odd wavenumbers (Madhava-Leibniz series)
k_mag = np.sqrt(K2)
odd_mask = np.abs(np.sin(np.pi * k_mag / 2)) > 0.5   # approximate odd integers
target_amp = np.zeros_like(K2, dtype=float)
for k in range(1, kmax + 1, 2):
    shell = (np.abs(k_mag - k) < 0.5) & mask.astype(bool)
    target_amp[shell] = 1.0 / k**2
target_amp *= mask

# Integrating factors for implicit diffusion
visc_factor = np.exp(-nu * K2 * dt)       # for Navier-Stokes
diff_factor = 1.0 / (1.0 + D_sigma * K2 * dt)  # for sigma (implicit)

# ===== Initial Conditions =====
# Random incompressible velocity with energy only in odd wavenumbers
u_hat = np.zeros((N, N), dtype=complex)
v_hat = np.zeros((N, N), dtype=complex)
for k in range(1, 10, 2):
    shell = (np.abs(k_mag - k) < 0.5) & mask.astype(bool)
    if np.any(shell):
        ampl = 0.05 * (np.random.randn(*u_hat[shell].shape) + 1j * np.random.randn(*u_hat[shell].shape))
        u_hat[shell] = ampl * (-Ky[shell] / (K2[shell] + 1e-16))
        v_hat[shell] = ampl * ( Kx[shell] / (K2[shell] + 1e-16))

u = np.real(ifft2(u_hat))
v = np.real(ifft2(v_hat))
sigma = 0.1 + 0.05 * np.random.randn(N, N)   # synaptic density

# ===== Helper Functions =====
def pressure_projection(u_hat, v_hat):
    """Enforce incompressibility on velocity field in Fourier space."""
    div = 1j * (Kx * u_hat + Ky * v_hat)
    u_hat -= 1j * Kx * (div * inv_K2) * mask
    v_hat -= 1j * Ky * (div * inv_K2) * mask
    return u_hat, v_hat

def geometric_gauge(u_hat):
    """Compute Madhava-Leibniz gauge potential Phi and its gradient."""
    omega_hat = 1j * (Kx * v_hat - Ky * u_hat)
    phi_hat = omega_hat * inv_K2  # ΔΦ = -ω → Φ = invLaplacian(ω)
    phi_hat[0, 0] = 0.0
    Phi = np.real(ifft2(phi_hat))
    grad_Phi_x = np.real(ifft2(1j * Kx * phi_hat))
    grad_Phi_y = np.real(ifft2(1j * Ky * phi_hat))
    return Phi, grad_Phi_x, grad_Phi_y

def spectral_nudge(u_hat, v_hat):
    """Softly adjust vorticity spectrum towards Madhava-Leibniz target."""
    omega_hat = 1j * (Kx * v_hat - Ky * u_hat)
    # Compute magnitude and preserve phase
    curr_amp = np.abs(omega_hat)
    phase = np.angle(omega_hat)
    # Mix current amplitude with target amplitude
    new_amp = (1 - eta) * curr_amp + eta * np.sqrt(target_amp) * np.sqrt(K2)  # convert target to amplitude
    omega_hat = new_amp * np.exp(1j * phase)
    # Reconstruct velocity from nudged vorticity
    u_hat = -1j * Ky * omega_hat * inv_K2
    v_hat =  1j * Kx * omega_hat * inv_K2
    return u_hat, v_hat

# ===== Main Time Loop =====
mean_sigma_list = []
time_list = []

for step in range(n_steps):
    t = step * dt

    # ------- Velocity (Navier–Stokes) -------
    u_hat = fft2(u)
    v_hat = fft2(v)

    # Nonlinear advection (dealiased)
    du_nl = fft2(u * ifft2(1j * Kx * u_hat) + v * ifft2(1j * Ky * u_hat)) * mask
    dv_nl = fft2(u * ifft2(1j * Kx * v_hat) + v * ifft2(1j * Ky * v_hat)) * mask

    # Geometric force from gauge field
    _, grad_Phi_x, grad_Phi_y = geometric_gauge(u_hat)
    fx_geo = lam * sigma * grad_Phi_x
    fy_geo = lam * sigma * grad_Phi_y

    # Combine explicit terms
    rhs_u = -du_nl + fft2(fx_geo)
    rhs_v = -dv_nl + fft2(fy_geo)

    # Semi-implicit step: exponential integration for viscosity
    u_hat = visc_factor * u_hat + dt * rhs_u
    v_hat = visc_factor * v_hat + dt * rhs_v

    # Pressure projection
    u_hat, v_hat = pressure_projection(u_hat, v_hat)

    # Spectral nudging towards Madhava-Leibniz series
    u_hat, v_hat = spectral_nudge(u_hat, v_hat)

    # Back to physical space
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))

    # ------- Synaptic Density (σ) -------
    sigma_hat = fft2(sigma)
    adv_sigma = fft2(u * ifft2(1j * Kx * sigma_hat) + v * ifft2(1j * Ky * sigma_hat)) * mask
    react = sigma * (gamma - phi_golden * sigma)

    # Implicit diffusion + explicit advection/reaction
    sigma_hat = diff_factor * (sigma_hat + dt * (-adv_sigma + fft2(react)))
    sigma = np.real(ifft2(sigma_hat))
    sigma = np.clip(sigma, 0, None)

    # ------- Diagnostics -------
    if step % plot_interval == 0:
        mean_sigma = np.mean(sigma)
        mean_sigma_list.append(mean_sigma)
        time_list.append(t)
        print(f"t={t:.2f}, <σ>={mean_sigma:.4f} (theory={gamma/phi_golden:.4f})")

# ===== Final Analysis & Figures =====
u_hat = fft2(u)
v_hat = fft2(v)
omega_hat = 1j * (Kx * v_hat - Ky * u_hat)
energy_2d = 0.5 * np.abs(omega_hat)**2

# Radial spectrum
k_vals = np.arange(0, N//2 + 1)
spec = np.array([np.mean(energy_2d[(np.abs(k_mag - k) < 0.5) & (k_mag > 0)]) for k in k_vals])

# Sigma homeostasis
theory_sigma = gamma / phi_golden

# Fibonacci angle detection
sigma_smooth = gaussian_filter(sigma, sigma=1.0)
gx, gy = np.gradient(sigma_smooth)
grad_mag = np.sqrt(gx**2 + gy**2)
angle = np.arctan2(gy, gx)
high_mag = grad_mag > np.percentile(grad_mag, 80)
angles_high = angle[high_mag]
# Compute pairwise differences (Fibonacci peak expected at π/5)
angle_diffs = []
for i in range(len(angles_high)):
    diffs = np.abs(angles_high[i] - angles_high)
    angle_diffs.extend(diffs[(diffs > 0) & (diffs < np.pi)])
angle_diffs = np.array(angle_diffs)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].loglog(k_vals[1:], spec[1:], 'b-', label='Simulated')
odd_k = np.arange(1, min(20, N//2), 2)
axes[0].loglog(odd_k, 0.01 / odd_k**2, 'r--', label=r'$\propto 1/k^2$ (odd)')
axes[0].set_xlabel('k'); axes[0].set_ylabel('E_ω(k)')
axes[0].legend(); axes[0].set_title('Vorticity Spectrum Locking')

axes[1].plot(time_list, mean_sigma_list, 'k-')
axes[1].axhline(theory_sigma, color='red', linestyle='--', label=f'γ/ϕ = {theory_sigma:.4f}')
axes[1].set_xlabel('Time'); axes[1].set_ylabel('<σ>')
axes[1].legend(); axes[1].set_title('Golden Ratio Homeostasis')

axes[2].hist(angle_diffs, bins=60, density=True, color='green', alpha=0.7)
axes[2].axvline(np.pi/5, color='orange', linestyle='--', label='π/5 (36°, Fibonacci)')
axes[2].set_xlabel('Angle difference'); axes[2].set_title('Branching Angle Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig('biophysical_ai_results.pdf', dpi=300)
plt.savefig('../paper/biophysical_ai_results.pdf', dpi=300)  # also copy to paper/
plt.show()
