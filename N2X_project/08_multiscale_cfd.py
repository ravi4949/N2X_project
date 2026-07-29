"""
================================================================================
N2X Model : 3D Multi-Scale Viscoelastic CFD Solver (08_multiscale_cfd.py)
================================================================================
Couples trained 3D N2X Neural Closure Model with a macroscopic 3D fluid 
mechanics solver (3D channel / duct flow under pressure gradient).

Governing Equations:
    1. Momentum Transport:
       ρ ∂u_x/∂t = f_x + η_s (∂²u_x/∂y² + ∂²u_x/∂z²) + ∂τ_xy/∂y + ∂τ_xz/∂z

    2. Polymer Conformation Dynamics (Live Neural Coupling):
       ∂C/∂t = κ C + C κᵀ − Ω_N2X(C, κ)

       where κ(y,z) is local 3D velocity gradient tensor:
       κ_xy = ∂u_x/∂y,  κ_xz = ∂u_x/∂z

    3. Polymer Stress:
       τ_p = n·kT (C − I₃) / λ

Simulates non-Newtonian flow development and steady-state velocity/stress 
profiles across 3D cross-section. Saves outputs to results/cfd_results.npz.
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os, time, json

class DumbbellParams:
    H = 1.0; b = 50.0; kT = 1.0; zeta = 1.0
    tau = zeta / (4 * H); D = kT / zeta

def tensor_invariants(C: np.ndarray) -> np.ndarray:
    I1 = np.trace(C)
    I4 = np.trace(C @ C)
    I2 = 0.5 * (I1**2 - I4)
    I3 = np.linalg.det(C)
    return np.array([I1, I2, I3, I4], dtype=np.float64)

class N2XCFDInferenceModel:
    """Fast vectorized inference model for CFD solver coupling."""
    def __init__(self, model_path: str, scaler_dir: str):
        data = np.load(model_path)
        self.weights = [(data[f'layer_{i}_W'], data[f'layer_{i}_b']) for i in range(4)]
        self.X_mean  = np.load(f'{scaler_dir}/scaler_X_mean.npy')
        self.X_std   = np.load(f'{scaler_dir}/scaler_X_std.npy')
        self.Y_mean  = np.load(f'{scaler_dir}/scaler_Y_mean.npy')
        self.Y_std   = np.load(f'{scaler_dir}/scaler_Y_std.npy')

    def predict_batch(self, X_phys: np.ndarray) -> np.ndarray:
        Xn = (X_phys - self.X_mean) / self.X_std
        x = Xn
        for i, (W, b) in enumerate(self.weights):
            x = x @ W + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
        Yn = x
        return Yn * self.Y_std + self.Y_mean


def run_multiscale_3d_cfd(Ny: int = 31, Nz: int = 31, n_steps: int = 400, dt: float = 0.002):
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60)
    print("  3D MULTI-SCALE VISCOELASTIC CFD SOLVER (N2X Live Coupling)")
    print("="*60)

    # Domain & Grid Setup
    Ly, Lz = 2.0, 2.0  # y ∈ [-1, 1], z ∈ [-1, 1]
    y = np.linspace(-1.0, 1.0, Ny)
    z = np.linspace(-1.0, 1.0, Nz)
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # Fluid Parameters
    rho   = 1.0      # density
    eta_s = 0.5      # solvent viscosity
    f_x   = 2.0      # driving pressure gradient force (-dp/dx)
    p_phys= DumbbellParams()

    # Model Inference Coupling
    nn_model = N2XCFDInferenceModel('models/N2X_model.npz', 'models')

    # Initial Fields
    ux = np.zeros((Ny, Nz), dtype=np.float64)       # Velocity profile
    C  = np.zeros((Ny, Nz, 3, 3), dtype=np.float64)  # Conformation tensor grid
    for j in range(Ny):
        for k in range(Nz):
            C[j, k] = np.eye(3, dtype=np.float64)

    t0 = time.time()

    print(f"  Grid Size: {Ny} × {Nz} ({Ny*Nz} cells)  | Time Steps: {n_steps} (dt = {dt})")
    print(f"  Driving Pressure Gradient: f_x = {f_x}")
    print("  Advancing coupled Navier-Stokes + N2X Conformation PDE …")

    history_u_max = []
    history_trC_max = []

    for step in range(n_steps):
        # 1. Compute velocity gradients via central differences (zero at no-slip boundaries)
        du_dy = np.zeros((Ny, Nz), dtype=np.float64)
        du_dz = np.zeros((Ny, Nz), dtype=np.float64)

        du_dy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / (2.0 * dy)
        du_dz[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / (2.0 * dz)

        # 2. Vectorized Feature Extraction for Neural Network
        X_features = []
        grid_indices = []

        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                C_jk = C[j, k]
                inv  = tensor_invariants(C_jk)
                kappa_jk = np.zeros((3, 3), dtype=np.float64)
                kappa_jk[0, 1] = du_dy[j, k]
                kappa_jk[0, 2] = du_dz[j, k]
                
                feat = np.concatenate([inv, kappa_jk.flatten()])
                X_features.append(feat)
                grid_indices.append((j, k))

        X_features = np.array(X_features, dtype=np.float64)
        
        # Batch Predict N2X Neural Closure
        Y_pred = nn_model.predict_batch(X_features)

        # 3. Update Conformation Tensor C using predicted closure Ω_N2X
        dC_dt_grid = np.zeros((Ny, Nz, 3, 3), dtype=np.float64)
        tau_p      = np.zeros((Ny, Nz, 3, 3), dtype=np.float64)

        for idx, (j, k) in enumerate(grid_indices):
            out = Y_pred[idx]
            Omega_N2X = np.array([
                [out[0], out[1], out[2]],
                [out[1], out[3], out[4]],
                [out[2], out[4], out[5]]
            ])

            C_jk = C[j, k]
            kappa_jk = np.zeros((3, 3), dtype=np.float64)
            kappa_jk[0, 1] = du_dy[j, k]
            kappa_jk[0, 2] = du_dz[j, k]

            transport = kappa_jk @ C_jk + C_jk @ kappa_jk.T
            dC_dt = transport - Omega_N2X
            
            C[j, k] += dt * dC_dt
            C[j, k]  = 0.5 * (C[j, k] + C[j, k].T)  # Keep symmetric

            # Polymer stress: τ_p = (C - I3) / τ
            tau_p[j, k] = (C[j, k] - np.eye(3)) / p_phys.tau

        # 4. Momentum Equation Update for u_x
        # Solvent viscous term: η_s * (∂²u/∂y² + ∂²u/∂z²)
        d2u_dy2 = (ux[2:, 1:-1] - 2.0*ux[1:-1, 1:-1] + ux[:-2, 1:-1]) / (dy**2)
        d2u_dz2 = (ux[1:-1, 2:] - 2.0*ux[1:-1, 1:-1] + ux[1:-1, :-2]) / (dz**2)
        visc_solvent = eta_s * (d2u_dy2 + d2u_dz2)

        # Polymer stress divergence term: ∂τ_xy/∂y + ∂τ_xz/∂z
        dtau_xy_dy = (tau_p[2:, 1:-1, 0, 1] - tau_p[:-2, 1:-1, 0, 1]) / (2.0 * dy)
        dtau_xz_dz = (tau_p[1:-1, 2:, 0, 2] - tau_p[1:-1, :-2, 0, 2]) / (2.0 * dz)
        poly_stress_div = dtau_xy_dy + dtau_xz_dz

        # Update inner domain velocity
        ux[1:-1, 1:-1] += dt / rho * (f_x + visc_solvent + poly_stress_div)

        # No-slip Boundary Conditions: u_x = 0 at y = ±1, z = ±1
        ux[0, :] = 0.0; ux[-1, :] = 0.0
        ux[:, 0] = 0.0; ux[:, -1] = 0.0

        trC = np.trace(C, axis1=2, axis2=3)
        history_u_max.append(float(np.max(ux)))
        history_trC_max.append(float(np.max(trC)))

        if (step + 1) % 100 == 0 or step == 0:
            print(f"    Step {step+1:>4}/{n_steps}  |  u_max = {np.max(ux):.4f}  |  tr(C)_max = {np.max(trC):.4f}")

    elapsed = time.time() - t0
    print(f"\n  3D CFD Simulation Complete in {elapsed:.1f} s")

    # Analytical Newtonian benchmark velocity for comparison
    # u_Newtonian = f_x / (2*eta_s) * (1 - y^2) in 1D slice
    u_newtonian_mid = f_x / (2.0 * eta_s) * (1.0 - y**2)

    # Extract 1D centerline profiles
    mid_z = Nz // 2
    u_centerline_n2x = ux[:, mid_z]
    tau_xy_centerline = tau_p[:, mid_z, 0, 1]
    trC_centerline    = np.trace(C[:, mid_z], axis1=1, axis2=2)

    np.savez('results/cfd_results.npz',
             y=y, z=z, ux=ux, C=C, tau_p=tau_p,
             u_newtonian_mid=u_newtonian_mid,
             u_centerline_n2x=u_centerline_n2x,
             tau_xy_centerline=tau_xy_centerline,
             trC_centerline=trC_centerline,
             history_u_max=history_u_max,
             history_trC_max=history_trC_max)

    summary = {
        'grid_size': [Ny, Nz],
        'n_steps': n_steps,
        'u_max_n2x': float(np.max(ux)),
        'u_max_newtonian': float(np.max(u_newtonian_mid)),
        'trC_max': float(np.max(trC_centerline)),
        'tau_xy_max': float(np.max(np.abs(tau_xy_centerline))),
        'simulation_time_sec': elapsed
    }
    with open('results/cfd_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  CFD Results saved to results/cfd_results.npz and results/cfd_summary.json ✓")
    return summary

if __name__ == '__main__':
    run_multiscale_3d_cfd()
