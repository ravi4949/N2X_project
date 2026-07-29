"""
================================================================================
N2X Model : Evaluation + UCM Comparison (3D)
================================================================================
Evaluates:
    1. 3D MSE, R², Relative Frobenius Error
    2. Comparison of N2X-predicted 3D stress with analytical UCM model
    3. 3D Forward integration of conformation tensor using learned closure
    4. Tensor trace comparison tr(C) over time in 3D
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os, sys, json

class DumbbellParams:
    H=1.0; b=50.0; kT=1.0; zeta=1.0
    tau=zeta/(4*H); D=kT/zeta

def tensor_invariants(C):
    I1 = np.trace(C)
    I4 = np.trace(C @ C)
    I2 = 0.5 * (I1**2 - I4)
    I3 = np.linalg.det(C)
    return np.array([I1, I2, I3, I4], dtype=np.float64)

def ucm_closure(C, p):
    return (C - np.eye(3)) / p.tau

def fene_p_closure(C, p):
    trC   = np.trace(C)
    denom = max(1.0 - trC / p.b, 1e-6)
    return (4.0 * p.H / p.zeta) * C / denom - 2.0 * p.kT / p.zeta * np.eye(3)

def shear_flow_kappa(g):
    k = np.zeros((3, 3), dtype=np.float64)
    k[0, 1] = g
    return k

def fene_force(r, p):
    r2 = np.sum(r**2, axis=1, keepdims=True)
    return p.H * r / np.maximum(1.0 - r2 / p.b, 1e-4)

def conformation_tensor(r):
    return np.einsum('ni,nj->ij', r, r) / r.shape[0]

def euler_maruyama_step(r, kappa, dt, p, rng):
    F     = fene_force(r, p)
    drift = (r @ kappa.T) - F / p.zeta
    noise = np.sqrt(2.0 * p.kT / p.zeta * dt) * rng.standard_normal(r.shape)
    return r + drift * dt + noise


def evaluate_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                     label: str = "Model") -> dict:
    mse      = float(np.mean((Y_true - Y_pred)**2))
    ss_res   = np.sum((Y_true - Y_pred)**2)
    ss_tot   = np.sum((Y_true - Y_true.mean(0))**2)
    r2       = float(1 - ss_res / ss_tot)
    rel_frob = float(np.mean(
        np.linalg.norm(Y_true - Y_pred, axis=1) /
        (np.linalg.norm(Y_true, axis=1) + 1e-12)
    ))
    per_component_mse = np.mean((Y_true - Y_pred)**2, axis=0)

    print(f"\n  ── {label} (3D) ──────────────────────────────")
    print(f"  MSE (overall)         : {mse:.6f}")
    print(f"  R²                    : {r2:.6f}")
    print(f"  Relative Frob. error  : {rel_frob*100:.3f} %")
    print(f"  Per-component MSE     : "
          f"Ωxx={per_component_mse[0]:.4f}  "
          f"Ωxy={per_component_mse[1]:.4f}  "
          f"Ωxz={per_component_mse[2]:.4f}\n"
          f"                          "
          f"Ωyy={per_component_mse[3]:.4f}  "
          f"Ωyz={per_component_mse[4]:.4f}  "
          f"Ωzz={per_component_mse[5]:.4f}")

    return {'mse': mse, 'r2': r2, 'rel_frob': rel_frob,
            'per_comp_mse': per_component_mse.tolist()}


class InferenceModel:
    """Loads 3D weights and runs forward pass."""
    def __init__(self, path_npz: str, path_scaler_dir: str):
        data   = np.load(path_npz)
        sdir   = path_scaler_dir
        self.weights = [(data[f'layer_{i}_W'], data[f'layer_{i}_b'])
                        for i in range(4)]
        self.X_mean  = np.load(f'{sdir}/scaler_X_mean.npy')
        self.X_std   = np.load(f'{sdir}/scaler_X_std.npy')
        self.Y_mean  = np.load(f'{sdir}/scaler_Y_mean.npy')
        self.Y_std   = np.load(f'{sdir}/scaler_Y_std.npy')

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(self.weights):
            x = x @ W + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
        return x

    def predict(self, X_physical: np.ndarray) -> np.ndarray:
        Xn = (X_physical - self.X_mean) / self.X_std
        Yn = self.predict_raw(Xn)
        return Yn * self.Y_std + self.Y_mean


def integrate_with_closure(closure_fn,
                            kappa: np.ndarray,
                            C0: np.ndarray,
                            dt: float,
                            N_steps: int) -> np.ndarray:
    """
    Integrate dC/dt = κC + CκT − Closure(C) for 3D tensor C (3, 3).
    Returns trajectory of shape (N_steps+1, 3, 3).
    """
    C_traj    = np.zeros((N_steps + 1, 3, 3))
    C_traj[0] = C0.copy()
    C         = C0.copy()
    for step in range(N_steps):
        Omega = closure_fn(C, kappa)
        dC    = kappa @ C + C @ kappa.T - Omega
        C     = C + dt * dC
        C     = 0.5 * (C + C.T)   # Symmetrize to prevent numerical drift
        C_traj[step + 1] = C
    return C_traj


def nn_closure_fn(model: InferenceModel):
    def fn(C: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        inv   = tensor_invariants(C)
        kflat = kappa.flatten()
        feat  = np.concatenate([inv, kflat])[np.newaxis, :]   # (1, 13)
        out   = model.predict(feat)[0]                        # (6,)
        Omega = np.array([
            [out[0], out[1], out[2]],
            [out[1], out[3], out[4]],
            [out[2], out[4], out[5]]
        ], dtype=np.float64)
        return Omega
    return fn

def ucm_closure_fn(C: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    p = DumbbellParams()
    return ucm_closure(C, p)

def fene_p_closure_fn(C: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    p = DumbbellParams()
    return fene_p_closure(C, p)


def reference_trajectory(kappa, N_dumbbells, N_steps, dt):
    p   = DumbbellParams()
    rng = np.random.default_rng(777)
    r   = rng.normal(0, np.sqrt(p.kT / p.H), (N_dumbbells, 3))
    C_traj = np.zeros((N_steps + 1, 3, 3))
    C_traj[0] = conformation_tensor(r)
    for step in range(N_steps):
        r = euler_maruyama_step(r, kappa, dt, p, rng)
        C_traj[step + 1] = conformation_tensor(r)
    return C_traj


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60)
    print("  STATIC CLOSURE EVALUATION (3D Test Set)")
    print("="*60)

    Y_pred = np.load('results/Y_pred.npy')
    Y_true = np.load('results/Y_true.npy')
    X_test = np.load('results/X_test.npy')    # shape (N, 13)

    nn_metrics = evaluate_metrics(Y_true, Y_pred, label="N2X NN (3D)")

    p = DumbbellParams()
    ucm_preds = []
    for feat in X_test:
        I1 = feat[0]
        C_approx  = np.eye(3) * (I1 / 3.0)
        Omega_ucm = ucm_closure(C_approx, p)
        ucm_preds.append([Omega_ucm[0,0], Omega_ucm[0,1], Omega_ucm[0,2],
                          Omega_ucm[1,1], Omega_ucm[1,2], Omega_ucm[2,2]])
    ucm_preds   = np.array(ucm_preds)
    ucm_metrics = evaluate_metrics(Y_true, ucm_preds, label="UCM (3D Analytical)")

    print("\n" + "="*60)
    print("  3D FORWARD TRAJECTORY COMPARISON (shear flow)")
    print("="*60)

    model  = InferenceModel('models/N2X_model.npz', 'models')
    nn_fn  = nn_closure_fn(model)

    kappa  = shear_flow_kappa(1.0)
    dt     = 0.005
    N_steps= 500
    C0     = np.eye(3)

    print("  Running 3D reference Brownian dynamics …")
    C_ref  = reference_trajectory(kappa, 5000, N_steps, dt)

    print("  Integrating with 3D N2X closure …")
    C_nn   = integrate_with_closure(nn_fn,        kappa, C0, dt, N_steps)

    print("  Integrating with 3D UCM closure …")
    C_ucm  = integrate_with_closure(ucm_closure_fn, kappa, C0, dt, N_steps)

    print("  Integrating with 3D FENE-P closure …")
    C_fenep= integrate_with_closure(fene_p_closure_fn, kappa, C0, dt, N_steps)

    H = p.H
    tau_ref   = H * C_ref   - np.eye(3)
    tau_nn    = H * C_nn    - np.eye(3)
    tau_ucm   = H * C_ucm   - np.eye(3)
    tau_fenep = H * C_fenep - np.eye(3)

    time_arr = np.arange(N_steps + 1) * dt
    np.save('results/time.npy',     time_arr)
    np.save('results/C_ref.npy',    C_ref)
    np.save('results/C_nn.npy',     C_nn)
    np.save('results/C_ucm.npy',    C_ucm)
    np.save('results/C_fenep.npy',  C_fenep)
    np.save('results/tau_ref.npy',  tau_ref)
    np.save('results/tau_nn.npy',   tau_nn)
    np.save('results/tau_ucm.npy',  tau_ucm)

    def traj_metrics(C_pred, C_true, name):
        tr_pred = np.trace(C_pred, axis1=1, axis2=2)
        tr_true = np.trace(C_true, axis1=1, axis2=2)
        mse_tr  = float(np.mean((tr_pred - tr_true)**2))
        print(f"  {name:18s}  tr(C) MSE = {mse_tr:.5f}")
        return mse_tr

    print("\n  Trajectory tr(C) metrics (3D):")
    traj_metrics(C_nn,    C_ref, "N2X NN (3D)")
    traj_metrics(C_ucm,   C_ref, "UCM (3D)")
    traj_metrics(C_fenep, C_ref, "FENE-P (3D)")

    summary = {
        'nn':   nn_metrics,
        'ucm':  ucm_metrics,
    }
    with open('results/eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n3D Evaluation complete ✓")
