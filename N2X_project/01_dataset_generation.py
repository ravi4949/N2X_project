"""
================================================================================
N2X Model :  Dataset Generation (3D Formulation)
================================================================================
Project : Machine Learning Based Modeling of Non-Newtonian Fluids (N2X 3D)
Author  : Raveendra
Description:
    Simulates 3D dumbbell polymer dynamics using Stochastic Differential
    Equations (SDE). Builds a dataset of (C, κ, C_next) triples used to
    train the 3D neural-network closure model.

Physical background (3D)
------------------------
* A polymer chain is idealised as two beads connected by a spring (dumbbell).
* r = (rx, ry, rz) — 3D end-to-end connector vector
* The spring force follows the FENE (Finite Extensible Nonlinear Elastic)
  model: F = H·r / (1 − |r|²/b)
* The stochastic equation of motion (Ito SDE) is
      dr = (κ·r − F/ζ) dt + √(2kT/ζ) dW
  where ζ = friction, kT = thermal energy, dW = 3D Wiener increment
* The 3D conformation tensor C = ⟨r rᵀ⟩ is the ensemble average of the
  outer product of connector vectors across many dumbbells.
* Evolution equation:
      dC/dt = κC + CκT − Closure(C)
  The N2X neural network learns Closure(C) from 3D data.
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. PHYSICAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class DumbbellParams:
    """All dimensional / dimensionless parameters for the 3D FENE dumbbell."""
    H      = 1.0      # Spring constant [non-dim]
    b      = 50.0     # Finite-extensibility parameter (max |r|² = b)
    kT     = 1.0      # Thermal energy [non-dim]
    zeta   = 1.0      # Friction coefficient [non-dim]
    tau    = zeta / (4 * H)   # Relaxation time λ = ζ/(4H)
    D      = kT / zeta        # Diffusion coefficient


# ─────────────────────────────────────────────────────────────────────────────
# 2. 3D FENE SPRING FORCE
# ─────────────────────────────────────────────────────────────────────────────
def fene_force(r: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """
    3D FENE connector force F = H·r / (1 − |r|²/b)
    
    Parameters
    ----------
    r : shape (N, 3) — 3D connector vectors of N dumbbells
    p : DumbbellParams
    
    Returns
    -------
    F : shape (N, 3)
    """
    r2    = np.sum(r**2, axis=1, keepdims=True)          # |r|², shape (N,1)
    denom = 1.0 - np.clip(r2 / p.b, None, 0.9999)       # avoid /0
    return p.H * r / denom


# ─────────────────────────────────────────────────────────────────────────────
# 3. SINGLE 3D SDE TIME-STEP (Euler–Maruyama)
# ─────────────────────────────────────────────────────────────────────────────
def euler_maruyama_step(r: np.ndarray,
                        kappa: np.ndarray,
                        dt: float,
                        p: DumbbellParams,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Advance 3D connector vectors by one Euler-Maruyama step.

    SDE: dr = [κ·r − F(r)/ζ] dt + √(2kT/ζ) dW
    """
    N     = r.shape[0]
    F     = fene_force(r, p)                            # (N, 3) spring force
    drift = (r @ kappa.T) - F / p.zeta                  # (N, 3)
    noise = np.sqrt(2.0 * p.kT / p.zeta * dt) * rng.standard_normal((N, 3))
    return r + drift * dt + noise


# ─────────────────────────────────────────────────────────────────────────────
# 4. 3D CONFORMATION TENSOR FROM ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def conformation_tensor(r: np.ndarray) -> np.ndarray:
    """
    C = ⟨r rᵀ⟩ — ensemble average of outer products.
    Returns (3, 3) symmetric matrix.
    """
    return np.einsum('ni,nj->ij', r, r) / r.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYTICAL CLOSURES (3D)
# ─────────────────────────────────────────────────────────────────────────────
def ucm_closure(C: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """3D Upper Convected Maxwell closure: (C − I3) / λ"""
    return (C - np.eye(3)) / p.tau

def fene_p_closure(C: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """3D FENE-P closure (Peterlin approximation)"""
    trC   = np.trace(C)
    denom = max(1.0 - trC / p.b, 1e-6)
    return (4.0 * p.H / p.zeta) * C / denom - 2.0 * p.kT / p.zeta * np.eye(3)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 3D VELOCITY-GRADIENT GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def shear_flow_kappa(gamma_dot: float) -> np.ndarray:
    """3D Simple shear flow: κ_01 = γ̇"""
    k = np.zeros((3, 3), dtype=np.float64)
    k[0, 1] = gamma_dot
    return k

def extensional_flow_kappa(epsilon_dot: float) -> np.ndarray:
    """3D Uniaxial extension flow: κ = diag(ε̇, -0.5ε̇, -0.5ε̇)"""
    return np.diag([epsilon_dot, -0.5 * epsilon_dot, -0.5 * epsilon_dot])

def random_kappa(scale: float, rng: np.random.Generator) -> np.ndarray:
    """Random 3D velocity gradient matrix."""
    return rng.uniform(-scale, scale, (3, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPUTE 3D TENSOR INVARIANTS
# ─────────────────────────────────────────────────────────────────────────────
def tensor_invariants(C: np.ndarray) -> np.ndarray:
    """
    Scalar rotational invariants for a 3×3 matrix C:
        I1 = tr(C)
        I2 = 0.5 * ((tr(C))^2 - tr(C^2))
        I3 = det(C)
        I4 = tr(C^2)
    Returns array [I1, I2, I3, I4] (length 4).
    """
    I1 = np.trace(C)
    I4 = np.trace(C @ C)
    I2 = 0.5 * (I1**2 - I4)
    I3 = np.linalg.det(C)
    return np.array([I1, I2, I3, I4], dtype=np.float64)


def exact_ensemble_closure(r: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """
    Exact microscopic kinetic ensemble closure tensor for FENE dumbbells in 3D:
        Ω = (2H / ζ) * ⟨ (r rᵀ) / (1 - |r|²/b) ⟩ - (2kT / ζ) * I₃
    """
    r2 = np.sum(r**2, axis=1, keepdims=True)
    denom = np.maximum(1.0 - r2 / p.b, 1e-4)
    weighted_rrT = (r[:, :, np.newaxis] * r[:, np.newaxis, :]) / denom[:, :, np.newaxis]
    avg_weighted = np.mean(weighted_rrT, axis=0)
    return (2.0 * p.H / p.zeta) * avg_weighted - (2.0 * p.kT / p.zeta) * np.eye(3)


# ─────────────────────────────────────────────────────────────────────────────
# 8. 3D SIMULATION — GENERATE ONE TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trajectory(kappa: np.ndarray,
                        N_dumbbells: int,
                        N_steps: int,
                        dt: float,
                        p: DumbbellParams,
                        rng: np.random.Generator) -> dict:
    # initialise equilibrium 3D connector: r ~ N(0, √(kT/H))
    r = rng.normal(0, np.sqrt(p.kT / p.H), (N_dumbbells, 3))

    C_traj   = np.zeros((N_steps + 1, 3, 3))
    C_traj[0] = conformation_tensor(r)
    closures = np.zeros((N_steps, 3, 3))

    for step in range(N_steps):
        closures[step]   = exact_ensemble_closure(r, p)
        r                = euler_maruyama_step(r, kappa, dt, p, rng)
        C_traj[step + 1] = conformation_tensor(r)

    return {'C_traj': C_traj, 'kappa': kappa, 'closure': closures, 'dt': dt}



# ─────────────────────────────────────────────────────────────────────────────
# 9. BUILD 3D DATASET
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(n_trajectories: int = 500,
                  N_dumbbells: int   = 2000,
                  N_steps: int       = 200,
                  dt: float          = 0.01,
                  seed: int          = 42) -> dict:
    """
    Features (13 values per sample):
        [I1, I2, I3, I4,  κxx, κxy, κxz, κyx, κyy, κyz, κzx, κzy, κzz]

    Targets (6 values — unique entries of symmetric 3×3 closure tensor):
        [Ω_xx, Ω_xy, Ω_xz, Ω_yy, Ω_yz, Ω_zz]
    """
    rng = np.random.default_rng(seed)
    p   = DumbbellParams()

    all_features   = []   # (13,)
    all_targets    = []   # (6,)
    all_C_now      = []   # (9,)
    all_C_next     = []   # (9,)
    all_kappa      = []   # (9,)

    flow_types = ['shear', 'extension', 'random']

    print(f"Generating {n_trajectories} 3D trajectories …")
    for traj_i in range(n_trajectories):
        ftype = rng.choice(flow_types)
        scale = rng.uniform(0.05, 2.0)

        if ftype == 'shear':
            kappa = shear_flow_kappa(scale)
        elif ftype == 'extension':
            kappa = extensional_flow_kappa(scale)
        else:
            kappa = random_kappa(scale * 0.5, rng)

        result   = simulate_trajectory(kappa, N_dumbbells, N_steps, dt, p, rng)
        C_traj   = result['C_traj']    # (N_steps+1, 3,3)
        closures = result['closure']   # (N_steps, 3,3)

        for step in range(N_steps):
            C_now  = C_traj[step]
            C_next = C_traj[step + 1]
            Omega  = closures[step]    # 3×3 symmetric

            inv        = tensor_invariants(C_now)               # 4 scalars
            kappa_flat = kappa.flatten()                        # 9 entries

            features = np.concatenate([inv, kappa_flat])       # length 13
            targets  = np.array([Omega[0,0], Omega[0,1], Omega[0,2],
                                 Omega[1,1], Omega[1,2], Omega[2,2]])  # 6 targets

            all_features.append(features)
            all_targets.append(targets)
            all_C_now.append(C_now.flatten())
            all_C_next.append(C_next.flatten())
            all_kappa.append(kappa_flat)

        if (traj_i + 1) % 100 == 0:
            print(f"  … {traj_i+1}/{n_trajectories} done")

    dataset = {
        'X'      : np.array(all_features, dtype=np.float32),   # (M, 13)
        'Y'      : np.array(all_targets,  dtype=np.float32),   # (M, 6)
        'C_now'  : np.array(all_C_now,    dtype=np.float32),   # (M, 9)
        'C_next' : np.array(all_C_next,   dtype=np.float32),   # (M, 9)
        'kappa'  : np.array(all_kappa,    dtype=np.float32),   # (M, 9)
    }
    return dataset


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    dataset = build_dataset(
        n_trajectories = 300,
        N_dumbbells    = 1500,
        N_steps        = 150,
        dt             = 0.01,
        seed           = 42,
    )

    for k, v in dataset.items():
        np.save(f'data/{k}.npy', v)
        print(f"Saved data/{k}.npy  shape={v.shape}  dtype={v.dtype}")

    X, Y = dataset['X'], dataset['Y']
    print(f"\n3D Dataset size: {X.shape[0]:,} samples")
    print(f"X shape: {X.shape}  |  Y shape: {Y.shape}")
    print("3D Dataset generation complete ✓")
