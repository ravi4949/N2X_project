"""
================================================================================
N2X Model — Task 1 & 2 & 3 & 4: Dataset Generation
================================================================================
Project : Machine Learning Based Modeling of Non-Newtonian Fluids (N2X)
Author  : Raveendra
Description:
    Simulates 2D dumbbell polymer dynamics using Stochastic Differential
    Equations (SDE).  Builds a dataset of (C, κ, C_next) triples used to
    train the neural-network closure model.

Physical background
-------------------
* A polymer chain is idealised as two beads connected by a spring (dumbbell).
* r = (rx, ry)  — end-to-end connector vector
* The spring force follows the FENE (Finite Extensible Nonlinear Elastic)
  model:  F = H·r / (1 − |r|²/b)
* The stochastic equation of motion (Ito SDE) is
      dr = (κ·r − F/ζ) dt + √(2kT/ζ) dW
  where  ζ = friction,  kT = thermal energy,  dW = Wiener increment
* The conformation tensor  C = ⟨r rᵀ⟩  is the ensemble average of the
  outer product of connector vectors across many dumbbells.
* Evolution equation:
      dC/dt = κC + CκT − Closure(C)
  The N2X neural network learns Closure(C) from data.
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class DumbbellParams:
    """All dimensional / dimensionless parameters for the FENE dumbbell."""
    H      = 1.0      # Spring constant  [non-dim]
    b      = 50.0     # Finite-extensibility parameter  (max |r|² = b)
    kT     = 1.0      # Thermal energy   [non-dim]
    zeta   = 1.0      # Friction coefficient [non-dim]
    tau    = zeta / (4 * H)   # Relaxation time  λ = ζ/(4H)
    D      = kT / zeta        # Diffusion coefficient


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FENE SPRING FORCE
# ─────────────────────────────────────────────────────────────────────────────
def fene_force(r: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """
    FENE connector force  F = H·r / (1 − |r|²/b)
    If |r|² ≥ b the chain is fully stretched — we clip to avoid blow-up.
    
    Parameters
    ----------
    r : shape (N, 2)  — connector vectors of N dumbbells
    p : DumbbellParams
    
    Returns
    -------
    F : shape (N, 2)
    """
    r2   = np.sum(r**2, axis=1, keepdims=True)          # |r|², shape (N,1)
    denom = 1.0 - np.clip(r2 / p.b, None, 0.9999)       # avoid /0
    return p.H * r / denom


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SINGLE SDE TIME-STEP  (Euler–Maruyama)
# ─────────────────────────────────────────────────────────────────────────────
def euler_maruyama_step(r: np.ndarray,
                        kappa: np.ndarray,
                        dt: float,
                        p: DumbbellParams,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Advance the connector vector by one Euler-Maruyama step.

    SDE:  dr = [κ·r  −  F(r)/ζ] dt  +  √(2kT/ζ) dW

    Parameters
    ----------
    r     : (N, 2) connector vectors at time t
    kappa : (2, 2) velocity-gradient tensor κ
    dt    : time-step
    p     : DumbbellParams
    rng   : NumPy random generator

    Returns
    -------
    r_new : (N, 2) connector vectors at time t+dt
    """
    N = r.shape[0]
    F        = fene_force(r, p)                             # (N, 2) spring force
    drift    = (r @ kappa.T) - F / p.zeta                  # (N, 2)
    noise    = np.sqrt(2.0 * p.kT / p.zeta * dt) * rng.standard_normal((N, 2))
    return r + drift * dt + noise


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONFORMATION TENSOR FROM ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def conformation_tensor(r: np.ndarray) -> np.ndarray:
    """
    C = ⟨r rᵀ⟩  — ensemble average of outer products.

    Returns the UNIQUE components as a (2,2) matrix.
    For 2-D:  C = [[Cxx, Cxy],
                   [Cxy, Cyy]]   (symmetric)
    """
    return np.einsum('ni,nj->ij', r, r) / r.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ANALYTICAL UCM CLOSURE  (for comparison / ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def ucm_closure(C: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """
    Upper Convected Maxwell (UCM / Hookean dumbbell) closure.
    For a Hookean spring:  Closure = (C − I) / λ
    where λ = relaxation time,  I = identity.
    This is the *linear* baseline the neural network must beat.
    """
    lam = p.tau
    return (C - np.eye(2)) / lam


def fene_p_closure(C: np.ndarray, p: DumbbellParams) -> np.ndarray:
    """
    FENE-P closure (Peterlin approximation):
      Closure = H·C / (1 − tr(C)/b) / ζ × 4   (returns relaxation term)
    Exact for the FENE dumbbell at the mean-field level.
    """
    trC   = np.trace(C)
    denom = 1.0 - trC / p.b
    denom = max(denom, 1e-6)
    return (4.0 * p.H / p.zeta) * C / denom  - 2.0 * p.kT / p.zeta * np.eye(2)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VELOCITY-GRADIENT GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def shear_flow_kappa(gamma_dot: float) -> np.ndarray:
    """Simple shear:  κ = [[0, γ̇], [0, 0]]"""
    return np.array([[0.0, gamma_dot],
                     [0.0, 0.0]])

def extensional_flow_kappa(epsilon_dot: float) -> np.ndarray:
    """Planar extension:  κ = [[ε̇, 0], [0, -ε̇]]"""
    return np.array([[epsilon_dot, 0.0],
                     [0.0, -epsilon_dot]])

def random_kappa(scale: float, rng: np.random.Generator) -> np.ndarray:
    """Random velocity gradient with given magnitude scale."""
    k = rng.uniform(-scale, scale, (2, 2))
    return k


# ─────────────────────────────────────────────────────────────────────────────
# 7.  COMPUTE TENSOR INVARIANTS  (input features for the NN)
# ─────────────────────────────────────────────────────────────────────────────
def tensor_invariants(C: np.ndarray) -> np.ndarray:
    """
    For a 2×2 symmetric tensor C the independent invariants are:
        I1 = tr(C)              — trace
        I2 = det(C)             — determinant
        I3 = tr(C²) = I1² - 2·I2
    These are *rotationally invariant* scalars — key for frame-indifference.
    Returns array [I1, I2, I3].
    """
    I1 = np.trace(C)
    I2 = np.linalg.det(C)
    I3 = np.trace(C @ C)
    return np.array([I1, I2, I3])


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL SIMULATION — GENERATE ONE TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trajectory(kappa: np.ndarray,
                         N_dumbbells: int,
                         N_steps: int,
                         dt: float,
                         p: DumbbellParams,
                         rng: np.random.Generator) -> dict:
    """
    Run a Brownian dynamics simulation of N_dumbbells FENE dumbbells for
    N_steps time steps with constant velocity gradient κ.

    Returns
    -------
    dict with keys:
        'C_traj'   : (N_steps+1, 2, 2)  conformation tensor trajectory
        'kappa'    : (2, 2)              velocity gradient (constant)
        'closure'  : (N_steps, 2, 2)    measured closure term
        'dt'       : float
    """
    params = p
    # initialise equilibrium: r ~ N(0, √(b/3)·I)
    r = rng.normal(0, np.sqrt(params.kT / params.H), (N_dumbbells, 2))

    C_traj   = np.zeros((N_steps + 1, 2, 2))
    C_traj[0] = conformation_tensor(r)

    closures = np.zeros((N_steps, 2, 2))

    for step in range(N_steps):
        C_old    = conformation_tensor(r)
        r        = euler_maruyama_step(r, kappa, dt, params, rng)
        C_new    = conformation_tensor(r)
        C_traj[step + 1] = C_new

        # Numerically back-out the closure term from the evolution equation:
        #   closure ≈ (κC + CκT) - (C_new - C_old)/dt
        transport   = kappa @ C_old + C_old @ kappa.T
        dC_dt       = (C_new - C_old) / dt
        closures[step] = transport - dC_dt

    return {'C_traj': C_traj, 'kappa': kappa,
            'closure': closures, 'dt': dt}


# ─────────────────────────────────────────────────────────────────────────────
# 9.  BUILD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(n_trajectories: int = 500,
                  N_dumbbells: int   = 2000,
                  N_steps: int       = 200,
                  dt: float          = 0.01,
                  seed: int          = 42) -> dict:
    """
    Generate a dataset of (features, targets) for training.

    Features (7 values per sample):
        [I1, I2, I3,  κxx, κxy, κyx, κyy]

    Targets (3 values — unique entries of symmetric 2×2 closure tensor):
        [Ω_xx, Ω_xy, Ω_yy]

    Also stores C_now (2×2) and C_next (2×2) for evaluation.
    """
    rng    = np.random.default_rng(seed)
    p      = DumbbellParams()

    all_features   = []   # (7,)
    all_targets    = []   # (3,)
    all_C_now      = []   # (2,2)
    all_C_next     = []   # (2,2)
    all_kappa      = []   # (4,)

    flow_types = ['shear', 'extension', 'random']

    print(f"Generating {n_trajectories} trajectories …")
    for traj_i in range(n_trajectories):
        # pick a random flow type and intensity
        ftype = rng.choice(flow_types)
        scale = rng.uniform(0.05, 2.0)

        if ftype == 'shear':
            kappa = shear_flow_kappa(scale)
        elif ftype == 'extension':
            kappa = extensional_flow_kappa(scale)
        else:
            kappa = random_kappa(scale * 0.5, rng)

        result = simulate_trajectory(kappa, N_dumbbells, N_steps, dt, p, rng)
        C_traj   = result['C_traj']    # (N_steps+1, 2,2)
        closures = result['closure']   # (N_steps,   2,2)

        for step in range(N_steps):
            C_now  = C_traj[step]
            C_next = C_traj[step + 1]
            Omega  = closures[step]    # 2×2 symmetric

            inv    = tensor_invariants(C_now)           # [I1,I2,I3]
            kappa_flat = kappa.flatten()                # [κxx,κxy,κyx,κyy]

            features = np.concatenate([inv, kappa_flat])   # length 7
            targets  = np.array([Omega[0,0], Omega[0,1], Omega[1,1]])  # 3

            all_features.append(features)
            all_targets.append(targets)
            all_C_now.append(C_now.flatten())
            all_C_next.append(C_next.flatten())
            all_kappa.append(kappa_flat)

        if (traj_i + 1) % 100 == 0:
            print(f"  … {traj_i+1}/{n_trajectories} done")

    dataset = {
        'X'      : np.array(all_features, dtype=np.float32),   # (M, 7)
        'Y'      : np.array(all_targets,  dtype=np.float32),   # (M, 3)
        'C_now'  : np.array(all_C_now,    dtype=np.float32),   # (M, 4)
        'C_next' : np.array(all_C_next,   dtype=np.float32),   # (M, 4)
        'kappa'  : np.array(all_kappa,    dtype=np.float32),   # (M, 4)
    }
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
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

    # Quick sanity-check statistics
    X, Y = dataset['X'], dataset['Y']
    print(f"\nDataset size   : {X.shape[0]:,} samples")
    print(f"Feature range  : [{X.min():.3f}, {X.max():.3f}]")
    print(f"Target range   : [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"Target std     : {Y.std(axis=0)}")
    print("\nDataset generation complete ✓")
