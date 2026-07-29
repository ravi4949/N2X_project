"""
================================================================================
N2X Model — 3D MASTER PIPELINE (Complete End-to-End Execution)
================================================================================
Runs the entire 3D project pipeline seamlessly from scratch:
    Step 1: Generate 3D Brownian dynamics dataset (200 trajectories, N=2500)
    Step 2: Train 3D N2X base neural network (13 inputs → 128 → 128 → 64 → 6)
    Step 3: Benchmark evaluation & comparison with 3D UCM and FENE-P
    Step 4: Train 3D Physics-Informed Neural Network (PINN) with SPD & trace loss
    Step 5: Verify 3D Spectral Decomposition & Rotational Frame Equivariance
    Step 6: Train 3D Tensor Basis Neural Network (TBNN - Pope 1975)
    Step 7: Execute 3D Multi-Scale Viscoelastic CFD Solver Integration
    Step 8: Generate complete 13-figure publication-quality plot suite

Run: python N2X_project/master_pipeline.py  (or python master_pipeline.py)
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os, sys, time, json, importlib, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure working directory is project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if os.path.exists(os.path.join(project_root, 'N2X_project')):
    os.chdir(project_root)
else:
    os.chdir(script_dir)

print("="*75)
print("  N2X 3D — MACHINE LEARNING BASED MODELING OF NON-NEWTONIAN FLUIDS")
print("  COMPLETE MASTER PIPELINE EXECUTION")
print("="*75)

os.makedirs('data',          exist_ok=True)
os.makedirs('models',        exist_ok=True)
os.makedirs('results',       exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

t_start_master = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS LAYER (3D)
# ═══════════════════════════════════════════════════════════════════════════════
class P:
    H = 1.0; b = 50.0; kT = 1.0; zeta = 1.0
    tau = zeta / (4 * H); D = kT / zeta

def fene_force(r):
    r2 = np.sum(r**2, axis=1, keepdims=True)
    return P.H * r / np.maximum(1.0 - r2 / P.b, 1e-4)

def euler_step(r, kappa, dt, rng):
    F = fene_force(r)
    drift = (r @ kappa.T) - F / P.zeta
    noise = np.sqrt(2.0 * P.kT / P.zeta * dt) * rng.standard_normal(r.shape)
    return r + drift * dt + noise

def conf_tensor(r):
    return np.einsum('ni,nj->ij', r, r) / r.shape[0]

def invariants(C):
    I1 = np.trace(C)
    I4 = np.trace(C @ C)
    I2 = 0.5 * (I1**2 - I4)
    I3 = np.linalg.det(C)
    return np.array([I1, I2, I3, I4], dtype=np.float64)

def ucm_closure(C):
    return (C - np.eye(3)) / P.tau

def fenep_closure(C):
    trC = np.trace(C)
    d   = max(1.0 - trC / P.b, 1e-6)
    return (4.0 * P.H / P.zeta) * C / d - 2.0 * P.kT / P.zeta * np.eye(3)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATASET GENERATION (3D)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 1: Generating 3D Dataset from Brownian Dynamics …")
print("─"*65)
t0 = time.time()

N_TRAJ      = 200
N_DUMBBELLS = 2500
N_STEPS     = 100
DT          = 0.01
SEED        = 42

rng_main = np.random.default_rng(SEED)

features_list, targets_list = [], []
C_now_list,  C_next_list    = [], []
kappa_list                  = []

def exact_ensemble_closure(r):
    r2 = np.sum(r**2, axis=1, keepdims=True)
    denom = np.maximum(1.0 - r2 / P.b, 1e-4)
    weighted_rrT = (r[:, :, np.newaxis] * r[:, np.newaxis, :]) / denom[:, :, np.newaxis]
    avg_weighted = np.mean(weighted_rrT, axis=0)
    return (2.0 * P.H / P.zeta) * avg_weighted - (2.0 * P.kT / P.zeta) * np.eye(3)

for traj in range(N_TRAJ):
    flow  = rng_main.choice(['shear', 'extension', 'random'])
    scale = float(rng_main.uniform(0.05, 1.5))
    if flow == 'shear':
        kappa = np.zeros((3, 3), dtype=np.float32)
        kappa[0, 1] = scale
    elif flow == 'extension':
        kappa = np.diag([scale, -0.5 * scale, -0.5 * scale]).astype(np.float32)
    else:
        kappa = rng_main.uniform(-scale * 0.5, scale * 0.5, (3, 3)).astype(np.float32)

    r = rng_main.normal(0, np.sqrt(P.kT / P.H), (N_DUMBBELLS, 3))
    C = conf_tensor(r)

    for step in range(N_STEPS):
        C_old = conf_tensor(r)
        Omega = exact_ensemble_closure(r)
        r     = euler_step(r, kappa, DT, rng_main)
        C_new = conf_tensor(r)

        inv   = invariants(C_old)
        kflat = kappa.flatten()
        features_list.append(np.concatenate([inv, kflat]).astype(np.float32))
        targets_list.append(np.array([Omega[0,0], Omega[0,1], Omega[0,2],
                                      Omega[1,1], Omega[1,2], Omega[2,2]], dtype=np.float32))
        C_now_list.append(C_old.flatten().astype(np.float32))
        C_next_list.append(C_new.flatten().astype(np.float32))
        kappa_list.append(kflat.astype(np.float32))


    if (traj + 1) % 50 == 0:
        print(f"    Trajectory {traj+1}/{N_TRAJ}  ({len(features_list):,} samples so far)")

X = np.array(features_list)
Y = np.array(targets_list)
np.save('data/X.npy', X)
np.save('data/Y.npy', Y)
np.save('data/C_now.npy',  np.array(C_now_list))
np.save('data/C_next.npy', np.array(C_next_list))
np.save('data/kappa.npy',  np.array(kappa_list))

print(f"  3D Dataset: {X.shape[0]:,} samples  | Time: {time.time()-t0:.1f}s")
print(f"  X shape: {X.shape} (13 features)  |  Y shape: {Y.shape} (6 targets)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — NEURAL NETWORK TRAINING (3D BASE MODEL)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 2: Training 3D N2X Base Neural Network …")
print("─"*65)

X64 = X.astype(np.float64)
Y64 = Y.astype(np.float64)

sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X64)
Ys = sY.fit_transform(Y64)

np.save('models/scaler_X_mean.npy', sX.mean_); np.save('models/scaler_X_std.npy', sX.scale_)
np.save('models/scaler_Y_mean.npy', sY.mean_); np.save('models/scaler_Y_std.npy', sY.scale_)

X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(Xs, Ys, test_size=0.20, random_state=7)
X_val, X_te,  Y_val, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=7)
print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")

class Layer:
    def __init__(self, ni, no, act, rng):
        sc = np.sqrt(2. / (ni + no))
        self.W = rng.normal(0, sc, (ni, no)); self.b = np.zeros(no)
        self.act = act
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)

    def fwd(self, x):
        self._x = x; self._z = x @ self.W + self.b
        self._a = np.tanh(self._z) if self.act == 'tanh' else self._z
        return self._a

    def bwd(self, g):
        d = (1 - self._a**2) * g if self.act == 'tanh' else g
        self.dW = self._x.T @ d / len(self._x)
        self.db = d.mean(0)
        return d @ self.W.T

class Net:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.layers = [Layer(13, 128, 'tanh', rng),
                       Layer(128, 128, 'tanh', rng),
                       Layer(128, 64, 'tanh', rng),
                       Layer(64,   6, 'none', rng)]
        self.t = 0

    def fwd(self, x):
        for l in self.layers: x = l.fwd(x)
        return x

    def bwd(self, g):
        for l in reversed(self.layers): g = l.bwd(g)

    def adam(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for l in self.layers:
            for p, g, m, v in [(l.W, l.dW, l.mW, l.vW), (l.b, l.db, l.mb, l.vb)]:
                m[:] = b1 * m + (1 - b1) * g
                v[:] = b2 * v + (1 - b2) * g**2
                p -= lr * (m / (1 - b1**self.t)) / (np.sqrt(v / (1 - b2**self.t)) + eps)

    def save(self):
        d = {f'layer_{i}_W': l.W for i, l in enumerate(self.layers)}
        d.update({f'layer_{i}_b': l.b for i, l in enumerate(self.layers)})
        np.savez('models/N2X_model.npz', **d)

model    = Net()
LR       = 8e-4
EPOCHS   = 300
BATCH    = 256
PATIENCE = 35
history  = {'train': [], 'val': []}
best_val = np.inf; best_w = None; wait = 0
rng_tr   = np.random.default_rng(99)

t0 = time.time()
print(f"\n  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}")
print(f"  {'-'*35}")

for epoch in range(EPOCHS):
    idx = rng_tr.permutation(len(X_tr))
    batch_losses = []
    for s in range(0, len(X_tr), BATCH):
        bi = idx[s:s+BATCH]
        xb, yb = X_tr[bi], Y_tr[bi]
        pred = model.fwd(xb)
        loss = float(np.mean((pred - yb)**2))
        model.bwd(2 * (pred - yb) / len(xb))
        model.adam(LR)
        batch_losses.append(loss)

    if (epoch + 1) % 10 == 0: LR *= 0.97

    val_pred = model.fwd(X_val)
    val_mse  = float(np.mean((val_pred - Y_val)**2))
    tr_mse   = float(np.mean(batch_losses))
    history['train'].append(tr_mse)
    history['val'].append(val_mse)

    if val_mse < best_val:
        best_val = val_mse; best_e = epoch
        best_w   = [(l.W.copy(), l.b.copy()) for l in model.layers]
        wait     = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\n  Early stop epoch {epoch+1}  best_val={best_val:.6f} @ epoch {best_e+1}")
            break

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"  {epoch+1:>6}  {tr_mse:>10.6f}  {val_mse:>10.6f}")

for i, (W, b) in enumerate(best_w):
    model.layers[i].W = W; model.layers[i].b = b
model.save()
print(f"\n  Training time: {time.time()-t0:.1f}s  |  Best val MSE: {best_val:.6f}")

with open('results/history.json', 'w') as f:
    json.dump({'train_loss': history['train'], 'val_loss': history['val']}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EVALUATION & BENCHMARK COMPARISONS (3D)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 3: 3D Evaluation & Benchmark Trajectories …")
print("─"*65)

Y_pred_s = model.fwd(X_te)
Y_pred   = sY.inverse_transform(Y_pred_s)
Y_true   = sY.inverse_transform(Y_te)

mse  = float(np.mean((Y_pred - Y_true)**2))
r2   = float(1 - np.sum((Y_true - Y_pred)**2) / np.sum((Y_true - Y_true.mean(0))**2))
frob = float(np.mean(np.linalg.norm(Y_true - Y_pred, axis=1) / (np.linalg.norm(Y_true, axis=1) + 1e-12)))

print(f"  3D Base Model Test MSE : {mse:.6f}")
print(f"  R² Score               : {r2:.6f}")
print(f"  Rel. Frob. Error       : {frob*100:.3f}%")

np.save('results/Y_pred.npy', Y_pred)
np.save('results/Y_true.npy', Y_true)
np.save('results/X_test.npy', sX.inverse_transform(X_te))

def nn_closure(C, kappa):
    inv  = invariants(C); kf = kappa.flatten()
    feat = np.concatenate([inv, kf]).reshape(1, -1)
    feats= (feat - sX.mean_) / sX.scale_
    out  = model.fwd(feats)[0] * sY.scale_ + sY.mean_
    return np.array([
        [out[0], out[1], out[2]],
        [out[1], out[3], out[4]],
        [out[2], out[4], out[5]]
    ])

def integrate(clf, kappa, C0, dt, steps):
    traj = np.zeros((steps + 1, 3, 3)); traj[0] = C0; C = C0.copy()
    for s in range(steps):
        Omega = clf(C, kappa)
        dC    = kappa @ C + C @ kappa.T - Omega
        C     = 0.5 * (C + C.T) + dC * dt
        C     = 0.5 * (C + C.T)
        traj[s+1] = C
    return traj

kappa_test = np.zeros((3, 3), dtype=np.float64); kappa_test[0, 1] = 1.0
rng_ref    = np.random.default_rng(777)
r_ref      = rng_ref.normal(0, np.sqrt(P.kT / P.H), (5000, 3))
N_EVAL     = 400
DT_EVAL    = 0.005
C_ref_traj = np.zeros((N_EVAL + 1, 3, 3))
C_ref_traj[0] = conf_tensor(r_ref)
for s in range(N_EVAL):
    r_ref = euler_step(r_ref, kappa_test, DT_EVAL, rng_ref)
    C_ref_traj[s+1] = conf_tensor(r_ref)

C0_eval    = np.eye(3)
C_nn_traj  = integrate(nn_closure,  kappa_test, C0_eval, DT_EVAL, N_EVAL)
C_ucm_traj = integrate(lambda C, k: ucm_closure(C), kappa_test, C0_eval, DT_EVAL, N_EVAL)
C_fp_traj  = integrate(lambda C, k: fenep_closure(C), kappa_test, C0_eval, DT_EVAL, N_EVAL)

time_arr = np.arange(N_EVAL + 1) * DT_EVAL
H        = P.H
tau_ref  = H * C_ref_traj  - np.eye(3)
tau_nn   = H * C_nn_traj   - np.eye(3)
tau_ucm  = H * C_ucm_traj  - np.eye(3)

np.save('results/time.npy',    time_arr)
np.save('results/C_ref.npy',   C_ref_traj)
np.save('results/C_nn.npy',    C_nn_traj)
np.save('results/C_ucm.npy',   C_ucm_traj)
np.save('results/C_fenep.npy', C_fp_traj)
np.save('results/tau_ref.npy', tau_ref)
np.save('results/tau_nn.npy',  tau_nn)
np.save('results/tau_ucm.npy', tau_ucm)


# Helper function to dynamically load extension modules from file path
import importlib.util

def load_ext_module(name, filename):
    filepath = os.path.join(script_dir, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(project_root, 'N2X_project', filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — 3D PHYSICS-INFORMED NEURAL NETWORK (05_pinn_model.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 4: 3D Physics-Informed Neural Network (PINN) …")
print("─"*65)
pinn = load_ext_module('pinn_mod', '05_pinn_model.py')
pinn_res = pinn.train_pinn()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 3D SPECTRAL DECOMPOSITION & ROTATIONAL EQUIVARIANCE (06_spectral_decomposition.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 5: 3D Spectral Decomposition & Rotational Equivariance …")
print("─"*65)
spec = load_ext_module('spec_mod', '06_spectral_decomposition.py')
spec_res = spec.verify_3d_rotational_equivariance(n_tests=50)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — 3D TENSOR BASIS NEURAL NETWORK (07_tbnn_model.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 6: 3D Tensor Basis Neural Network (TBNN) …")
print("─"*65)
tbnn = load_ext_module('tbnn_mod', '07_tbnn_model.py')
tbnn_res = tbnn.train_tbnn()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MULTI-SCALE 3D VISCOELASTIC CFD SOLVER (08_multiscale_cfd.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 7: Multi-Scale 3D Viscoelastic CFD Solver Integration …")
print("─"*65)
cfd = load_ext_module('cfd_mod', '08_multiscale_cfd.py')
cfd_res = cfd.run_multiscale_3d_cfd(Ny=25, Nz=25, n_steps=300, dt=0.002)



# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — VISUALIZATION SUITE GENERATION (FIGS 1 - 13)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  STEP 8: Generating Complete Publication Plot Suite (Figs 1 - 13) …")
print("─"*65)
viz_path = os.path.join(script_dir, '04_visualization.py')
subprocess.run([sys.executable, viz_path], check=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════
t_master_total = time.time() - t_start_master

print("\n" + "="*75)
print("  FINAL 3D PROJECT MASTER PIPELINE SUMMARY")
print("="*75)
print(f"  Dataset Size                  : {X.shape[0]:,} samples (3D Brownian Dynamics)")
print(f"  Base Model R² Score           : {r2:.4f}")
print(f"  5.1 PINN Model Test R²         : {pinn_res['r2']:.4f}")
print(f"  5.2 Spectral Rotational Error  : {spec_res['spectral_mean_rel_error']:.2e} (Exact Frame Invariance ✓)")
print(f"  5.3 TBNN Model Test R²         : {tbnn_res['r2']:.4f}")
print(f"  5.4 CFD Peak Velocity          : {cfd_res['u_max_n2x']:.4f}")
print(f"  Plot Suite Generated          : 13 Publication Figures (results/plots/fig1..fig13.png)")
print(f"  Total Master Execution Time   : {t_master_total:.1f} s")
print("="*75)
print("  N2X 3D MASTER PIPELINE COMPLETE ✓")

print("  N2X 3D MASTER PIPELINE COMPLETE ✓")
