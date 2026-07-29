"""
================================================================================
N2X Model : 3D Physics-Informed Neural Network (05_pinn_model.py)
================================================================================
Implements physical soft constraints in the loss function:
    L_total = L_MSE + λ_SPD * L_SPD + λ_trace * L_trace

    • L_SPD   : Penalty if predicted next conformation tensor C_next has 
                negative eigenvalues (C ⊁ 0, violated positive definiteness)
    • L_trace : Penalty if tr(C_next) > b (polymer chain max extension limit)

Saves trained model to models/N2X_pinn_model.npz
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os, time, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DumbbellParams:
    H = 1.0; b = 50.0; kT = 1.0; zeta = 1.0
    tau = zeta / (4 * H); D = kT / zeta

class PINNLayer:
    """Fully-connected layer with analytical backprop."""
    def __init__(self, n_in: int, n_out: int, activation: str = 'tanh', rng = None):
        if rng is None: rng = np.random.default_rng(0)
        scale  = np.sqrt(2.0 / (n_in + n_out))
        self.W = rng.normal(0, scale, (n_in, n_out)).astype(np.float64)
        self.b = np.zeros(n_out, dtype=np.float64)
        self.act = activation
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        self._a = np.tanh(self._z) if self.act == 'tanh' else self._z
        return self._a

    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        d_act = (1.0 - self._a**2) if self.act == 'tanh' else 1.0
        dz = grad_a * d_act
        self.dW = self._x.T @ dz / len(self._x)
        self.db = dz.mean(axis=0)
        return dz @ self.W.T


class PINN_N2XNet:
    """3D PINN Neural Network for closure prediction."""
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.layers = [
            PINNLayer(13, 128, 'tanh', rng),
            PINNLayer(128, 128, 'tanh', rng),
            PINNLayer(128, 64, 'tanh', rng),
            PINNLayer(64,   6, 'none', rng),
        ]
        self.t_adam = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def adam_step(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t_adam += 1
        t = self.t_adam
        for layer in self.layers:
            for (param, g, m, v) in [(layer.W, layer.dW, layer.mW, layer.vW),
                                     (layer.b, layer.db, layer.mb, layer.vb)]:
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * g**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: str):
        weights = {f'layer_{i}_W': l.W for i, l in enumerate(self.layers)}
        weights.update({f'layer_{i}_b': l.b for i, l in enumerate(self.layers)})
        np.savez(path, **weights)
        print(f"PINN Model saved to {path}")


def compute_pinn_loss_and_grad_vectorized(model: PINN_N2XNet,
                                          Xb: np.ndarray,
                                          Yb: np.ndarray,
                                          C_now_b: np.ndarray,
                                          kappa_b: np.ndarray,
                                          scaler_Y_mean: np.ndarray,
                                          scaler_Y_std: np.ndarray,
                                          dt: float = 0.01,
                                          lambda_spd: float = 0.1,
                                          lambda_trace: float = 0.05,
                                          b_max: float = 50.0):
    B = len(Xb)
    Y_pred_s = model.forward(Xb)
    Y_pred_phys = Y_pred_s * scaler_Y_std + scaler_Y_mean

    res_s = Y_pred_s - Yb
    mse_loss = float(np.mean(res_s**2))
    grad_s = 2.0 * res_s / B

    # Vectorized 3D Tensor Reconstruction
    Omega = np.zeros((B, 3, 3), dtype=np.float64)
    Omega[:, 0, 0] = Y_pred_phys[:, 0]
    Omega[:, 0, 1] = Y_pred_phys[:, 1]; Omega[:, 1, 0] = Y_pred_phys[:, 1]
    Omega[:, 0, 2] = Y_pred_phys[:, 2]; Omega[:, 2, 0] = Y_pred_phys[:, 2]
    Omega[:, 1, 1] = Y_pred_phys[:, 3]
    Omega[:, 1, 2] = Y_pred_phys[:, 4]; Omega[:, 2, 1] = Y_pred_phys[:, 4]
    Omega[:, 2, 2] = Y_pred_phys[:, 5]

    C_grid = C_now_b.reshape(B, 3, 3)
    k_grid = kappa_b.reshape(B, 3, 3)

    dC_dt  = np.matmul(k_grid, C_grid) + np.matmul(C_grid, np.swapaxes(k_grid, 1, 2)) - Omega
    C_next = C_grid + dt * dC_dt
    C_next = 0.5 * (C_next + np.swapaxes(C_next, 1, 2))

    # Batch Eigenvalues for SPD
    eigvals = np.linalg.eigvalsh(C_next)
    min_eig = np.min(eigvals, axis=1)

    spd_mask = min_eig < 0
    spd_loss = float(np.mean(np.where(spd_mask, (-min_eig)**2, 0.0)))

    # Trace penalty
    trC = np.trace(C_next, axis1=1, axis2=2)
    tr_mask = trC > b_max
    tr_loss = float(np.mean(np.where(tr_mask, (trC - b_max)**2, 0.0)))

    # Vectorized Physics Gradients
    phys_grads = np.zeros_like(Y_pred_phys)

    if np.any(spd_mask):
        g_spd = np.where(spd_mask, 2.0 * min_eig * dt, 0.0)
        phys_grads[:, 0] += lambda_spd * g_spd
        phys_grads[:, 1] += lambda_spd * g_spd * 2.0
        phys_grads[:, 2] += lambda_spd * g_spd * 2.0
        phys_grads[:, 3] += lambda_spd * g_spd
        phys_grads[:, 4] += lambda_spd * g_spd * 2.0
        phys_grads[:, 5] += lambda_spd * g_spd

    if np.any(tr_mask):
        g_tr = np.where(tr_mask, 2.0 * (trC - b_max) * (-dt), 0.0)
        phys_grads[:, 0] += lambda_trace * g_tr
        phys_grads[:, 3] += lambda_trace * g_tr
        phys_grads[:, 5] += lambda_trace * g_tr

    total_grad_s = grad_s + (phys_grads / B) * scaler_Y_std
    model.backward(total_grad_s)

    total_loss = mse_loss + lambda_spd * spd_loss + lambda_trace * tr_loss
    return total_loss, mse_loss, spd_loss, tr_loss


def train_pinn():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60)
    print("  3D PINN (Physics-Informed Neural Network) Training")
    print("="*60)

    X = np.load('data/X.npy').astype(np.float64)
    Y = np.load('data/Y.npy').astype(np.float64)
    C_now = np.load('data/C_now.npy').astype(np.float64)
    kappa = np.load('data/kappa.npy').astype(np.float64)

    sX = StandardScaler(); sY = StandardScaler()
    Xs = sX.fit_transform(X)
    Ys = sY.fit_transform(Y)

    # Train/Val/Test Split
    idx_tr, idx_tmp = train_test_split(np.arange(len(X)), test_size=0.20, random_state=42)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    X_tr, Y_tr = Xs[idx_tr], Ys[idx_tr]
    X_val, Y_val = Xs[idx_val], Ys[idx_val]
    X_te, Y_te = Xs[idx_te], Ys[idx_te]

    C_now_tr, kappa_tr = C_now[idx_tr], kappa[idx_tr]

    model = PINN_N2XNet(seed=42)
    
    n_epochs   = 200
    batch_size = 256
    lr         = 8e-4
    
    history = {'total_loss': [], 'mse_loss': [], 'spd_loss': [], 'val_mse': []}
    
    t0 = time.time()
    rng = np.random.default_rng(99)
    n_samples = len(X_tr)

    print(f"  {'Epoch':>6}  {'Total Loss':>12}  {'MSE Loss':>10}  {'SPD Penalty':>12}  {'Val MSE':>10}")
    print(f"  {'-'*60}")

    for epoch in range(n_epochs):
        perm = rng.permutation(n_samples)
        ep_tot, ep_mse, ep_spd = [], [], []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bi  = perm[start:end]
            Xb, Yb = X_tr[bi], Y_tr[bi]
            Cb, kb = C_now_tr[bi], kappa_tr[bi]

            tot_l, mse_l, spd_l, tr_l = compute_pinn_loss_and_grad_vectorized(
                model, Xb, Yb, Cb, kb, sY.mean_, sY.scale_
            )
            model.adam_step(lr)

            ep_tot.append(tot_l); ep_mse.append(mse_l); ep_spd.append(spd_l)

        if (epoch + 1) % 10 == 0: lr *= 0.97

        val_pred = model.forward(X_val)
        val_mse  = float(np.mean((val_pred - Y_val)**2))
        
        tr_tot = float(np.mean(ep_tot))
        tr_mse = float(np.mean(ep_mse))
        tr_spd = float(np.mean(ep_spd))

        history['total_loss'].append(tr_tot)
        history['mse_loss'].append(tr_mse)
        history['spd_loss'].append(tr_spd)
        history['val_mse'].append(val_mse)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  {epoch+1:>6}  {tr_tot:>12.6f}  {tr_mse:>10.6f}  {tr_spd:>12.6e}  {val_mse:>10.6f}")

    elapsed = time.time() - t0
    print(f"\n  PINN Training complete in {elapsed:.1f} s")
    model.save('models/N2X_pinn_model')

    # Evaluate PINN Test Set
    Y_pred_s = model.forward(X_te)
    Y_pred = sY.inverse_transform(Y_pred_s)
    Y_true = sY.inverse_transform(Y_te)

    test_mse = float(np.mean((Y_pred - Y_true)**2))
    r2 = float(1 - np.sum((Y_true - Y_pred)**2) / np.sum((Y_true - Y_true.mean(0))**2))
    frob_rel = float(np.mean(np.linalg.norm(Y_true - Y_pred, axis=1) / (np.linalg.norm(Y_true, axis=1) + 1e-12)))

    print(f"\n  PINN 3D Test Set Results:")
    print(f"    MSE              : {test_mse:.6f}")
    print(f"    R²               : {r2:.6f}")
    print(f"    Rel. Frob. Error : {frob_rel*100:.3f}%")

    results = {
        'test_mse': test_mse,
        'r2': r2,
        'frob_rel_pct': frob_rel * 100,
        'final_spd_penalty': history['spd_loss'][-1],
        'history': history
    }
    with open('results/pinn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    train_pinn()
