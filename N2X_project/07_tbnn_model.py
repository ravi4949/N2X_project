"""
================================================================================
N2X Model : 3D Tensor Basis Neural Network (07_tbnn_model.py)
================================================================================
Implements isotropic 3D tensor basis expansion (Pope 1975 formulation):
    Ω = Σₖ αₖ(I₁, I₂, I₃, I₄) · Tₖ(C, S, W)

    where:
    • S = 0.5 · (κ + κᵀ)   (Symmetric strain rate tensor)
    • W = 0.5 · (κ − κᵀ)   (Anti-symmetric vorticity tensor)
    • T₁ = I₃               (Identity basis)
    • T₂ = C                (Conformation tensor basis)
    • T₃ = C²               (Square conformation basis)
    • T₄ = S                (Strain-rate basis)
    • T₅ = S·C + C·S        (Strain-conformation coupling)
    • T₆ = W·C − C·W        (Vorticity-conformation coupling)

    • α₁…α₆ : Scalar coefficients predicted by a neural network operating 
              EXCLUSIVELY on scalar invariants (I₁, I₂, I₃, I₄).

This architecture mathematically guarantees EXACT material frame indifference 
(SO(3) rotational equivariance) by construction.
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os, time, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compute_3d_tensor_bases(C: np.ndarray, kappa: np.ndarray):
    """
    Computes 6 symmetric 3D tensor bases T₁…T₆ for sample (C, κ).
    Returns shape (6, 3, 3).
    """
    I3 = np.eye(3, dtype=np.float64)
    S  = 0.5 * (kappa + kappa.T)
    W  = 0.5 * (kappa - kappa.T)
    C2 = C @ C

    T1 = I3
    T2 = C
    T3 = C2
    T4 = S
    T5 = S @ C + C @ S
    T6 = W @ C - C @ W  # Symmetric since W is anti-symmetric and C is symmetric

    return np.stack([T1, T2, T3, T4, T5, T6], axis=0)


class TBNNLayer:
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


class TBNN_N2XNet:
    """
    TBNN Neural Network: maps 4 scalar invariants -> 6 scalar coefficients α₁…α₆.
    """
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.layers = [
            TBNNLayer(4,  64, 'tanh', rng),
            TBNNLayer(64, 64, 'tanh', rng),
            TBNNLayer(64, 32, 'tanh', rng),
            TBNNLayer(32,  6, 'none', rng),
        ]
        self.t_adam = 0

    def forward(self, x_inv: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x_inv = layer.forward(x_inv)
        return x_inv

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
        print(f"TBNN Model saved to {path}")


def train_tbnn():
    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60)
    print("  3D TBNN (Tensor Basis Neural Network) Training")
    print("="*60)

    X_all = np.load('data/X.npy').astype(np.float64)       # (N, 13) -> first 4 are invariants
    Y_all = np.load('data/Y.npy').astype(np.float64)       # (N, 6)
    C_now = np.load('data/C_now.npy').astype(np.float64)   # (N, 9)
    kappa = np.load('data/kappa.npy').astype(np.float64)   # (N, 9)

    Invariants = X_all[:, :4]  # (N, 4)

    sInv = StandardScaler()
    Inv_scaled = sInv.fit_transform(Invariants)

    np.save('models/tbnn_scaler_Inv_mean.npy', sInv.mean_)
    np.save('models/tbnn_scaler_Inv_std.npy',  sInv.scale_)

    # Split
    idx_tr, idx_tmp = train_test_split(np.arange(len(X_all)), test_size=0.20, random_state=42)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    Inv_tr, Y_tr = Inv_scaled[idx_tr], Y_all[idx_tr]
    Inv_val, Y_val = Inv_scaled[idx_val], Y_all[idx_val]
    Inv_te, Y_te = Inv_scaled[idx_te], Y_all[idx_te]

    C_tr, kappa_tr = C_now[idx_tr], kappa[idx_tr]
    C_val, kappa_val = C_now[idx_val], kappa[idx_val]
    C_te, kappa_te = C_now[idx_te], kappa[idx_te]

    # Precompute Tensor Bases for train, val, test
    print("  Precomputing 3D Tensor Bases (T₁…T₆) for dataset …")
    
    def compute_all_bases(C_arr, k_arr):
        bases = []
        for i in range(len(C_arr)):
            Ci = C_arr[i].reshape(3, 3)
            ki = k_arr[i].reshape(3, 3)
            bases.append(compute_3d_tensor_bases(Ci, ki))
        return np.array(bases)  # shape (N, 6, 3, 3)

    Bases_tr  = compute_all_bases(C_tr, kappa_tr)
    Bases_val = compute_all_bases(C_val, kappa_val)
    Bases_te  = compute_all_bases(C_te, kappa_te)

    model = TBNN_N2XNet(seed=42)

    n_epochs   = 250
    batch_size = 256
    lr         = 1e-3
    history    = {'train_mse': [], 'val_mse': []}

    t0 = time.time()
    rng = np.random.default_rng(99)
    n_samples = len(Inv_tr)

    print(f"\n  {'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}")
    print(f"  {'-'*40}")

    for epoch in range(n_epochs):
        perm = rng.permutation(n_samples)
        batch_mses = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bi  = perm[start:end]
            
            Inv_b  = Inv_tr[bi]         # (B, 4)
            Y_b    = Y_tr[bi]           # (B, 6)
            Bases_b= Bases_tr[bi]       # (B, 6, 3, 3)

            # Forward pass: predict α (B, 6)
            alphas = model.forward(Inv_b)

            # Construct predicted Ω_b: (B, 3, 3) = sum_k α_k * T_k
            Omega_pred = np.einsum('bk,bkij->bij', alphas, Bases_b)

            # Flatten to 6 unique components: (B, 6)
            Y_pred_b = np.stack([
                Omega_pred[:, 0, 0], Omega_pred[:, 0, 1], Omega_pred[:, 0, 2],
                Omega_pred[:, 1, 1], Omega_pred[:, 1, 2], Omega_pred[:, 2, 2]
            ], axis=1)

            res = Y_pred_b - Y_b
            loss = float(np.mean(res**2))
            batch_mses.append(loss)

            # Compute gradient w.r.t. alphas: dL/dα_k = (2/B) sum_{ij} (Y_pred - Y_true)_ij * T_k,ij
            res_tensor = np.zeros((len(bi), 3, 3))
            res_tensor[:, 0, 0] = res[:, 0]
            res_tensor[:, 0, 1] = res[:, 1]; res_tensor[:, 1, 0] = res[:, 1]
            res_tensor[:, 0, 2] = res[:, 2]; res_tensor[:, 2, 0] = res[:, 2]
            res_tensor[:, 1, 1] = res[:, 3]
            res_tensor[:, 1, 2] = res[:, 4]; res_tensor[:, 2, 1] = res[:, 4]
            res_tensor[:, 2, 2] = res[:, 5]

            grad_alpha = 2.0 * np.einsum('bij,bkij->bk', res_tensor, Bases_b) / len(bi)

            model.backward(grad_alpha)
            model.adam_step(lr)

        if (epoch + 1) % 10 == 0: lr *= 0.97

        # Validation MSE
        alphas_val = model.forward(Inv_val)
        Omega_val = np.einsum('bk,bkij->bij', alphas_val, Bases_val)
        Y_pred_val = np.stack([
            Omega_val[:, 0, 0], Omega_val[:, 0, 1], Omega_val[:, 0, 2],
            Omega_val[:, 1, 1], Omega_val[:, 1, 2], Omega_val[:, 2, 2]
        ], axis=1)
        val_mse = float(np.mean((Y_pred_val - Y_val)**2))
        tr_mse  = float(np.mean(batch_mses))

        history['train_mse'].append(tr_mse)
        history['val_mse'].append(val_mse)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  {epoch+1:>6}  {tr_mse:>12.6f}  {val_mse:>12.6f}")

    elapsed = time.time() - t0
    print(f"\n  TBNN Training complete in {elapsed:.1f} s")
    model.save('models/N2X_tbnn_model')

    # Test set evaluation
    alphas_te = model.forward(Inv_te)
    Omega_te  = np.einsum('bk,bkij->bij', alphas_te, Bases_te)
    Y_pred_te = np.stack([
        Omega_te[:, 0, 0], Omega_te[:, 0, 1], Omega_te[:, 0, 2],
        Omega_te[:, 1, 1], Omega_te[:, 1, 2], Omega_te[:, 2, 2]
    ], axis=1)

    test_mse = float(np.mean((Y_pred_te - Y_te)**2))
    r2       = float(1 - np.sum((Y_te - Y_pred_te)**2) / np.sum((Y_te - Y_te.mean(0))**2))
    frob_rel = float(np.mean(np.linalg.norm(Y_te - Y_pred_te, axis=1) / (np.linalg.norm(Y_te, axis=1) + 1e-12)))

    print(f"\n  TBNN 3D Test Set Results:")
    print(f"    MSE              : {test_mse:.6f}")
    print(f"    R²               : {r2:.6f}")
    print(f"    Rel. Frob. Error : {frob_rel*100:.3f}%")

    results = {
        'test_mse': test_mse,
        'r2': r2,
        'frob_rel_pct': frob_rel * 100,
        'history': history
    }
    with open('results/tbnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    train_tbnn()
