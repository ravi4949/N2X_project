"""
================================================================================
N2X Model — Task 5, 6, 7: Neural Network Definition + Training
================================================================================
Architecture mirrors the N2X paper:
    Input  (7)  →  Dense(64, tanh)  →  Dense(64, tanh)  →  Dense(32, tanh)
                →  Output(3)

Input features  : [I1, I2, I3, κxx, κxy, κyx, κyy]
Output (3 vals) : unique entries of symmetric 2×2 closure tensor
                  [Ω_xx, Ω_xy, Ω_yy]

Loss function   : MSE  +  λ_sym · symmetry_penalty
                  (symmetry is built-in here, but we add a penalty layer
                   that can be used when the output is generalised to 4 entries)
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import os, time, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight NumPy-only neural network (no PyTorch needed)
# ─────────────────────────────────────────────────────────────────────────────

class Layer:
    """Single fully-connected layer with activation."""
    def __init__(self, n_in: int, n_out: int, activation: str = 'tanh',
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(0)
        # Xavier / Glorot initialisation
        scale     = np.sqrt(2.0 / (n_in + n_out))
        self.W    = rng.normal(0, scale, (n_in, n_out)).astype(np.float64)
        self.b    = np.zeros(n_out, dtype=np.float64)
        self.act  = activation
        # Adam state
        self.mW = np.zeros_like(self.W);  self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b);  self.vb = np.zeros_like(self.b)

    # --- forward ---
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        self._z     = x @ self.W + self.b
        if self.act == 'tanh':
            self._a = np.tanh(self._z)
        elif self.act == 'relu':
            self._a = np.maximum(0, self._z)
        else:   # linear
            self._a = self._z
        return self._a

    # --- backward ---
    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        if self.act == 'tanh':
            d_act = 1.0 - self._a**2
        elif self.act == 'relu':
            d_act = (self._z > 0).astype(float)
        else:
            d_act = np.ones_like(self._z)
        dz = grad_a * d_act                  # element-wise
        self.dW = self._input.T @ dz / len(self._input)
        self.db = dz.mean(axis=0)
        return dz @ self.W.T                 # gradient w.r.t. input


class N2XNet:
    """
    N2X neural network for closure model prediction.
    Architecture:
        7  → 64 (tanh) → 64 (tanh) → 32 (tanh) → 3 (linear)
    """
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = [
            Layer( 7, 64, 'tanh', rng),
            Layer(64, 64, 'tanh', rng),
            Layer(64, 32, 'tanh', rng),
            Layer(32,  3, 'none', rng),
        ]
        self.t_adam = 0   # Adam step counter

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return [(l.W, l.b, l.mW, l.vW, l.mb, l.vb) for l in self.layers]

    def adam_step(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t_adam += 1
        t = self.t_adam
        for layer in self.layers:
            for (param, grad, m, v) in [(layer.W, layer.dW, layer.mW, layer.vW),
                                         (layer.b, layer.db, layer.mb, layer.vb)]:
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: str):
        weights = {f'layer_{i}_W': l.W for i, l in enumerate(self.layers)}
        weights.update({f'layer_{i}_b': l.b for i, l in enumerate(self.layers)})
        np.savez(path, **weights)
        print(f"Model saved to {path}.npz")

    def load(self, path: str):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.W = data[f'layer_{i}_W']
            layer.b = data[f'layer_{i}_b']
        print(f"Model loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target)**2))

def symmetry_penalty(pred: np.ndarray) -> float:
    """
    For a 2D closure tensor the output is [Ω_xx, Ω_xy, Ω_yy].
    Symmetry is already enforced by having only 3 outputs (Ω_xy == Ω_yx).
    This placeholder penalty would apply if we had 4 outputs [xx, xy, yx, yy]:
        penalty = mean((pred[:, 1] - pred[:, 2])^2)
    With 3-output network, symmetry_penalty = 0 by construction.
    """
    return 0.0

def total_loss(pred: np.ndarray, target: np.ndarray,
               lambda_sym: float = 0.01) -> tuple:
    mse  = mse_loss(pred, target)
    sym  = symmetry_penalty(pred)
    total = mse + lambda_sym * sym
    return total, mse, sym

def loss_gradient(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. predictions."""
    return 2.0 * (pred - target) / pred.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(model: N2XNet,
          X_train: np.ndarray, Y_train: np.ndarray,
          X_val:   np.ndarray, Y_val:   np.ndarray,
          n_epochs: int    = 200,
          batch_size: int  = 512,
          lr: float        = 1e-3,
          lr_decay: float  = 0.98,
          lambda_sym: float= 0.01,
          patience: int    = 20) -> dict:
    """
    Mini-batch Adam training with learning-rate decay and early stopping.
    
    Returns
    -------
    history : dict with 'train_loss', 'val_loss', 'train_mse', 'val_mse'
    """
    n_samples = X_train.shape[0]
    history   = {'train_loss': [], 'val_loss': [],
                 'train_mse':  [], 'val_mse':  [], 'lr': []}

    best_val  = np.inf
    best_weights = None
    wait      = 0

    print(f"\n{'='*60}")
    print(f"  N2X Training — {n_epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}")
    print(f"  {'-'*50}")

    rng = np.random.default_rng(99)

    for epoch in range(n_epochs):
        # Shuffle
        idx = rng.permutation(n_samples)
        epoch_loss = []

        for start in range(0, n_samples, batch_size):
            end    = min(start + batch_size, n_samples)
            bi     = idx[start:end]
            Xb, Yb = X_train[bi], Y_train[bi]

            # Forward
            pred = model.forward(Xb)
            loss, mse, sym = total_loss(pred, Yb, lambda_sym)

            # Backward
            grad = loss_gradient(pred, Yb)
            model.backward(grad)
            model.adam_step(lr)
            epoch_loss.append(mse)

        # Validation
        val_pred          = model.predict(X_val)
        val_loss, val_mse, _ = total_loss(val_pred, Y_val, lambda_sym)
        train_mse_epoch   = float(np.mean(epoch_loss))

        # Learning-rate decay every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr *= lr_decay

        history['train_loss'].append(train_mse_epoch)
        history['val_loss'].append(val_mse)
        history['train_mse'].append(train_mse_epoch)
        history['val_mse'].append(val_mse)
        history['lr'].append(lr)

        # Early stopping
        if val_mse < best_val:
            best_val    = val_mse
            best_epoch  = epoch
            # save best weights
            best_weights = [(l.W.copy(), l.b.copy()) for l in model.layers]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}  "
                      f"(best val MSE = {best_val:.6f} at epoch {best_epoch+1})")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:>6}  {train_mse_epoch:>10.6f}  "
                  f"{val_mse:>10.6f}  {lr:>10.2e}")

    # Restore best weights
    for i, (W, b) in enumerate(best_weights):
        model.layers[i].W = W
        model.layers[i].b = b
    print(f"\n  Best val MSE = {best_val:.6f} (restored from epoch {best_epoch+1})")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset …")
    X = np.load('data/X.npy').astype(np.float64)
    Y = np.load('data/Y.npy').astype(np.float64)
    print(f"  X: {X.shape}   Y: {Y.shape}")

    # ── Normalise features ────────────────────────────────────────────────────
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled  = scaler_X.fit_transform(X)
    Y_scaled  = scaler_Y.fit_transform(Y)

    # Save scalers (mean / std) for inference
    np.save('models/scaler_X_mean.npy', scaler_X.mean_)
    np.save('models/scaler_X_std.npy',  scaler_X.scale_)
    np.save('models/scaler_Y_mean.npy', scaler_Y.mean_)
    np.save('models/scaler_Y_std.npy',  scaler_Y.scale_)

    # ── Train / val / test split ──────────────────────────────────────────────
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(
        X_scaled, Y_scaled, test_size=0.20, random_state=7)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.50, random_state=7)

    print(f"  Train: {X_tr.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # ── Build & train model ───────────────────────────────────────────────────
    model   = N2XNet(seed=42)
    t0      = time.time()
    history = train(model, X_tr, Y_tr, X_val, Y_val,
                    n_epochs   = 300,
                    batch_size = 512,
                    lr         = 5e-4,
                    lr_decay   = 0.97,
                    lambda_sym = 0.005,
                    patience   = 30)
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f} s")

    # ── Save artefacts ────────────────────────────────────────────────────────
    model.save('models/N2X_model')
    np.save('results/history.npy', history)

    # ── Quick test-set evaluation ─────────────────────────────────────────────
    Y_pred_scaled = model.predict(X_test)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_true = scaler_Y.inverse_transform(Y_test)

    test_mse = mse_loss(Y_pred, Y_true)
    ss_res   = np.sum((Y_true - Y_pred)**2)
    ss_tot   = np.sum((Y_true - Y_true.mean(0))**2)
    r2       = 1 - ss_res / ss_tot
    frob_rel = np.mean(np.linalg.norm(Y_true - Y_pred, axis=1) /
                       (np.linalg.norm(Y_true, axis=1) + 1e-12))

    print(f"\n{'='*40}")
    print(f"  TEST SET RESULTS")
    print(f"  MSE              : {test_mse:.6f}")
    print(f"  R²               : {r2:.6f}")
    print(f"  Rel. Frobenius   : {frob_rel:.4f} ({frob_rel*100:.2f}%)")
    print(f"{'='*40}")

    # Save test predictions for evaluation script
    np.save('results/Y_pred.npy', Y_pred)
    np.save('results/Y_true.npy', Y_true)
    np.save('results/X_test.npy', scaler_X.inverse_transform(X_test))

    # Save history as JSON
    history_serializable = {k: [float(v) for v in vals]
                             for k, vals in history.items()}
    with open('results/history.json', 'w') as f:
        json.dump(history_serializable, f, indent=2)

    print("\nTraining complete ✓")
