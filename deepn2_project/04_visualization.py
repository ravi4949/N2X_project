"""
================================================================================
N2X Model — Task 10 & 11: Visualization & Plotting
================================================================================
Generates all publication-quality plots:
    Fig 1 — Training loss curve (train vs validation)
    Fig 2 — Parity plots (predicted vs true closure components)
    Fig 3 — Tensor trace  tr(C)  over time: NN vs UCM vs FENE-P vs Reference
    Fig 4 — Stress component  τ_xy  over time
    Fig 5 — Stress component  τ_xx  over time
    Fig 6 — Conformation tensor component  C_xx  trajectory
    Fig 7 — Error histogram
    Fig 8 — Relative Frobenius error distribution
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import matplotlib
matplotlib.use('Agg')       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import json, os

os.makedirs('results/plots', exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'     : 'DejaVu Serif',
    'font.size'       : 11,
    'axes.labelsize'  : 12,
    'axes.titlesize'  : 13,
    'axes.spines.top' : False,
    'axes.spines.right':False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'lines.linewidth' : 2.0,
    'legend.framealpha':0.8,
    'figure.dpi'      : 130,
    'savefig.dpi'     : 150,
    'savefig.bbox'    : 'tight',
})

COLORS = {
    'nn'    : '#2196F3',   # blue
    'ucm'   : '#F44336',   # red
    'fenep' : '#4CAF50',   # green
    'ref'   : '#212121',   # near-black
    'train' : '#9C27B0',   # purple
    'val'   : '#FF9800',   # orange
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def load_results():
    r = {}
    try:
        with open('results/history.json') as f:
            r['history'] = json.load(f)
    except FileNotFoundError:
        r['history'] = None

    for name in ['Y_pred','Y_true','X_test','time',
                 'C_ref','C_nn','C_ucm','C_fenep',
                 'tau_ref','tau_nn','tau_ucm']:
        path = f'results/{name}.npy'
        if os.path.exists(path):
            r[name] = np.load(path)
        else:
            r[name] = None
    return r


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Training / Validation Loss
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_loss(history, save_path):
    if history is None:
        print("  [skip] No history found.")
        return

    train_loss = history['train_loss']
    val_loss   = history['val_loss']
    epochs     = np.arange(1, len(train_loss)+1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Linear scale
    ax = axes[0]
    ax.plot(epochs, train_loss, color=COLORS['train'], label='Train MSE', lw=2)
    ax.plot(epochs, val_loss,   color=COLORS['val'],   label='Val MSE',   lw=2, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Training Curve (linear scale)')
    ax.legend()
    # mark minimum
    best = int(np.argmin(val_loss))
    ax.axvline(best+1, color='gray', ls=':', lw=1)
    ax.annotate(f'Best epoch\n{best+1}', xy=(best+1, val_loss[best]),
                xytext=(best+1+3, val_loss[best]*1.5),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Log scale
    ax = axes[1]
    ax.semilogy(epochs, train_loss, color=COLORS['train'], label='Train MSE', lw=2)
    ax.semilogy(epochs, val_loss,   color=COLORS['val'],   label='Val MSE',   lw=2, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE  (log scale)')
    ax.set_title('Training Curve (log scale)')
    ax.legend()

    fig.suptitle('N2X — Training Loss History', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Parity Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_parity(Y_true, Y_pred, save_path):
    if Y_true is None:
        return

    labels = [r'$\Omega_{xx}$ (closure)', r'$\Omega_{xy}$ (closure)', r'$\Omega_{yy}$ (closure)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for i, (ax, lab) in enumerate(zip(axes, labels)):
        yt = Y_true[:, i]
        yp = Y_pred[:, i]
        vmin = min(yt.min(), yp.min())
        vmax = max(yt.max(), yp.max())

        ax.scatter(yt, yp, alpha=0.15, s=3, color=COLORS['nn'], rasterized=True)
        ax.plot([vmin,vmax],[vmin,vmax],'k--', lw=1.5, label='Perfect fit')

        r2 = float(1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2))
        ax.text(0.05, 0.92, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=10, color='navy')
        ax.set_xlabel(f'True {lab}')
        ax.set_ylabel(f'Predicted {lab}')
        ax.set_title(f'Parity — {lab}')
        ax.legend(fontsize=9)

    fig.suptitle('N2X — Closure Parity Plots', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Tensor Trace tr(C) over time
# ─────────────────────────────────────────────────────────────────────────────
def plot_tensor_trace(time, C_ref, C_nn, C_ucm, C_fenep, save_path):
    if time is None or C_ref is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    def tr(C): return np.trace(C, axis1=1, axis2=2) if C.ndim==3 else None

    ax.plot(time, tr(C_ref),   color=COLORS['ref'],   lw=2.5, label='Reference (BD)')
    ax.plot(time, tr(C_nn),    color=COLORS['nn'],    lw=2,   ls='-',  label='N2X NN')
    ax.plot(time, tr(C_ucm),   color=COLORS['ucm'],   lw=2,   ls='--', label='UCM (analytical)')
    if C_fenep is not None:
        ax.plot(time, tr(C_fenep), color=COLORS['fenep'], lw=2, ls=':', label='FENE-P')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'tr$(C) = C_{xx} + C_{yy}$')
    ax.set_title('Conformation Tensor Trace  tr(C)  vs Time\n(Shear flow,  $\\dot\\gamma=1$)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Stress evolution  τ_xy
# ─────────────────────────────────────────────────────────────────────────────
def plot_stress_xy(time, tau_ref, tau_nn, tau_ucm, save_path):
    if time is None or tau_ref is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, tau_ref[:,0,1], color=COLORS['ref'], lw=2.5, label='Reference (BD)')
    ax.plot(time, tau_nn[:,0,1],  color=COLORS['nn'],  lw=2,   ls='-',  label='N2X NN')
    ax.plot(time, tau_ucm[:,0,1], color=COLORS['ucm'], lw=2,   ls='--', label='UCM')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'Shear stress  $\tau_{xy}$')
    ax.set_title(r'Polymer Shear Stress $\tau_{xy}$ vs Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Normal stress τ_xx
# ─────────────────────────────────────────────────────────────────────────────
def plot_stress_xx(time, tau_ref, tau_nn, tau_ucm, save_path):
    if time is None or tau_ref is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, tau_ref[:,0,0], color=COLORS['ref'], lw=2.5, label='Reference (BD)')
    ax.plot(time, tau_nn[:,0,0],  color=COLORS['nn'],  lw=2,   ls='-',  label='N2X NN')
    ax.plot(time, tau_ucm[:,0,0], color=COLORS['ucm'], lw=2,   ls='--', label='UCM')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'Normal stress  $\tau_{xx}$')
    ax.set_title(r'Polymer Normal Stress $\tau_{xx}$ vs Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Conformation component C_xx
# ─────────────────────────────────────────────────────────────────────────────
def plot_Cxx(time, C_ref, C_nn, C_ucm, C_fenep, save_path):
    if time is None or C_ref is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, C_ref[:,0,0],   color=COLORS['ref'],   lw=2.5, label='Reference (BD)')
    ax.plot(time, C_nn[:,0,0],    color=COLORS['nn'],    lw=2,   ls='-',  label='N2X NN')
    ax.plot(time, C_ucm[:,0,0],   color=COLORS['ucm'],   lw=2,   ls='--', label='UCM')
    if C_fenep is not None:
        ax.plot(time, C_fenep[:,0,0], color=COLORS['fenep'], lw=2, ls=':', label='FENE-P')
    ax.set_xlabel('Time');  ax.set_ylabel(r'$C_{xx}$')
    ax.set_title(r'Conformation Tensor Component $C_{xx}$ vs Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Error histogram
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_histogram(Y_true, Y_pred, save_path):
    if Y_true is None:
        return
    errors = (Y_pred - Y_true).flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=80, color=COLORS['nn'], alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.set_xlabel('Prediction Error  (predicted − true)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution — N2X Closure Predictions')
    mu, sigma = errors.mean(), errors.std()
    ax.text(0.72, 0.90, f'μ = {mu:.4f}\nσ = {sigma:.4f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Relative Frobenius error distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_frob_dist(Y_true, Y_pred, save_path):
    if Y_true is None:
        return
    frob_err = np.linalg.norm(Y_pred-Y_true, axis=1) / (np.linalg.norm(Y_true, axis=1)+1e-12)
    fig, ax  = plt.subplots(figsize=(8, 4))
    ax.hist(frob_err*100, bins=60, color=COLORS['fenep'], alpha=0.8, edgecolor='white')
    ax.set_xlabel('Relative Frobenius Error  (%)')
    ax.set_ylabel('Count')
    ax.set_title('Relative Frobenius Error Distribution')
    med = np.median(frob_err)*100
    ax.axvline(med, color='red', lw=1.5, ls='--', label=f'Median = {med:.2f}%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Summary Dashboard (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dashboard(data, save_path):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Panel A: training loss ---
    if data['history']:
        ax = fig.add_subplot(gs[0, 0])
        tl = data['history']['train_loss']
        vl = data['history']['val_loss']
        ep = np.arange(1, len(tl)+1)
        ax.semilogy(ep, tl, color=COLORS['train'], lw=1.8, label='Train')
        ax.semilogy(ep, vl, color=COLORS['val'],   lw=1.8, ls='--', label='Val')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
        ax.set_title('A — Training Loss'); ax.legend(fontsize=9)

    # --- Panel B: parity Ω_xx ---
    if data['Y_true'] is not None:
        ax = fig.add_subplot(gs[0, 1])
        yt,yp = data['Y_true'][:,0], data['Y_pred'][:,0]
        lim   = [min(yt.min(),yp.min()), max(yt.max(),yp.max())]
        ax.scatter(yt, yp, s=2, alpha=0.15, color=COLORS['nn'], rasterized=True)
        ax.plot(lim,lim,'k--',lw=1.2)
        r2 = 1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
        ax.text(0.05,0.90,f'R²={r2:.3f}', transform=ax.transAxes, fontsize=10)
        ax.set_xlabel(r'True $\Omega_{xx}$'); ax.set_ylabel(r'Pred $\Omega_{xx}$')
        ax.set_title(r'B — Parity $\Omega_{xx}$')

    # --- Panel C: error histogram ---
    if data['Y_true'] is not None:
        ax = fig.add_subplot(gs[0, 2])
        err= (data['Y_pred']-data['Y_true']).flatten()
        ax.hist(err, bins=60, color=COLORS['nn'], alpha=0.8, edgecolor='white')
        ax.axvline(0, color='black', lw=1.2, ls='--')
        ax.set_xlabel('Error'); ax.set_ylabel('Count')
        ax.set_title('C — Error Distribution')

    # --- Panel D: tr(C) ---
    if data['time'] is not None and data['C_ref'] is not None:
        ax  = fig.add_subplot(gs[1, :2])
        def tr(C): return np.trace(C,axis1=1,axis2=2)
        ax.plot(data['time'], tr(data['C_ref']), color=COLORS['ref'],   lw=2.5, label='BD Reference')
        ax.plot(data['time'], tr(data['C_nn']),  color=COLORS['nn'],    lw=2,   label='N2X NN')
        ax.plot(data['time'], tr(data['C_ucm']), color=COLORS['ucm'],   lw=2, ls='--', label='UCM')
        if data['C_fenep'] is not None:
            ax.plot(data['time'], tr(data['C_fenep']), color=COLORS['fenep'], lw=2, ls=':', label='FENE-P')
        ax.set_xlabel('Time'); ax.set_ylabel('tr(C)')
        ax.set_title(r'D — Tensor Trace tr(C)  (Shear flow $\dot\gamma=1$)')
        ax.legend(fontsize=9)

    # --- Panel E: τ_xy ---
    if data['tau_ref'] is not None:
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(data['time'], data['tau_ref'][:,0,1], color=COLORS['ref'], lw=2.5, label='BD Ref')
        ax.plot(data['time'], data['tau_nn'][:,0,1],  color=COLORS['nn'],  lw=2,   label='NN')
        ax.plot(data['time'], data['tau_ucm'][:,0,1], color=COLORS['ucm'], lw=2, ls='--', label='UCM')
        ax.set_xlabel('Time'); ax.set_ylabel(r'$\tau_{xy}$')
        ax.set_title(r'E — Shear Stress $\tau_{xy}$')
        ax.legend(fontsize=9)

    fig.suptitle('N2X — Summary Dashboard', fontsize=15, fontweight='bold')
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading results …")
    data = load_results()

    print("\nGenerating plots …")

    plot_training_loss(data['history'],
        'results/plots/fig1_training_loss.png')

    plot_parity(data['Y_true'], data['Y_pred'],
        'results/plots/fig2_parity_plots.png')

    plot_tensor_trace(data['time'], data['C_ref'], data['C_nn'],
                       data['C_ucm'], data['C_fenep'],
        'results/plots/fig3_tensor_trace.png')

    plot_stress_xy(data['time'], data['tau_ref'], data['tau_nn'], data['tau_ucm'],
        'results/plots/fig4_stress_xy.png')

    plot_stress_xx(data['time'], data['tau_ref'], data['tau_nn'], data['tau_ucm'],
        'results/plots/fig5_stress_xx.png')

    plot_Cxx(data['time'], data['C_ref'], data['C_nn'],
              data['C_ucm'], data['C_fenep'],
        'results/plots/fig6_Cxx.png')

    plot_error_histogram(data['Y_true'], data['Y_pred'],
        'results/plots/fig7_error_histogram.png')

    plot_frob_dist(data['Y_true'], data['Y_pred'],
        'results/plots/fig8_frobenius_dist.png')

    plot_dashboard(data,
        'results/plots/fig9_dashboard.png')

    print("\nAll plots generated ✓")
