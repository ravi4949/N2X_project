"""
================================================================================
N2X Model : Visualization & Plotting (3D)
================================================================================
Generates publication-quality 3D plots:
    Fig 1 — Training loss curve (train vs validation)
    Fig 2 — Parity plots for 6 3D closure components
    Fig 3 — Tensor trace tr(C) = Cxx + Cyy + Czz over time
    Fig 4 — Shear stress τ_xy and τ_xz over time
    Fig 5 — Normal stress τ_xx and τ_zz over time
    Fig 6 — Conformation tensor component C_xx trajectory
    Fig 7 — Error distribution histogram
    Fig 8 — Relative Frobenius error distribution
    Fig 9 — 3D Results Dashboard
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os

os.makedirs('results/plots', exist_ok=True)

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
    'nn'    : '#2196F3',
    'ucm'   : '#F44336',
    'fenep' : '#4CAF50',
    'ref'   : '#212121',
    'train' : '#9C27B0',
    'val'   : '#FF9800',
}


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


def plot_training_loss(history, save_path):
    if history is None:
        return
    train_loss = history['train_loss']
    val_loss   = history['val_loss']
    epochs     = np.arange(1, len(train_loss)+1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(epochs, train_loss, color=COLORS['train'], label='Train MSE', lw=2)
    ax.plot(epochs, val_loss,   color=COLORS['val'],   label='Val MSE',   lw=2, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Training Curve (linear scale)')
    ax.legend()
    best = int(np.argmin(val_loss))
    ax.axvline(best+1, color='gray', ls=':', lw=1)

    ax = axes[1]
    ax.semilogy(epochs, train_loss, color=COLORS['train'], label='Train MSE', lw=2)
    ax.semilogy(epochs, val_loss,   color=COLORS['val'],   label='Val MSE',   lw=2, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Training Curve (log scale)')
    ax.legend()

    fig.suptitle('N2X 3D — Training Loss History', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_parity_3d(Y_true, Y_pred, save_path):
    if Y_true is None:
        return

    labels = [r'$\Omega_{xx}$', r'$\Omega_{xy}$', r'$\Omega_{xz}$',
              r'$\Omega_{yy}$', r'$\Omega_{yz}$', r'$\Omega_{zz}$']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, (ax, lab) in enumerate(zip(axes.flatten(), labels)):
        yt = Y_true[:, i]
        yp = Y_pred[:, i]
        vmin = min(yt.min(), yp.min())
        vmax = max(yt.max(), yp.max())

        ax.scatter(yt, yp, alpha=0.15, s=3, color=COLORS['nn'], rasterized=True)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, label='y=x')

        r2 = float(1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2))
        ax.text(0.05, 0.88, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=10, color='navy')
        ax.set_xlabel(f'True {lab}')
        ax.set_ylabel(f'Predicted {lab}')
        ax.set_title(f'Parity — {lab}')
        ax.legend(fontsize=9)

    fig.suptitle('N2X 3D — Closure Component Parity Plots', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tensor_trace_3d(time, C_ref, C_nn, C_ucm, C_fenep, save_path):
    if time is None or C_ref is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    def tr(C): return np.trace(C, axis1=1, axis2=2)

    ax.plot(time, tr(C_ref),   color=COLORS['ref'],   lw=2.5, label='Reference (3D BD)')
    ax.plot(time, tr(C_nn),    color=COLORS['nn'],    lw=2,   ls='-',  label='N2X NN (3D)')
    ax.plot(time, tr(C_ucm),   color=COLORS['ucm'],   lw=2,   ls='--', label='UCM (3D)')
    if C_fenep is not None:
        ax.plot(time, tr(C_fenep), color=COLORS['fenep'], lw=2, ls=':', label='FENE-P (3D)')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'tr$(C) = C_{xx} + C_{yy} + C_{zz}$')
    ax.set_title('3D Conformation Tensor Trace tr(C) vs Time\n(Shear flow,  $\\dot\\gamma=1$)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stress_xy_3d(time, tau_ref, tau_nn, tau_ucm, save_path):
    if time is None or tau_ref is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, tau_ref[:,0,1], color=COLORS['ref'], lw=2.5, label='Reference (3D BD)')
    ax.plot(time, tau_nn[:,0,1],  color=COLORS['nn'],  lw=2,   ls='-',  label='N2X NN (3D)')
    ax.plot(time, tau_ucm[:,0,1], color=COLORS['ucm'], lw=2,   ls='--', label='UCM')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'Shear stress  $\tau_{xy}$')
    ax.set_title(r'Polymer Shear Stress $\tau_{xy}$ vs Time (3D)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stress_xx_3d(time, tau_ref, tau_nn, tau_ucm, save_path):
    if time is None or tau_ref is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, tau_ref[:,0,0], color=COLORS['ref'], lw=2.5, label='Reference (3D BD)')
    ax.plot(time, tau_nn[:,0,0],  color=COLORS['nn'],  lw=2,   ls='-',  label='N2X NN (3D)')
    ax.plot(time, tau_ucm[:,0,0], color=COLORS['ucm'], lw=2,   ls='--', label='UCM')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'Normal stress  $\tau_{xx}$')
    ax.set_title(r'Polymer Normal Stress $\tau_{xx}$ vs Time (3D)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_Cxx_3d(time, C_ref, C_nn, C_ucm, C_fenep, save_path):
    if time is None or C_ref is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, C_ref[:,0,0],   color=COLORS['ref'],   lw=2.5, label='Reference (3D BD)')
    ax.plot(time, C_nn[:,0,0],    color=COLORS['nn'],    lw=2,   ls='-',  label='N2X NN (3D)')
    ax.plot(time, C_ucm[:,0,0],   color=COLORS['ucm'],   lw=2,   ls='--', label='UCM')
    if C_fenep is not None:
        ax.plot(time, C_fenep[:,0,0], color=COLORS['fenep'], lw=2, ls=':', label='FENE-P')
    ax.set_xlabel('Time');  ax.set_ylabel(r'$C_{xx}$')
    ax.set_title(r'3D Conformation Tensor Component $C_{xx}$ vs Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_error_histogram(Y_true, Y_pred, save_path):
    if Y_true is None:
        return
    errors = (Y_pred - Y_true).flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=80, color=COLORS['nn'], alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.set_xlabel('Prediction Error  (predicted − true)')
    ax.set_ylabel('Count')
    ax.set_title('3D Error Distribution — N2X Closure Predictions')
    mu, sigma = errors.mean(), errors.std()
    ax.text(0.72, 0.90, f'μ = {mu:.4f}\nσ = {sigma:.4f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_frob_dist(Y_true, Y_pred, save_path):
    if Y_true is None:
        return
    frob_err = np.linalg.norm(Y_pred-Y_true, axis=1) / (np.linalg.norm(Y_true, axis=1)+1e-12)
    fig, ax  = plt.subplots(figsize=(8, 4))
    ax.hist(frob_err*100, bins=60, color=COLORS['fenep'], alpha=0.8, edgecolor='white')
    ax.set_xlabel('Relative Frobenius Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('3D Relative Frobenius Error Distribution')
    med = np.median(frob_err)*100
    ax.axvline(med, color='red', lw=1.5, ls='--', label=f'Median = {med:.2f}%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dashboard_3d(data, save_path):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    if data['history']:
        ax = fig.add_subplot(gs[0, 0])
        tl = data['history']['train_loss']
        vl = data['history']['val_loss']
        ep = np.arange(1, len(tl)+1)
        ax.semilogy(ep, tl, color=COLORS['train'], lw=1.8, label='Train')
        ax.semilogy(ep, vl, color=COLORS['val'],   lw=1.8, ls='--', label='Val')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
        ax.set_title('A — 3D Training Loss'); ax.legend(fontsize=9)

    if data['Y_true'] is not None:
        ax = fig.add_subplot(gs[0, 1])
        yt, yp = data['Y_true'][:,0], data['Y_pred'][:,0]
        lim   = [min(yt.min(),yp.min()), max(yt.max(),yp.max())]
        ax.scatter(yt, yp, s=2, alpha=0.15, color=COLORS['nn'], rasterized=True)
        ax.plot(lim, lim, 'k--', lw=1.2)
        r2 = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
        ax.text(0.05, 0.90, f'R²={r2:.3f}', transform=ax.transAxes, fontsize=10)
        ax.set_xlabel(r'True $\Omega_{xx}$'); ax.set_ylabel(r'Pred $\Omega_{xx}$')
        ax.set_title(r'B — Parity $\Omega_{xx}$')

    if data['Y_true'] is not None:
        ax = fig.add_subplot(gs[0, 2])
        err= (data['Y_pred']-data['Y_true']).flatten()
        ax.hist(err, bins=60, color=COLORS['nn'], alpha=0.8, edgecolor='white')
        ax.axvline(0, color='black', lw=1.2, ls='--')
        ax.set_xlabel('Error'); ax.set_ylabel('Count')
        ax.set_title('C — Error Distribution')

    if data['time'] is not None and data['C_ref'] is not None:
        ax  = fig.add_subplot(gs[1, :2])
        def tr(C): return np.trace(C, axis1=1, axis2=2)
        ax.plot(data['time'], tr(data['C_ref']), color=COLORS['ref'],   lw=2.5, label='BD Reference (3D)')
        ax.plot(data['time'], tr(data['C_nn']),  color=COLORS['nn'],    lw=2,   label='N2X NN (3D)')
        ax.plot(data['time'], tr(data['C_ucm']), color=COLORS['ucm'],   lw=2, ls='--', label='UCM')
        if data['C_fenep'] is not None:
            ax.plot(data['time'], tr(data['C_fenep']), color=COLORS['fenep'], lw=2, ls=':', label='FENE-P')
        ax.set_xlabel('Time'); ax.set_ylabel('tr(C)')
        ax.set_title(r'D — 3D Tensor Trace tr(C) (Shear flow $\dot\gamma=1$)')
        ax.legend(fontsize=9)

    if data['tau_ref'] is not None:
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(data['time'], data['tau_ref'][:,0,1], color=COLORS['ref'], lw=2.5, label='BD Ref')
        ax.plot(data['time'], data['tau_nn'][:,0,1],  color=COLORS['nn'],  lw=2,   label='NN')
        ax.plot(data['time'], data['tau_ucm'][:,0,1], color=COLORS['ucm'], lw=2, ls='--', label='UCM')
        ax.set_xlabel('Time'); ax.set_ylabel(r'$\tau_{xy}$')
        ax.set_title(r'E — Shear Stress $\tau_{xy}$')
        ax.legend(fontsize=9)

    fig.suptitle('N2X 3D — Complete Results Dashboard', fontsize=15, fontweight='bold')
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pinn_results(save_path):
    path = 'results/pinn_results.json'
    if not os.path.exists(path): return
    with open(path) as f: res = json.load(f)
    hist = res['history']
    ep = np.arange(1, len(hist['total_loss'])+1)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.semilogy(ep, hist['total_loss'], color='#9C27B0', lw=2, label='Total Loss (MSE + Physics)')
    ax.semilogy(ep, hist['mse_loss'],   color='#2196F3', lw=2, ls='--', label='MSE Loss')
    ax.semilogy(ep, hist['val_mse'],    color='#FF9800', lw=2, ls=':',  label='Validation MSE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.set_title('3D PINN Training Loss'); ax.legend()

    ax = axes[1]
    spd_vals = np.clip(hist['spd_loss'], 1e-12, None)
    ax.semilogy(ep, spd_vals, color='#E91E63', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel(r'SPD Penalty $L_{SPD}$')
    ax.set_title(r'Positive-Definiteness Penalty ($C \succ 0$)')

    fig.suptitle('3D PINN (Physics-Informed NN) Results', fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_spectral_equivariance(save_path):
    path = 'results/spectral_results.json'
    if not os.path.exists(path): return
    with open(path) as f: res = json.load(f)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    angles = np.array(res['rotation_angles']) * 180 / np.pi
    dir_errs  = np.clip(res['frob_errors_direct'], 1e-12, None)
    spec_errs = np.clip(res['frob_errors_spectral'], 1e-12, None)

    ax.scatter(angles, dir_errs,  color='#F44336', alpha=0.7, s=25, label='Direct NN (Raw Invariants)')
    ax.scatter(angles, spec_errs, color='#4CAF50', alpha=0.9, s=35, label='Spectral Method (Exact Frame Invariance)')
    ax.set_yscale('log')
    ax.set_xlabel(r'3D Rotation Angle $\alpha$ (degrees)')
    ax.set_ylabel('Relative Rotational Frame Error')
    ax.set_title('3D Rotational Frame Equivariance Proof (SO(3) Invariance)')
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")



def plot_tbnn_results(save_path):
    path = 'results/tbnn_results.json'
    if not os.path.exists(path): return
    with open(path) as f: res = json.load(f)
    hist = res['history']
    ep = np.arange(1, len(hist['train_mse'])+1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(ep, hist['train_mse'], color='#009688', lw=2, label='Train MSE')
    ax.semilogy(ep, hist['val_mse'],   color='#FF5722', lw=2, ls='--', label='Val MSE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('3D Tensor Basis Neural Network (TBNN) Training')
    ax.text(0.60, 0.80, f"Test R² = {res['r2']:.4f}\nRel. Frob = {res['frob_rel_pct']:.2f}%",
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_cfd_results(save_path):
    path = 'results/cfd_results.npz'
    if not os.path.exists(path): return
    data = np.load(path)
    y = data['y']; z = data['z']; ux = data['ux']
    u_newt = data['u_newtonian_mid']; u_n2x = data['u_centerline_n2x']
    trC_mid = data['trC_centerline']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Contour of 3D Velocity
    ax = axes[0]
    Y, Z = np.meshgrid(y, z, indexing='ij')
    c = ax.contourf(Y, Z, ux, 20, cmap='viridis')
    plt.colorbar(c, ax=ax, label=r'Velocity $u_x(y,z)$')
    ax.set_xlabel('y'); ax.set_ylabel('z')
    ax.set_title('A — 3D CFD Velocity Field $u_x(y,z)$')

    # Centerline Velocity Profile
    ax = axes[1]
    ax.plot(y, u_newt, color='#9E9E9E', lw=2.5, ls='--', label='Newtonian Poiseuille')
    ax.plot(y, u_n2x,  color='#2196F3', lw=2.5, label='Viscoelastic N2X Coupled')
    ax.set_xlabel('Channel Position y'); ax.set_ylabel(r'Velocity $u_x(y, z=0)$')
    ax.set_title('B — Centerline Velocity Profile')
    ax.legend()

    # Centerline Polymer Extension tr(C)
    ax = axes[2]
    ax.plot(y, trC_mid, color='#E91E63', lw=2.5)
    ax.set_xlabel('Channel Position y'); ax.set_ylabel(r'Polymer Trace tr$(C)$')
    ax.set_title(r'C — Polymer Stretch tr$(C)$ Profile')

    fig.suptitle('3D Multi-Scale CFD Simulation Results (N2X Neural Coupling)', fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    print("Loading 3D results …")
    data = load_results()

    print("\nGenerating 3D plots …")

    plot_training_loss(data['history'],
        'results/plots/fig1_training_loss.png')

    plot_parity_3d(data['Y_true'], data['Y_pred'],
        'results/plots/fig2_parity_plots.png')

    plot_tensor_trace_3d(data['time'], data['C_ref'], data['C_nn'],
                          data['C_ucm'], data['C_fenep'],
        'results/plots/fig3_tensor_trace.png')

    plot_stress_xy_3d(data['time'], data['tau_ref'], data['tau_nn'], data['tau_ucm'],
        'results/plots/fig4_stress_xy.png')

    plot_stress_xx_3d(data['time'], data['tau_ref'], data['tau_nn'], data['tau_ucm'],
        'results/plots/fig5_stress_xx.png')

    plot_Cxx_3d(data['time'], data['C_ref'], data['C_nn'],
                 data['C_ucm'], data['C_fenep'],
        'results/plots/fig6_Cxx.png')

    plot_error_histogram(data['Y_true'], data['Y_pred'],
        'results/plots/fig7_error_histogram.png')

    plot_frob_dist(data['Y_true'], data['Y_pred'],
        'results/plots/fig8_frobenius_dist.png')

    plot_dashboard_3d(data,
        'results/plots/fig9_dashboard.png')

    # Research Extension Plots
    plot_pinn_results('results/plots/fig10_pinn_results.png')
    plot_spectral_equivariance('results/plots/fig11_spectral_equivariance.png')
    plot_tbnn_results('results/plots/fig12_tbnn_results.png')
    plot_cfd_results('results/plots/fig13_cfd_multiscale.png')

    print("\nAll 3D plots (Standard + Research Extensions) generated ✓")

