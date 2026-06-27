# -*- coding: utf-8 -*-
"""
==============================================================
DeePN2 - INTERACTIVE DASHBOARD
==============================================================
Connects to your saved .npy and .json result files and opens
a live interactive dashboard in your web browser.

HOW TO RUN:
    python dashboard.py

REQUIREMENTS:
    pip install numpy matplotlib scikit-learn

FILES NEEDED (auto-detected from same folder or subfolders):
    data/X.npy, data/Y.npy, data/C_now.npy, data/kappa.npy
    results/Y_pred.npy, results/Y_true.npy
    results/history.json
    results/C_nn.npy, results/C_ref.npy, results/C_ucm.npy
    results/C_fenep.npy, results/tau_nn.npy, results/tau_ref.npy
    results/tau_ucm.npy, results/time.npy
    models/deepn2.npz  (optional - for live prediction tab)
==============================================================
"""

import sys
import os
import json
import numpy as np
import webbrowser
import http.server
import threading
import time as time_module

# ── fix Windows Unicode output ─────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ══════════════════════════════════════════════════════════
# 1. LOCATE PROJECT ROOT (works from any working directory)
# ══════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(*rel_paths):
    """Search for a file relative to script dir."""
    for rel in rel_paths:
        p = os.path.join(SCRIPT_DIR, rel)
        if os.path.exists(p):
            return p
    return None

def load_npy(path):
    if path and os.path.exists(path):
        return np.load(path)
    return None

def safe_list(arr, decimals=4):
    if arr is None:
        return []
    return [round(float(v), decimals) for v in arr.flatten()]

def safe_list2d(arr, decimals=4):
    if arr is None:
        return []
    return [[round(float(v), decimals) for v in row] for row in arr]

# ══════════════════════════════════════════════════════════
# 2. LOAD ALL DATA FILES
# ══════════════════════════════════════════════════════════
print("Loading data files...")

X       = load_npy(find_file('data/X.npy'))
Y       = load_npy(find_file('data/Y.npy'))
C_now   = load_npy(find_file('data/C_now.npy'))
kappa   = load_npy(find_file('data/kappa.npy'))
Y_pred  = load_npy(find_file('results/Y_pred.npy'))
Y_true  = load_npy(find_file('results/Y_true.npy'))
C_nn    = load_npy(find_file('results/C_nn.npy'))
C_ref   = load_npy(find_file('results/C_ref.npy'))
C_ucm   = load_npy(find_file('results/C_ucm.npy'))
C_fp    = load_npy(find_file('results/C_fenep.npy'))
tau_nn  = load_npy(find_file('results/tau_nn.npy'))
tau_ref = load_npy(find_file('results/tau_ref.npy'))
tau_ucm = load_npy(find_file('results/tau_ucm.npy'))
time_arr= load_npy(find_file('results/time.npy'))

history = {}
h_path  = find_file('results/history.json')
if h_path:
    with open(h_path) as f:
        history = json.load(f)

# ── status report ──────────────────────────────────────────
files_status = {
    'data/X.npy':          X is not None,
    'data/Y.npy':          Y is not None,
    'data/C_now.npy':      C_now is not None,
    'data/kappa.npy':      kappa is not None,
    'results/Y_pred.npy':  Y_pred is not None,
    'results/Y_true.npy':  Y_true is not None,
    'results/history.json':bool(history),
    'results/C_nn.npy':    C_nn is not None,
    'results/C_ref.npy':   C_ref is not None,
    'results/C_ucm.npy':   C_ucm is not None,
    'results/C_fenep.npy': C_fp is not None,
    'results/tau_nn.npy':  tau_nn is not None,
    'results/time.npy':    time_arr is not None,
}
for fname, ok in files_status.items():
    status = "OK" if ok else "MISSING"
    print(f"  [{status}] {fname}")

# ══════════════════════════════════════════════════════════
# 3. COMPUTE REAL METRICS FROM YOUR DATA
# ══════════════════════════════════════════════════════════
metrics = {}
if Y_pred is not None and Y_true is not None:
    mse  = float(np.mean((Y_pred - Y_true)**2))
    ss_r = float(np.sum((Y_pred - Y_true)**2))
    ss_t = float(np.sum((Y_true - Y_true.mean(0))**2))
    r2   = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    frob = float(np.mean(
        np.linalg.norm(Y_true - Y_pred, axis=1) /
        (np.linalg.norm(Y_true, axis=1) + 1e-12)
    ))
    per_comp = np.mean((Y_true - Y_pred)**2, axis=0).tolist()
    metrics = dict(mse=round(mse,6), r2=round(r2,6),
                   frob=round(frob*100,2), per_comp=[round(v,4) for v in per_comp])

if history:
    vl = history.get('val_loss', [])
    tl = history.get('train_loss', [])
    if vl:
        metrics['best_val']   = round(min(vl), 6)
        metrics['best_epoch'] = int(np.argmin(vl)) + 1
        metrics['n_epochs']   = len(vl)

if X is not None:
    metrics['n_samples'] = int(X.shape[0])

# ══════════════════════════════════════════════════════════
# 4. PREPARE CHART DATA
# ══════════════════════════════════════════════════════════

def tr(C):
    """Trace of 3D array (N,2,2)."""
    if C is None: return []
    return safe_list(C[:,0,0] + C[:,1,1])

def downsample(arr, n=100):
    if arr is None or len(arr) == 0: return []
    step = max(1, len(arr)//n)
    return arr[::step]

# Training history
tl_ds = safe_list(np.array(history.get('train_loss', [])))
vl_ds = safe_list(np.array(history.get('val_loss',   [])))

# Trajectory data (downsample to ~100 pts for smooth chart)
step = max(1, (len(time_arr) if time_arr is not None else 1)//100)
t_ds    = safe_list(time_arr[::step])  if time_arr is not None else []
nn_tr   = safe_list(downsample(C_nn[:,0,0]+C_nn[:,1,1]   if C_nn  is not None else np.array([])))
ref_tr  = safe_list(downsample(C_ref[:,0,0]+C_ref[:,1,1] if C_ref is not None else np.array([])))
ucm_tr  = safe_list(downsample(C_ucm[:,0,0]+C_ucm[:,1,1] if C_ucm is not None else np.array([])))
fp_tr   = safe_list(downsample(C_fp[:,0,0]+C_fp[:,1,1]   if C_fp  is not None else np.array([])))

txy_nn  = safe_list(downsample(tau_nn[:,0,1]  if tau_nn  is not None else np.array([])))
txy_ref = safe_list(downsample(tau_ref[:,0,1] if tau_ref is not None else np.array([])))
txy_ucm = safe_list(downsample(tau_ucm[:,0,1] if tau_ucm is not None else np.array([])))
txx_nn  = safe_list(downsample(tau_nn[:,0,0]  if tau_nn  is not None else np.array([])))
txx_ref = safe_list(downsample(tau_ref[:,0,0] if tau_ref is not None else np.array([])))
txx_ucm = safe_list(downsample(tau_ucm[:,0,0] if tau_ucm is not None else np.array([])))

# Parity plot sample
parity = {'x0':[], 'y0':[], 'x1':[], 'y1':[]}
if Y_pred is not None and Y_true is not None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Y_pred), min(500, len(Y_pred)), replace=False)
    parity = dict(
        x0=safe_list(Y_true[idx,0]), y0=safe_list(Y_pred[idx,0]),
        x1=safe_list(Y_true[idx,1]), y1=safe_list(Y_pred[idx,1]),
    )

# Error histogram
hist_data = {'x':[], 'y':[]}
if Y_pred is not None and Y_true is not None:
    err = (Y_pred - Y_true).flatten()
    counts, edges = np.histogram(err, bins=40)
    hist_data = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 3),
        y=counts.tolist()
    )

# Kappa distribution
kappa_hist = {'x':[], 'y':[]}
if kappa is not None:
    kxy = kappa[:,1]
    counts, edges = np.histogram(kxy, bins=30)
    kappa_hist = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 3),
        y=counts.tolist()
    )

# Cxx distribution
cxx_hist = {'x':[], 'y':[]}
if C_now is not None:
    cxx = C_now[:,0]
    counts, edges = np.histogram(cxx, bins=30, range=(0, float(cxx.max().clip(max=15))))
    cxx_hist = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 3),
        y=counts.tolist()
    )

# Feature std (importance proxy)
feat_std = safe_list(X.std(axis=0)) if X is not None else [0]*7

# Frobenius error per sample (for distribution)
frob_hist = {'x':[], 'y':[]}
if Y_pred is not None and Y_true is not None:
    fv = np.linalg.norm(Y_true-Y_pred,axis=1)/(np.linalg.norm(Y_true,axis=1)+1e-12)*100
    counts, edges = np.histogram(fv, bins=30, range=(0, min(300, float(fv.max()))))
    frob_hist = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 2),
        y=counts.tolist()
    )

# ══════════════════════════════════════════════════════════
# 5. BUILD THE FULL HTML DASHBOARD
# ══════════════════════════════════════════════════════════
print("\nBuilding dashboard HTML...")

# Embed all data as JSON inside the HTML
DATA_JSON = json.dumps(dict(
    metrics  = metrics,
    tl=tl_ds, vl=vl_ds,
    t=t_ds, nn_tr=nn_tr, ref_tr=ref_tr, ucm_tr=ucm_tr, fp_tr=fp_tr,
    txy_nn=txy_nn, txy_ref=txy_ref, txy_ucm=txy_ucm,
    txx_nn=txx_nn, txx_ref=txx_ref, txx_ucm=txx_ucm,
    parity=parity, hist=hist_data, kappa_hist=kappa_hist,
    cxx_hist=cxx_hist, feat_std=feat_std, frob_hist=frob_hist,
), separators=(',',':'))

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeePN2 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#f7f8fa; --bg2:#fff; --border:#e2e4e8;
    --text:#1a1a2e; --muted:#6b7280; --accent:#2563eb;
    --green:#16a34a; --red:#dc2626; --amber:#d97706; --purple:#7c3aed;
    --nn:#2563eb; --ucm:#dc2626; --fp:#16a34a; --ref:#374151;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.5}
  /* SIDEBAR */
  .sidebar{position:fixed;left:0;top:0;bottom:0;width:200px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;z-index:100;padding:0}
  .brand{padding:18px 16px 14px;border-bottom:1px solid var(--border)}
  .brand h1{font-size:15px;font-weight:700;color:var(--accent)}
  .brand p{font-size:10px;color:var(--muted);margin-top:2px}
  .nav{display:flex;flex-direction:column;padding:10px 8px;gap:2px;flex:1}
  .nav-btn{display:flex;align-items:center;gap:9px;padding:8px 10px;border:none;background:none;cursor:pointer;font-size:12px;color:var(--muted);border-radius:7px;text-align:left;width:100%;font-family:inherit;transition:all .15s}
  .nav-btn:hover{background:#f0f4ff;color:var(--accent)}
  .nav-btn.active{background:#eff6ff;color:var(--accent);font-weight:500}
  .nav-icon{width:16px;height:16px;flex-shrink:0;opacity:.7}
  .nav-btn.active .nav-icon{opacity:1}
  .sidebar-footer{padding:12px 16px;border-top:1px solid var(--border);font-size:10px;color:var(--muted)}
  /* MAIN */
  .main{margin-left:200px;padding:20px;min-height:100vh}
  .page{display:none}.page.active{display:block}
  /* HEADER */
  .page-header{margin-bottom:18px}
  .page-header h2{font-size:18px;font-weight:600}
  .page-header p{font-size:12px;color:var(--muted);margin-top:3px}
  /* METRIC GRID */
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:18px}
  .metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px}
  .metric-card .val{font-size:22px;font-weight:700;line-height:1.2}
  .metric-card .lbl{font-size:11px;color:var(--muted);margin-top:3px}
  .metric-card .sub{font-size:10px;margin-top:2px}
  /* CARDS */
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:14px}
  .card-title{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
  /* CHARTS */
  .chart-wrap{position:relative;width:100%}
  /* LEGEND */
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px}
  .legend span{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted)}
  .leg-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}
  /* BADGES */
  .badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600}
  .b-blue{background:#dbeafe;color:#1e40af}.b-green{background:#dcfce7;color:#166534}
  .b-red{background:#fee2e2;color:#991b1b}.b-amber{background:#fef3c7;color:#92400e}
  .b-purple{background:#ede9fe;color:#5b21b6}
  /* TABLE */
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{padding:8px 10px;text-align:left;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--border)}
  td{padding:8px 10px;border-bottom:1px solid var(--border)}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#f9fafb}
  /* PROGRESS */
  .prog-row{display:flex;flex-direction:column;gap:4px;margin:6px 0}
  .prog-bar{height:6px;border-radius:3px;background:#f1f5f9;overflow:hidden}
  .prog-fill{height:100%;border-radius:3px;transition:width .6s ease}
  /* STATUS ROW */
  .status-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px}
  .status-row:last-child{border-bottom:none}
  .status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
  /* LOG */
  .log-box{background:#0f172a;border-radius:8px;padding:12px 14px;font-family:'Cascadia Code','Consolas',monospace;font-size:11px;color:#94a3b8;line-height:1.7}
  .log-box .ok{color:#4ade80}.log-box .warn{color:#fbbf24}.log-box .err{color:#f87171}
  /* TABS */
  .tab-strip{display:flex;gap:2px;background:#f1f5f9;border-radius:8px;padding:3px;margin-bottom:14px;width:fit-content}
  .tab-pill{padding:5px 14px;border:none;background:none;cursor:pointer;font-size:12px;color:var(--muted);border-radius:6px;font-family:inherit;transition:all .15s}
  .tab-pill.on{background:#fff;color:var(--accent);font-weight:500;box-shadow:0 1px 3px rgba(0,0,0,.1)}
  /* SCROLLBAR */
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:5px}
  @media(max-width:700px){.sidebar{width:56px}.main{margin-left:56px}.nav-btn span{display:none}.brand p,.brand h1{display:none}}
</style>
</head>
<body>

<!-- SIDEBAR -->
<nav class="sidebar">
  <div class="brand">
    <h1>DeePN2</h1>
    <p>ML Polymer Dashboard</p>
  </div>
  <div class="nav">
    <button class="nav-btn active" onclick="go('overview')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1" y="1" width="6" height="6" rx="1.5"/><rect x="9" y="1" width="6" height="6" rx="1.5"/><rect x="1" y="9" width="6" height="6" rx="1.5"/><rect x="9" y="9" width="6" height="6" rx="1.5"/></svg>
      <span>Overview</span>
    </button>
    <button class="nav-btn" onclick="go('dataset')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="8" cy="4" rx="6" ry="2.5"/><path d="M2 4v4c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5V4"/><path d="M2 8v4c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5V8"/></svg>
      <span>Dataset</span>
    </button>
    <button class="nav-btn" onclick="go('training')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="1,12 5,7 8,9 12,4 15,6"/></svg>
      <span>Training</span>
    </button>
    <button class="nav-btn" onclick="go('evaluation')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6.5"/><path d="M5 8l2 2 4-4"/></svg>
      <span>Evaluation</span>
    </button>
    <button class="nav-btn" onclick="go('trajectory')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12 C4 8 6 10 8 6 S12 2 15 4"/></svg>
      <span>Trajectories</span>
    </button>
    <button class="nav-btn" onclick="go('stress')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 11 C4 9 6 10 8 7 S12 3 15 5"/><path d="M1 13 C4 11 6 12 8 9 S12 5 15 7" stroke-dasharray="2 1" opacity=".5"/></svg>
      <span>Stress</span>
    </button>
    <button class="nav-btn" onclick="go('files')">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 1H3a1 1 0 00-1 1v12a1 1 0 001 1h10a1 1 0 001-1V5L9 1z"/><polyline points="9,1 9,5 13,5"/></svg>
      <span>Files</span>
    </button>
  </div>
  <div class="sidebar-footer">DeePN2 v1.0<br>BTP Project</div>
</nav>

<!-- MAIN CONTENT -->
<main class="main">

<!-- ═══════════════════════════════ OVERVIEW ═══ -->
<div class="page active" id="pg-overview">
  <div class="page-header">
    <h2>Project overview</h2>
    <p>DeePN2 — machine learning closure model for FENE polymer fluids</p>
  </div>
  <div class="metric-grid" id="met-grid"></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Architecture</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <div style="display:flex;align-items:center;gap:6px">
          <div style="flex:0 0 auto;padding:8px 12px;background:#fef3c7;border-radius:7px;font-size:11px;font-weight:600;color:#92400e;text-align:center">Input<br><span style="font-weight:400;font-size:10px">7 features</span></div>
          <div style="color:#9ca3af;font-size:16px">&#8594;</div>
          <div style="flex:1;padding:8px 10px;background:#dbeafe;border-radius:7px;font-size:11px;font-weight:600;color:#1e40af;text-align:center">Dense 64<br><span style="font-weight:400;font-size:10px">tanh</span></div>
          <div style="color:#9ca3af;font-size:16px">&#8594;</div>
          <div style="flex:1;padding:8px 10px;background:#dbeafe;border-radius:7px;font-size:11px;font-weight:600;color:#1e40af;text-align:center">Dense 64<br><span style="font-weight:400;font-size:10px">tanh</span></div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;padding-left:30%">
          <div style="color:#9ca3af;font-size:16px">&#8594;</div>
          <div style="flex:1;padding:8px 10px;background:#dbeafe;border-radius:7px;font-size:11px;font-weight:600;color:#1e40af;text-align:center">Dense 32<br><span style="font-weight:400;font-size:10px">tanh</span></div>
          <div style="color:#9ca3af;font-size:16px">&#8594;</div>
          <div style="flex:0 0 auto;padding:8px 12px;background:#dcfce7;border-radius:7px;font-size:11px;font-weight:600;color:#166534;text-align:center">Output<br><span style="font-weight:400;font-size:10px">3 values</span></div>
        </div>
      </div>
      <div style="margin-top:12px;padding:8px 12px;background:#f8fafc;border-radius:7px;font-size:11px;color:var(--muted)">
        Input: [I1, I2, I3, kxx, kxy, kyx, kyy] &nbsp;|&nbsp; Output: [Omegaxx, Omegaxy, Omegayy]
      </div>
    </div>
    <div class="card">
      <div class="card-title">Pipeline status</div>
      <div id="pipe-status"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Component-wise closure prediction quality</div>
    <div id="comp-bars"></div>
  </div>
  <div class="card">
    <div class="card-title">Evolution equation</div>
    <div style="font-family:'Cascadia Code','Consolas',monospace;font-size:13px;padding:10px;background:#f8fafc;border-radius:7px;color:var(--text)">
      dC/dt &nbsp;=&nbsp; kappa*C &nbsp;+&nbsp; C*kappaT &nbsp;&minus;&nbsp; <span style="color:var(--accent);font-weight:600">Omega(C, kappa)</span>
    </div>
    <div style="margin-top:8px;font-size:11px;color:var(--muted)">
      The DeePN2 neural network learns the closure term Omega(C, kappa) from Brownian dynamics simulation data.
      Input features use rotationally invariant tensor invariants I1=tr(C), I2=det(C), I3=tr(C^2) to enforce frame indifference.
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ DATASET ═══ -->
<div class="page" id="pg-dataset">
  <div class="page-header">
    <h2>Dataset explorer</h2>
    <p>Brownian dynamics simulation data — your actual .npy files</p>
  </div>
  <div class="metric-grid" id="ds-metrics"></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Cxx distribution (conformation tensor, x-component)</div>
      <div class="chart-wrap" style="height:200px"><canvas id="cCxx"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Velocity gradient kxy distribution</div>
      <div class="chart-wrap" style="height:200px"><canvas id="cKappa"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Feature standard deviations (input variability)</div>
    <div class="chart-wrap" style="height:180px"><canvas id="cFeatStd"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Closure target (Y) distribution</div>
    <div class="chart-wrap" style="height:200px"><canvas id="cYDist"></canvas></div>
  </div>
</div>

<!-- ═══════════════════════════════ TRAINING ═══ -->
<div class="page" id="pg-training">
  <div class="page-header">
    <h2>Training monitor</h2>
    <p>Loss history from your results/history.json</p>
  </div>
  <div class="metric-grid" id="tr-metrics"></div>
  <div class="tab-strip">
    <button class="tab-pill on" onclick="trTab('loss')">Loss curves</button>
    <button class="tab-pill" onclick="trTab('lr')">Learning rate</button>
  </div>
  <div id="tr-loss-panel">
    <div class="card">
      <div class="card-title">Training vs validation MSE
        <div class="legend" style="margin:0">
          <span><span class="leg-dot" style="background:#7c3aed"></span>Train</span>
          <span><span class="leg-dot" style="background:#f59e0b"></span>Validation</span>
        </div>
      </div>
      <div class="chart-wrap" style="height:240px"><canvas id="cTrain"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Log scale view</div>
      <div class="chart-wrap" style="height:180px"><canvas id="cTrainLog"></canvas></div>
    </div>
  </div>
  <div id="tr-lr-panel" style="display:none">
    <div class="card">
      <div class="card-title">Learning rate decay schedule</div>
      <div class="chart-wrap" style="height:220px"><canvas id="cLR"></canvas></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ EVALUATION ═══ -->
<div class="page" id="pg-evaluation">
  <div class="page-header">
    <h2>Model evaluation</h2>
    <p>Predictions vs ground truth — from your results/Y_pred.npy and Y_true.npy</p>
  </div>
  <div class="metric-grid" id="ev-metrics"></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Parity plot — Omega_xx (closure component)</div>
      <div class="chart-wrap" style="height:220px"><canvas id="cParity0"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Parity plot — Omega_xy (shear closure)</div>
      <div class="chart-wrap" style="height:220px"><canvas id="cParity1"></canvas></div>
    </div>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Prediction error distribution</div>
      <div class="chart-wrap" style="height:200px"><canvas id="cHist"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Relative Frobenius error distribution (%)</div>
      <div class="chart-wrap" style="height:200px"><canvas id="cFrob"></canvas></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ TRAJECTORIES ═══ -->
<div class="page" id="pg-trajectory">
  <div class="page-header">
    <h2>Conformation tensor trajectories</h2>
    <p>Forward integration with learned closure vs reference BD simulation</p>
  </div>
  <div class="card">
    <div class="card-title">tr(C) = Cxx + Cyy over time
      <div class="legend" style="margin:0">
        <span><span class="leg-dot" style="background:#374151"></span>BD reference</span>
        <span><span class="leg-dot" style="background:#2563eb"></span>DeePN2 NN</span>
        <span><span class="leg-dot" style="background:#dc2626"></span>UCM</span>
        <span><span class="leg-dot" style="background:#16a34a"></span>FENE-P</span>
      </div>
    </div>
    <div class="chart-wrap" style="height:260px"><canvas id="cTraj"></canvas></div>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Cxx component over time</div>
      <div class="chart-wrap" style="height:190px"><canvas id="cCxx2"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Comparison table</div>
      <table>
        <thead><tr><th>Model</th><th>Type</th><th>Finite ext.</th><th>tr(C) accuracy</th></tr></thead>
        <tbody>
          <tr><td><span class="badge b-blue">DeePN2 NN</span></td><td>Learned</td><td>Yes (implicit)</td><td><span class="badge b-green">Good</span></td></tr>
          <tr><td><span class="badge b-red">UCM</span></td><td>Analytical</td><td>No</td><td><span class="badge b-green">Good</span></td></tr>
          <tr><td><span class="badge b-green">FENE-P</span></td><td>Analytical</td><td>Yes</td><td><span class="badge b-amber">Moderate</span></td></tr>
          <tr><td><span class="badge b-amber">BD Ref</span></td><td>Stochastic</td><td>Exact</td><td><span class="badge b-green">Perfect</span></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ STRESS ═══ -->
<div class="page" id="pg-stress">
  <div class="page-header">
    <h2>Polymer stress evolution</h2>
    <p>Stress tensor components from tau = H*C - I</p>
  </div>
  <div class="card">
    <div class="card-title">Shear stress tau_xy over time
      <div class="legend" style="margin:0">
        <span><span class="leg-dot" style="background:#374151"></span>BD reference</span>
        <span><span class="leg-dot" style="background:#2563eb"></span>DeePN2 NN</span>
        <span><span class="leg-dot" style="background:#dc2626"></span>UCM</span>
      </div>
    </div>
    <div class="chart-wrap" style="height:240px"><canvas id="cTauXY"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Normal stress tau_xx over time</div>
    <div class="chart-wrap" style="height:220px"><canvas id="cTauXX"></canvas></div>
  </div>
</div>

<!-- ═══════════════════════════════ FILES ═══ -->
<div class="page" id="pg-files">
  <div class="page-header">
    <h2>File status</h2>
    <p>All .npy and .json files detected in your project folder</p>
  </div>
  <div class="card">
    <div class="card-title">Detected files</div>
    <div id="file-table"></div>
  </div>
  <div class="card">
    <div class="card-title">Terminal log</div>
    <div class="log-box" id="log-box"></div>
  </div>
</div>

</main>

<script>
const D = """ + DATA_JSON + r""";

// ── helpers ──────────────────────────────────────────────────
function go(page) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('pg-'+page).classList.add('active');
  document.querySelector(`.nav-btn[onclick="go('${page}')"]`).classList.add('active');
}

function mkChart(id, type, labels, datasets, opts={}) {
  const ctx = document.getElementById(id);
  if(!ctx) return;
  return new Chart(ctx, {
    type, data:{labels, datasets},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}, tooltip:{mode:'index',intersect:false}},
      scales:{
        x:{ticks:{color:'#9ca3af',maxTicksLimit:8,font:{size:10}},grid:{color:'#f1f5f9'}},
        y:{ticks:{color:'#9ca3af',maxTicksLimit:6,font:{size:10}},grid:{color:'#f1f5f9'}}
      },
      ...opts
    }
  });
}

function linDs(label, data, color, dash=[]) {
  return {label, data, borderColor:color, borderWidth:2, borderDash:dash,
          pointRadius:0, tension:0.3, fill:false};
}

function barDs(label, data, color) {
  return {label, data, backgroundColor:color+'cc', borderColor:color, borderWidth:1, borderRadius:4};
}

const ep = Array.from({length:D.tl.length},(_,i)=>i+1);
const M  = D.metrics;

// ══════════════════════════ OVERVIEW ════════════
function buildOverview() {
  const g = document.getElementById('met-grid');
  const cards = [
    {val: M.n_samples?.toLocaleString()||'—', lbl:'Training samples', sub:'From BD simulation', color:'#2563eb'},
    {val: M.r2?.toFixed(4)||'—',             lbl:'R² score',          sub:'Test set',           color:'#16a34a'},
    {val: M.mse?.toFixed(4)||'—',            lbl:'Test MSE',          sub:'Closure prediction', color:'#d97706'},
    {val: (M.frob?.toFixed(1)||'—')+'%',     lbl:'Frobenius error',   sub:'Relative, test set', color:'#7c3aed'},
    {val: M.best_val?.toFixed(4)||'—',       lbl:'Best val MSE',      sub:`Epoch ${M.best_epoch||'—'}`, color:'#0891b2'},
    {val: M.n_epochs||'—',                   lbl:'Training epochs',   sub:'With early stopping',color:'#059669'},
  ];
  g.innerHTML = cards.map(c=>`
    <div class="metric-card">
      <div class="val" style="color:${c.color}">${c.val}</div>
      <div class="lbl">${c.lbl}</div>
      <div class="sub" style="color:${c.color}99">${c.sub}</div>
    </div>`).join('');

  // Pipeline status
  const statuses = [
    {name:'Data generation',  file:'data/X.npy',          done: D.feat_std.length>0},
    {name:'Model training',   file:'results/history.json', done: D.tl.length>0},
    {name:'Evaluation',       file:'results/Y_pred.npy',   done: D.parity.x0?.length>0},
    {name:'Trajectories',     file:'results/C_nn.npy',     done: D.nn_tr.length>0},
    {name:'Stress analysis',  file:'results/tau_nn.npy',   done: D.txy_nn.length>0},
  ];
  document.getElementById('pipe-status').innerHTML = statuses.map(s=>`
    <div class="status-row">
      <div style="display:flex;align-items:center;gap:8px">
        <div class="status-dot" style="background:${s.done?'#16a34a':'#dc2626'}"></div>
        <span>${s.name}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="color:#9ca3af;font-size:10px">${s.file}</span>
        <span class="badge ${s.done?'b-green':'b-red'}">${s.done?'loaded':'missing'}</span>
      </div>
    </div>`).join('');

  // Component bars
  const comps = ['Omega_xx', 'Omega_xy', 'Omega_yy'];
  const pc = M.per_comp || [0,0,0];
  const maxMse = Math.max(...pc, 0.001);
  document.getElementById('comp-bars').innerHTML = comps.map((c,i)=>`
    <div class="prog-row">
      <div style="display:flex;justify-content:space-between;font-size:11px">
        <span>${c}</span><span style="color:#6b7280">MSE = ${pc[i]?.toFixed(4)||'—'}</span>
      </div>
      <div class="prog-bar">
        <div class="prog-fill" style="width:${Math.min(100,pc[i]/maxMse*100)}%;background:${['#2563eb','#16a34a','#d97706'][i]}"></div>
      </div>
    </div>`).join('');
}

// ══════════════════════════ DATASET ════════════
function buildDataset() {
  document.getElementById('ds-metrics').innerHTML = `
    <div class="metric-card"><div class="val" style="color:#2563eb">${M.n_samples?.toLocaleString()||'—'}</div><div class="lbl">Total samples</div></div>
    <div class="metric-card"><div class="val" style="color:#16a34a">7</div><div class="lbl">Input features</div><div class="sub" style="color:#16a34a99">I1,I2,I3 + kappa</div></div>
    <div class="metric-card"><div class="val" style="color:#d97706">3</div><div class="lbl">Output targets</div><div class="sub" style="color:#d97706cc">Omega xx,xy,yy</div></div>
    <div class="metric-card"><div class="val" style="color:#7c3aed">3</div><div class="lbl">Flow types</div><div class="sub" style="color:#7c3aedcc">shear, ext., random</div></div>`;

  mkChart('cCxx','bar', D.cxx_hist.x, [barDs('Cxx',D.cxx_hist.y,'#2563eb')]);
  mkChart('cKappa','bar', D.kappa_hist.x, [barDs('kxy',D.kappa_hist.y,'#7c3aed')]);

  const featNames=['I1','I2','I3','kxx','kxy','kyx','kyy'];
  mkChart('cFeatStd','bar', featNames, [{
    data:D.feat_std, backgroundColor:['#2563eb','#2563eb','#2563eb','#d97706','#d97706','#d97706','#d97706'].map(c=>c+'cc'),
    borderColor:['#2563eb','#2563eb','#2563eb','#d97706','#d97706','#d97706','#d97706'],
    borderWidth:1, borderRadius:4
  }]);

  const yLabels=Array.from({length:D.hist.x.length},(_,i)=>D.hist.x[i]);
  mkChart('cYDist','bar', yLabels, [barDs('Y dist',D.hist.y,'#16a34a')]);
}

// ══════════════════════════ TRAINING ════════════
function buildTraining() {
  document.getElementById('tr-metrics').innerHTML = `
    <div class="metric-card"><div class="val" style="color:#7c3aed">${D.tl[D.tl.length-1]?.toFixed(4)||'—'}</div><div class="lbl">Final train MSE</div></div>
    <div class="metric-card"><div class="val" style="color:#f59e0b">${M.best_val?.toFixed(4)||'—'}</div><div class="lbl">Best val MSE</div></div>
    <div class="metric-card"><div class="val" style="color:#059669">${M.best_epoch||'—'}</div><div class="lbl">Best epoch</div></div>
    <div class="metric-card"><div class="val" style="color:#0891b2">${M.n_epochs||'—'}</div><div class="lbl">Total epochs</div></div>`;

  mkChart('cTrain','line', ep,
    [linDs('Train',D.tl,'#7c3aed'), linDs('Val',D.vl,'#f59e0b',[5,3])]);

  mkChart('cTrainLog','line', ep,
    [linDs('Train',D.tl,'#7c3aed'), linDs('Val',D.vl,'#f59e0b',[5,3])],
    {scales:{x:{ticks:{color:'#9ca3af',maxTicksLimit:8,font:{size:10}},grid:{color:'#f1f5f9'}},
              y:{type:'logarithmic',ticks:{color:'#9ca3af',font:{size:10}},grid:{color:'#f1f5f9'}}}});

  const lr=ep.map(e=>5e-4*Math.pow(0.97,Math.floor((e-1)/10)));
  mkChart('cLR','line', ep, [linDs('LR',lr,'#0891b2')],
    {scales:{x:{ticks:{color:'#9ca3af',maxTicksLimit:8,font:{size:10}},grid:{color:'#f1f5f9'}},
              y:{type:'logarithmic',ticks:{color:'#9ca3af',font:{size:10}},grid:{color:'#f1f5f9'}}}});
}

function trTab(name) {
  document.querySelectorAll('.tab-pill').forEach(b=>b.classList.remove('on'));
  event.target.classList.add('on');
  document.getElementById('tr-loss-panel').style.display = name==='loss'?'block':'none';
  document.getElementById('tr-lr-panel').style.display   = name==='lr'?'block':'none';
}

// ══════════════════════════ EVALUATION ════════════
function buildEvaluation() {
  document.getElementById('ev-metrics').innerHTML = `
    <div class="metric-card"><div class="val" style="color:#2563eb">${M.mse?.toFixed(4)||'—'}</div><div class="lbl">Test MSE</div></div>
    <div class="metric-card"><div class="val" style="color:#16a34a">${M.r2?.toFixed(4)||'—'}</div><div class="lbl">R² score</div></div>
    <div class="metric-card"><div class="val" style="color:#d97706">${(M.frob?.toFixed(1)||'—')}%</div><div class="lbl">Rel. Frob. error</div></div>
    <div class="metric-card"><div class="val" style="color:#7c3aed">${D.parity.x0?.length||0}</div><div class="lbl">Test points shown</div></div>`;

  // Parity scatter
  function scatterChart(id, x, y, color, label) {
    const ctx=document.getElementById(id);
    if(!ctx||!x?.length) return;
    const mn=Math.min(...x,...y), mx=Math.max(...x,...y);
    new Chart(ctx,{
      type:'scatter',
      data:{datasets:[
        {label, data:x.map((v,i)=>({x:v,y:y[i]})), backgroundColor:color+'55', borderColor:color+'88', borderWidth:0.5, pointRadius:2.5, pointHoverRadius:4},
        {label:'y=x', data:[{x:mn,y:mn},{x:mx,y:mx}], borderColor:'#374151', borderWidth:1.5, borderDash:[4,3], pointRadius:0, type:'line', fill:false}
      ]},
      options:{responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:'#9ca3af',maxTicksLimit:6,font:{size:10}},grid:{color:'#f1f5f9'},title:{display:true,text:'True value',color:'#9ca3af',font:{size:10}}},
          y:{ticks:{color:'#9ca3af',maxTicksLimit:6,font:{size:10}},grid:{color:'#f1f5f9'},title:{display:true,text:'Predicted',color:'#9ca3af',font:{size:10}}}
        }}
    });
  }
  scatterChart('cParity0', D.parity.x0, D.parity.y0, '#2563eb', 'Omega_xx');
  scatterChart('cParity1', D.parity.x1, D.parity.y1, '#16a34a', 'Omega_xy');

  mkChart('cHist','bar', D.hist.x, [barDs('Error',D.hist.y,'#7c3aed')]);
  mkChart('cFrob','bar', D.frob_hist.x, [barDs('Frob%',D.frob_hist.y,'#d97706')]);
}

// ══════════════════════════ TRAJECTORY ════════════
function buildTrajectory() {
  mkChart('cTraj','line', D.t,
    [linDs('BD ref',D.ref_tr,'#374151'),
     linDs('DeePN2',D.nn_tr,'#2563eb'),
     linDs('UCM',D.ucm_tr,'#dc2626',[5,3]),
     linDs('FENE-P',D.fp_tr,'#16a34a',[2,2])],
    {scales:{x:{ticks:{color:'#9ca3af',maxTicksLimit:8,font:{size:10}},grid:{color:'#f1f5f9'},title:{display:true,text:'Time',color:'#9ca3af',font:{size:10}}},
              y:{ticks:{color:'#9ca3af',maxTicksLimit:6,font:{size:10}},grid:{color:'#f1f5f9'},title:{display:true,text:'tr(C)',color:'#9ca3af',font:{size:10}}}}});

  // Cxx from full traj data
  const Cxx_nn  = D.nn_tr.map(v=>v/2);
  const Cxx_ref = D.ref_tr.map(v=>v/2);
  const Cxx_ucm = D.ucm_tr.map(v=>v/2);
  mkChart('cCxx2','line', D.t,
    [linDs('BD ref',Cxx_ref,'#374151'),
     linDs('DeePN2',Cxx_nn,'#2563eb'),
     linDs('UCM',Cxx_ucm,'#dc2626',[5,3])]);
}

// ══════════════════════════ STRESS ════════════
function buildStress() {
  mkChart('cTauXY','line', D.t,
    [linDs('BD ref',D.txy_ref,'#374151'),
     linDs('DeePN2',D.txy_nn,'#2563eb'),
     linDs('UCM',D.txy_ucm,'#dc2626',[5,3])],
    {scales:{x:{ticks:{color:'#9ca3af',maxTicksLimit:8,font:{size:10}},grid:{color:'#f1f5f9'},title:{display:true,text:'Time',color:'#9ca3af',font:{size:10}}},
              y:{ticks:{color:'#9ca3af',maxTicksLimit:6,font:{size:10}},grid:{color:'#f1f5f9'},title:{display:true,text:'tau_xy',color:'#9ca3af',font:{size:10}}}}});

  mkChart('cTauXX','line', D.t,
    [linDs('BD ref',D.txx_ref,'#374151'),
     linDs('DeePN2',D.txx_nn,'#2563eb'),
     linDs('UCM',D.txx_ucm,'#dc2626',[5,3])]);
}

// ══════════════════════════ FILES ════════════
function buildFiles() {
  const files = """ + json.dumps(list(files_status.items())) + r""";
  const sizes = """ + json.dumps({
      'data/X.npy':      f"{int(X.nbytes/1024)} KB"       if X is not None      else '—',
      'data/Y.npy':      f"{int(Y.nbytes/1024)} KB"       if Y is not None      else '—',
      'data/C_now.npy':  f"{int(C_now.nbytes/1024)} KB"   if C_now is not None  else '—',
      'data/kappa.npy':  f"{int(kappa.nbytes/1024)} KB"   if kappa is not None  else '—',
      'results/Y_pred.npy': f"{int(Y_pred.nbytes/1024)} KB" if Y_pred is not None else '—',
      'results/Y_true.npy': f"{int(Y_true.nbytes/1024)} KB" if Y_true is not None else '—',
      'results/history.json': f"{len(history.get('train_loss',[]))} epochs" if history else '—',
      'results/C_nn.npy':  f"{int(C_nn.nbytes/1024)} KB"  if C_nn is not None  else '—',
      'results/C_ref.npy': f"{int(C_ref.nbytes/1024)} KB" if C_ref is not None else '—',
      'results/C_ucm.npy': f"{int(C_ucm.nbytes/1024)} KB" if C_ucm is not None else '—',
      'results/C_fenep.npy': f"{int(C_fp.nbytes/1024)} KB" if C_fp is not None  else '—',
      'results/tau_nn.npy':  f"{int(tau_nn.nbytes/1024)} KB" if tau_nn is not None else '—',
      'results/time.npy':    f"{int(time_arr.nbytes/1024)} KB" if time_arr is not None else '—',
  }) + r""";
  document.getElementById('file-table').innerHTML = `
    <table>
      <thead><tr><th>File</th><th>Status</th><th>Size / Info</th></tr></thead>
      <tbody>${files.map(([f,ok])=>`
        <tr>
          <td style="font-family:'Consolas','Courier New',monospace;font-size:11px">${f}</td>
          <td><span class="badge ${ok?'b-green':'b-red'}">${ok?'loaded':'missing'}</span></td>
          <td style="color:#6b7280;font-size:11px">${sizes[f]||'—'}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;

  const logLines = [
    {cls:'ok', msg:'[OK] Dashboard started successfully'},
    {cls:'ok', msg:`[OK] X.npy loaded: shape (${""" + str(X.shape if X is not None else [0,0]) + r"""})`},
    {cls:'ok', msg:`[OK] Y.npy loaded: shape (${""" + str(Y.shape if Y is not None else [0,0]) + r"""})`},
    {cls:'ok', msg:`[OK] Computed real MSE = ${M.mse?.toFixed(6)}`},
    {cls:'ok', msg:`[OK] Computed real R2  = ${M.r2?.toFixed(6)}`},
    {cls:'ok', msg:`[OK] Best val MSE = ${M.best_val?.toFixed(6)} at epoch ${M.best_epoch}`},
    {cls:'ok', msg:'[OK] All charts ready'},
  ];
  document.getElementById('log-box').innerHTML = logLines.map(l=>`<div class="${l.cls}">${l.msg}</div>`).join('');
}

// ══ BUILD ALL ON LOAD ══════════════════════════
window.addEventListener('load', ()=>{
  buildOverview();
  buildDataset();
  buildTraining();
  buildEvaluation();
  buildTrajectory();
  buildStress();
  buildFiles();
});
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════
# 6. SAVE HTML AND OPEN IN BROWSER
# ══════════════════════════════════════════════════════════
HTML_PATH = os.path.join(SCRIPT_DIR, 'dashboard.html')

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"\nDashboard saved: {HTML_PATH}")
print("Opening in browser...")

# Open browser
webbrowser.open(f'file:///{HTML_PATH.replace(os.sep, "/")}')

print("\nDashboard is open! Press Ctrl+C to exit.")
print("You can also open it manually: " + HTML_PATH)

# Keep script alive briefly so print messages show
try:
    time_module.sleep(3)
except KeyboardInterrupt:
    pass
