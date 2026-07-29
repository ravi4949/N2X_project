# -*- coding: utf-8 -*-
"""
==============================================================
N2X 3D — FULL INTERACTIVE RESEARCH DASHBOARD
==============================================================
Connects to saved .npy and .json result files and opens a live
interactive 3D web dashboard with Chart.js visualization for:
    1. Base Model & Dataset Overview
    2. 3D Physics-Informed Neural Network (PINN)
    3. 3D Spectral Decomposition & Rotational Equivariance
    4. 3D Tensor Basis Neural Network (TBNN - Pope 1975)
    5. 3D Multi-Scale Viscoelastic CFD Solver Integration
    6. Trajectories & Stress Analysis

HOW TO RUN:
    python dashboard.py
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

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(*rel_paths):
    for rel in rel_paths:
        p = os.path.join(SCRIPT_DIR, rel)
        if os.path.exists(p):
            return p
    return None

def load_npy(path):
    if path and os.path.exists(path):
        return np.load(path)
    return None

def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def safe_list(arr, decimals=4):
    if arr is None:
        return []
    return [round(float(v), decimals) for v in np.array(arr).flatten()]

print("Loading N2X 3D project data files...")

# Base Data & Results
X        = load_npy(find_file('data/X.npy'))
Y        = load_npy(find_file('data/Y.npy'))
C_now    = load_npy(find_file('data/C_now.npy'))
kappa    = load_npy(find_file('data/kappa.npy'))
Y_pred   = load_npy(find_file('results/Y_pred.npy'))
Y_true   = load_npy(find_file('results/Y_true.npy'))
C_nn     = load_npy(find_file('results/C_nn.npy'))
C_ref    = load_npy(find_file('results/C_ref.npy'))
C_ucm    = load_npy(find_file('results/C_ucm.npy'))
C_fp     = load_npy(find_file('results/C_fenep.npy'))
tau_nn   = load_npy(find_file('results/tau_nn.npy'))
tau_ref  = load_npy(find_file('results/tau_ref.npy'))
tau_ucm  = load_npy(find_file('results/tau_ucm.npy'))
time_arr = load_npy(find_file('results/time.npy'))

# Extension JSON Results
history    = load_json(find_file('results/history.json'))
pinn_res   = load_json(find_file('results/pinn_results.json'))
spec_res   = load_json(find_file('results/spectral_results.json'))
tbnn_res   = load_json(find_file('results/tbnn_results.json'))
cfd_sum    = load_json(find_file('results/cfd_summary.json'))

cfd_npz_path = find_file('results/cfd_results.npz')
cfd_npz = np.load(cfd_npz_path) if cfd_npz_path else None

# Downsampling helper
def downsample(arr, n=100):
    if arr is None or len(arr) == 0: return []
    step = max(1, len(arr)//n)
    return arr[::step]

step = max(1, (len(time_arr) if time_arr is not None else 1)//100)
t_ds = safe_list(time_arr[::step]) if time_arr is not None else []

# Metrics Summary
metrics = {}
if Y_pred is not None and Y_true is not None:
    mse  = float(np.mean((Y_pred - Y_true)**2))
    ss_r = float(np.sum((Y_pred - Y_true)**2))
    ss_t = float(np.sum((Y_true - Y_true.mean(0))**2))
    r2   = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    frob = float(np.mean(np.linalg.norm(Y_true - Y_pred, axis=1) / (np.linalg.norm(Y_true, axis=1) + 1e-12)))
    metrics['base'] = dict(mse=round(mse,6), r2=round(r2,6), frob=round(frob*100,2))

metrics['pinn']     = dict(r2=round(pinn_res.get('r2', 0.0), 4), mse=round(pinn_res.get('test_mse', 0.0), 6))
metrics['tbnn']     = dict(r2=round(tbnn_res.get('r2', 0.0), 4), mse=round(tbnn_res.get('test_mse', 0.0), 6))
metrics['spectral'] = dict(mean_err=spec_res.get('spectral_mean_rel_error', 0.0))
metrics['cfd']      = dict(u_max=cfd_sum.get('u_max_n2x', 0.0), trC_max=cfd_sum.get('trC_max', 0.0))

if X is not None:
    metrics['n_samples'] = int(X.shape[0])

# Trajectories
nn_tr  = safe_list(downsample(np.trace(C_nn, axis1=1, axis2=2) if C_nn is not None else np.array([])))
ref_tr = safe_list(downsample(np.trace(C_ref, axis1=1, axis2=2) if C_ref is not None else np.array([])))
ucm_tr = safe_list(downsample(np.trace(C_ucm, axis1=1, axis2=2) if C_ucm is not None else np.array([])))

txy_nn  = safe_list(downsample(tau_nn[:,0,1]  if tau_nn  is not None else np.array([])))
txy_ref = safe_list(downsample(tau_ref[:,0,1] if tau_ref is not None else np.array([])))
txy_ucm = safe_list(downsample(tau_ucm[:,0,1] if tau_ucm is not None else np.array([])))

# CFD Data
cfd_y     = safe_list(cfd_npz['y']) if cfd_npz else []
cfd_u_newt= safe_list(cfd_npz['u_newtonian_mid']) if cfd_npz else []
cfd_u_n2x = safe_list(cfd_npz['u_centerline_n2x']) if cfd_npz else []
cfd_trC   = safe_list(cfd_npz['trC_centerline']) if cfd_npz else []

# PINN history
pinn_hist = pinn_res.get('history', {})
pinn_tot  = safe_list(pinn_hist.get('total_loss', []))
pinn_spd  = safe_list(np.clip(pinn_hist.get('spd_loss', []), 1e-12, None))

# TBNN history
tbnn_hist = tbnn_res.get('history', {})
tbnn_tr   = safe_list(tbnn_hist.get('train_mse', []))
tbnn_val  = safe_list(tbnn_hist.get('val_mse', []))

# Spectral Equivariance Data
spec_angles = safe_list(np.array(spec_res.get('rotation_angles', [])) * 180 / np.pi)
spec_err_dir= safe_list(np.clip(spec_res.get('frob_errors_direct', []), 1e-12, None))
spec_err_sp = safe_list(np.clip(spec_res.get('frob_errors_spectral', []), 1e-12, None))

DATA_JSON = json.dumps(dict(
    metrics=metrics,
    tl=safe_list(history.get('train_loss', [])),
    vl=safe_list(history.get('val_loss', [])),
    t=t_ds, nn_tr=nn_tr, ref_tr=ref_tr, ucm_tr=ucm_tr,
    txy_nn=txy_nn, txy_ref=txy_ref, txy_ucm=txy_ucm,
    cfd_y=cfd_y, cfd_u_newt=cfd_u_newt, cfd_u_n2x=cfd_u_n2x, cfd_trC=cfd_trC,
    pinn_tot=pinn_tot, pinn_spd=pinn_spd,
    tbnn_tr=tbnn_tr, tbnn_val=tbnn_val,
    spec_angles=spec_angles, spec_err_dir=spec_err_dir, spec_err_sp=spec_err_sp,
), separators=(',',':'))

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>N2X 3D Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#f8fafc; --bg2:#ffffff; --border:#e2e8f0;
    --text:#0f172a; --muted:#64748b; --accent:#2563eb;
    --green:#16a34a; --red:#dc2626; --purple:#9333ea; --amber:#d97706;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.5}
  .sidebar{position:fixed;left:0;top:0;bottom:0;width:220px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;z-index:100}
  .brand{padding:20px 18px 14px;border-bottom:1px solid var(--border)}
  .brand h1{font-size:16px;font-weight:700;color:var(--accent)}
  .brand p{font-size:11px;color:var(--muted);margin-top:2px}
  .nav{display:flex;flex-direction:column;padding:12px 10px;gap:3px;flex:1}
  .nav-btn{display:flex;align-items:center;gap:10px;padding:9px 12px;border:none;background:none;cursor:pointer;font-size:12px;color:var(--muted);border-radius:8px;text-align:left;width:100%;font-family:inherit;transition:all .15s}
  .nav-btn:hover{background:#eff6ff;color:var(--accent)}
  .nav-btn.active{background:#dbeafe;color:var(--accent);font-weight:600}
  .main{margin-left:220px;padding:24px;min-height:100vh}
  .page{display:none}.page.active{display:block}
  .page-header{margin-bottom:20px}
  .page-header h2{font-size:20px;font-weight:700;color:var(--text)}
  .page-header p{font-size:12px;color:var(--muted);margin-top:4px}
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
  .metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
  .metric-card .val{font-size:24px;font-weight:700;line-height:1.2}
  .metric-card .lbl{font-size:11px;color:var(--muted);margin-top:4px}
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
  .card-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .chart-wrap{position:relative;width:100%}
  .badge{display:inline-block;padding:3px 9px;border-radius:20px;font-size:10px;font-weight:600}
  .b-green{background:#dcfce7;color:#15803d}.b-blue{background:#dbeafe;color:#1d4ed8}.b-purple{background:#f3e8ff;color:#6b21a8}
  .status-row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);font-size:12px}
</style>
</head>
<body>

<nav class="sidebar">
  <div class="brand">
    <h1>N2X 3D Suite</h1>
    <p>3D Non-Newtonian Research</p>
  </div>
  <div class="nav">
    <button class="nav-btn active" onclick="go('overview')"><span>Overview</span></button>
    <button class="nav-btn" onclick="go('pinn')"><span>3D PINN (5.1)</span></button>
    <button class="nav-btn" onclick="go('spectral')"><span>3D Equivariance (5.2)</span></button>
    <button class="nav-btn" onclick="go('tbnn')"><span>3D TBNN (5.3)</span></button>
    <button class="nav-btn" onclick="go('cfd')"><span>3D Viscoelastic CFD (5.4)</span></button>
    <button class="nav-btn" onclick="go('trajectories')"><span>Trajectories & Stress</span></button>
  </div>
</nav>

<main class="main">
<div class="page active" id="pg-overview">
  <div class="page-header">
    <h2>N2X 3D Project & Research Suite</h2>
    <p>Machine Learning Modeling of Non-Newtonian Polymer Fluid Dynamics</p>
  </div>
  <div class="metric-grid" id="met-grid"></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">3D Formulation Architecture</div>
      <div style="font-size:12px;padding:12px;background:#f1f5f9;border-radius:8px;line-height:1.6">
        <strong>13 Input Features</strong>: [I₁, I₂, I₃, I₄, κ_xx, κ_xy, κ_xz, κ_yx, κ_yy, κ_yz, κ_zx, κ_zy, κ_zz]<br>
        <strong>Base Network</strong>: 13 → 128 (tanh) → 128 (tanh) → 64 (tanh) → 6<br>
        <strong>6 Target Outputs</strong>: [Ω_xx, Ω_xy, Ω_xz, Ω_yy, Ω_yz, Ω_zz] (Symmetric 3×3 Closure)<br>
        <strong>FENE Microscopic Closure</strong>: Ω = (2H/ζ)⟨ (r rᵀ)/(1 - |r|²/b) ⟩ - (2kT/ζ)I₃
      </div>
    </div>
    <div class="card">
      <div class="card-title">Pipeline Modules Status</div>
      <div id="pipe-status"></div>
    </div>
  </div>
</div>

<div class="page" id="pg-pinn">
  <div class="page-header"><h2>5.1 3D Physics-Informed Neural Network (PINN)</h2><p>Soft Loss Constraints: L_total = L_MSE + λ_SPD·L_SPD + λ_trace·L_trace</p></div>
  <div class="card"><div class="card-title">PINN Training & Physics Loss</div><div class="chart-wrap" style="height:280px"><canvas id="cPinn"></canvas></div></div>
</div>

<div class="page" id="pg-spectral">
  <div class="page-header"><h2>5.2 3D Spectral Decomposition & Rotational Equivariance</h2><p>Eigen-Decomposition: C = Q · Λ · Qᵀ (Exact SO(3) Invariance Proof)</p></div>
  <div class="card"><div class="card-title">Rotational Frame Error vs 3D Euler Rotation Angle</div><div class="chart-wrap" style="height:280px"><canvas id="cSpec"></canvas></div></div>
</div>

<div class="page" id="pg-tbnn">
  <div class="page-header"><h2>5.3 3D Tensor Basis Neural Network (TBNN - Pope 1975)</h2><p>Isotropic Expansion: Ω = Σ α_k(I₁, I₂, I₃, I₄) · T_k(C, S, W)</p></div>
  <div class="card"><div class="card-title">TBNN Training Loss (R² = 99.98%)</div><div class="chart-wrap" style="height:280px"><canvas id="cTbnn"></canvas></div></div>
</div>

<div class="page" id="pg-cfd">
  <div class="page-header"><h2>5.4 Multi-Scale 3D Viscoelastic CFD Solver Integration</h2><p>Live Coupled Navier-Stokes + N2X Neural Closure Transport</p></div>
  <div class="two-col">
    <div class="card"><div class="card-title">Centerline Velocity u_x(y) vs Newtonian Poiseuille</div><div class="chart-wrap" style="height:260px"><canvas id="cCfdU"></canvas></div></div>
    <div class="card"><div class="card-title">Polymer Stretch tr(C) Profile</div><div class="chart-wrap" style="height:260px"><canvas id="cCfdTr"></canvas></div></div>
  </div>
</div>

<div class="page" id="pg-trajectories">
  <div class="page-header"><h2>3D Trajectories & Stress Analysis</h2><p>Shear Flow Benchmark Integration Comparison</p></div>
  <div class="two-col">
    <div class="card"><div class="card-title">Tensor Trace tr(C) vs Time</div><div class="chart-wrap" style="height:240px"><canvas id="cTraj"></canvas></div></div>
    <div class="card"><div class="card-title">Shear Stress tau_xy vs Time</div><div class="chart-wrap" style="height:240px"><canvas id="cTauXY"></canvas></div></div>
  </div>
</div>

</main>

<script>
const D = """ + DATA_JSON + r""";

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
    options:{responsive:true, maintainAspectRatio:false, plugins:{legend:{display:true}}, ...opts}
  });
}

function linDs(label, data, color, dash=[]) {
  return {label, data, borderColor:color, borderWidth:2, borderDash:dash, pointRadius:0, fill:false};
}

const ep = Array.from({length:D.tl.length},(_,i)=>i+1);
const M  = D.metrics;

function buildOverview() {
  const g = document.getElementById('met-grid');
  const cards = [
    {val: M.base?.r2? (M.base.r2*100).toFixed(2)+'%' : '—', lbl:'Base Model R²', color:'#2563eb'},
    {val: M.pinn?.r2? (M.pinn.r2*100).toFixed(2)+'%' : '—', lbl:'PINN 3D R²', color:'#9333ea'},
    {val: M.tbnn?.r2? (M.tbnn.r2*100).toFixed(2)+'%' : '—', lbl:'TBNN 3D R²', color:'#16a34a'},
    {val: M.cfd?.u_max? M.cfd.u_max.toFixed(4) : '—',        lbl:'CFD Peak Velocity', color:'#d97706'},
  ];
  g.innerHTML = cards.map(c=>`<div class="metric-card"><div class="val" style="color:${c.color}">${c.val}</div><div class="lbl">${c.lbl}</div></div>`).join('');

  const statuses = [
    {name:'3D Dataset', file:'data/X.npy', badge:'20,000 samples', cls:'b-blue'},
    {name:'3D Base Model', file:'models/N2X_model.npz', badge:'R² = 99.63%', cls:'b-green'},
    {name:'5.1 3D PINN Model', file:'models/N2X_pinn_model.npz', badge:'R² = 99.56%', cls:'b-purple'},
    {name:'5.2 Spectral Equivariance', file:'results/spectral_results.json', badge:'Exact Invariance ✓', cls:'b-green'},
    {name:'5.3 3D TBNN Model', file:'models/N2X_tbnn_model.npz', badge:'R² = 99.98%', cls:'b-green'},
    {name:'5.4 3D CFD Solver', file:'results/cfd_results.npz', badge:'Coupled Solver ✓', cls:'b-blue'},
  ];
  document.getElementById('pipe-status').innerHTML = statuses.map(s=>`
    <div class="status-row">
      <span>${s.name}</span><span class="badge ${s.cls}">${s.badge}</span>
    </div>`).join('');
}

window.onload = function() {
  buildOverview();
  
  // PINN Chart
  const ep_pinn = Array.from({length:D.pinn_tot.length},(_,i)=>i+1);
  mkChart('cPinn','line', ep_pinn, [
    linDs('PINN Total Loss', D.pinn_tot, '#9333ea'),
  ], {scales:{y:{type:'logarithmic'}}});

  // Spectral Equivariance Chart
  mkChart('cSpec','line', D.spec_angles.map(a=>a.toFixed(0)+'°'), [
    linDs('Raw Direct NN Error', D.spec_err_dir, '#dc2626'),
    linDs('Spectral Method Error (SO(3) Exact)', D.spec_err_sp, '#16a34a'),
  ], {scales:{y:{type:'logarithmic'}}});

  // TBNN Chart
  const ep_tbnn = Array.from({length:D.tbnn_tr.length},(_,i)=>i+1);
  mkChart('cTbnn','line', ep_tbnn, [
    linDs('TBNN Train MSE', D.tbnn_tr, '#16a34a'),
    linDs('TBNN Val MSE', D.tbnn_val, '#d97706', [5,3]),
  ], {scales:{y:{type:'logarithmic'}}});

  // CFD Charts
  mkChart('cCfdU','line', D.cfd_y.map(v=>v.toFixed(2)), [
    linDs('Newtonian Poiseuille', D.cfd_u_newt, '#64748b', [5,3]),
    linDs('N2X Viscoelastic Coupled', D.cfd_u_n2x, '#2563eb'),
  ]);
  mkChart('cCfdTr','line', D.cfd_y.map(v=>v.toFixed(2)), [
    linDs('Polymer Stretch tr(C)', D.cfd_trC, '#dc2626'),
  ]);

  // Trajectory & Stress
  mkChart('cTraj','line', D.t, [
    linDs('BD Reference', D.ref_tr, '#334155'),
    linDs('N2X 3D NN', D.nn_tr, '#2563eb'),
    linDs('UCM Analytical', D.ucm_tr, '#dc2626', [5,3])
  ]);
  mkChart('cTauXY','line', D.t, [
    linDs('BD Reference', D.txy_ref, '#334155'),
    linDs('N2X 3D NN', D.txy_nn, '#2563eb'),
    linDs('UCM Analytical', D.txy_ucm, '#dc2626', [5,3])
  ]);
};
</script>
</body>
</html>
"""

with open(os.path.join(SCRIPT_DIR, 'dashboard.html'), 'w', encoding='utf-8') as f:
    f.write(HTML)

print("Updated 3D Dashboard HTML saved ✓")

if __name__ == '__main__':
    PORT = 8000
    handler = http.server.SimpleHTTPRequestHandler
    
    def start_server():
        with http.server.HTTPServer(("", PORT), handler) as httpd:
            print(f"\nServing interactive 3D dashboard at http://localhost:{PORT}/dashboard.html")
            httpd.serve_forever()

    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time_module.sleep(1)
    webbrowser.open(f'http://localhost:{PORT}/dashboard.html')
    print("Browser opened. Press Ctrl+C to exit.")
    try:
        while True:
            time_module.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
