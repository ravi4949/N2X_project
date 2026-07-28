# -*- coding: utf-8 -*-
"""
==============================================================
N2X 3D - INTERACTIVE DASHBOARD
==============================================================
Connects to your saved .npy and .json result files and opens
a live interactive 3D dashboard in your web browser.

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

def safe_list(arr, decimals=4):
    if arr is None:
        return []
    return [round(float(v), decimals) for v in arr.flatten()]

print("Loading 3D data files...")

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
    metrics['n_features'] = int(X.shape[1])
if Y is not None:
    metrics['n_targets']  = int(Y.shape[1])

def tr(C):
    if C is None or C.size == 0: return []
    if C.ndim == 3 and C.shape[1] == 3:
        return safe_list(C[:,0,0] + C[:,1,1] + C[:,2,2])
    elif C.ndim == 3 and C.shape[1] == 2:
        return safe_list(C[:,0,0] + C[:,1,1])
    return safe_list(C)

def downsample(arr, n=100):
    if arr is None or len(arr) == 0: return []
    step = max(1, len(arr)//n)
    return arr[::step]

tl_ds = safe_list(np.array(history.get('train_loss', [])))
vl_ds = safe_list(np.array(history.get('val_loss',   [])))

step = max(1, (len(time_arr) if time_arr is not None else 1)//100)
t_ds    = safe_list(time_arr[::step]) if time_arr is not None else []

nn_tr   = safe_list(downsample(np.trace(C_nn, axis1=1, axis2=2) if C_nn is not None else np.array([])))
ref_tr  = safe_list(downsample(np.trace(C_ref, axis1=1, axis2=2) if C_ref is not None else np.array([])))
ucm_tr  = safe_list(downsample(np.trace(C_ucm, axis1=1, axis2=2) if C_ucm is not None else np.array([])))
fp_tr   = safe_list(downsample(np.trace(C_fp, axis1=1, axis2=2) if C_fp is not None else np.array([])))

txy_nn  = safe_list(downsample(tau_nn[:,0,1]  if tau_nn  is not None else np.array([])))
txy_ref = safe_list(downsample(tau_ref[:,0,1] if tau_ref is not None else np.array([])))
txy_ucm = safe_list(downsample(tau_ucm[:,0,1] if tau_ucm is not None else np.array([])))
txx_nn  = safe_list(downsample(tau_nn[:,0,0]  if tau_nn  is not None else np.array([])))
txx_ref = safe_list(downsample(tau_ref[:,0,0] if tau_ref is not None else np.array([])))
txx_ucm = safe_list(downsample(tau_ucm[:,0,0] if tau_ucm is not None else np.array([])))

parity = {'x0':[], 'y0':[], 'x1':[], 'y1':[]}
if Y_pred is not None and Y_true is not None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Y_pred), min(500, len(Y_pred)), replace=False)
    parity = dict(
        x0=safe_list(Y_true[idx,0]), y0=safe_list(Y_pred[idx,0]),
        x1=safe_list(Y_true[idx,1]), y1=safe_list(Y_pred[idx,1]),
    )

hist_data = {'x':[], 'y':[]}
if Y_pred is not None and Y_true is not None:
    err = (Y_pred - Y_true).flatten()
    counts, edges = np.histogram(err, bins=40)
    hist_data = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 3),
        y=counts.tolist()
    )

kappa_hist = {'x':[], 'y':[]}
if kappa is not None:
    kxy = kappa[:,1] if kappa.ndim == 2 else kappa[:,0,1]
    counts, edges = np.histogram(kxy, bins=30)
    kappa_hist = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 3),
        y=counts.tolist()
    )

cxx_hist = {'x':[], 'y':[]}
if C_now is not None:
    cxx = C_now[:,0]
    counts, edges = np.histogram(cxx, bins=30, range=(0, float(cxx.max().clip(max=15))))
    cxx_hist = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 3),
        y=counts.tolist()
    )

feat_std = safe_list(X.std(axis=0)) if X is not None else [0]*13

frob_hist = {'x':[], 'y':[]}
if Y_pred is not None and Y_true is not None:
    fv = np.linalg.norm(Y_true-Y_pred,axis=1)/(np.linalg.norm(Y_true,axis=1)+1e-12)*100
    counts, edges = np.histogram(fv, bins=30, range=(0, min(300, float(fv.max()))))
    frob_hist = dict(
        x=safe_list((edges[:-1]+edges[1:])/2, 2),
        y=counts.tolist()
    )

print("\nBuilding 3D dashboard HTML...")

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
<title>N2X 3D Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#f7f8fa; --bg2:#fff; --border:#e2e4e8;
    --text:#1a1a2e; --muted:#6b7280; --accent:#2563eb;
    --green:#16a34a; --red:#dc2626; --amber:#d97706; --purple:#7c3aed;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.5}
  .sidebar{position:fixed;left:0;top:0;bottom:0;width:200px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;z-index:100;padding:0}
  .brand{padding:18px 16px 14px;border-bottom:1px solid var(--border)}
  .brand h1{font-size:15px;font-weight:700;color:var(--accent)}
  .brand p{font-size:10px;color:var(--muted);margin-top:2px}
  .nav{display:flex;flex-direction:column;padding:10px 8px;gap:2px;flex:1}
  .nav-btn{display:flex;align-items:center;gap:9px;padding:8px 10px;border:none;background:none;cursor:pointer;font-size:12px;color:var(--muted);border-radius:7px;text-align:left;width:100%;font-family:inherit;transition:all .15s}
  .nav-btn:hover{background:#f0f4ff;color:var(--accent)}
  .nav-btn.active{background:#eff6ff;color:var(--accent);font-weight:500}
  .main{margin-left:200px;padding:20px;min-height:100vh}
  .page{display:none}.page.active{display:block}
  .page-header{margin-bottom:18px}
  .page-header h2{font-size:18px;font-weight:600}
  .page-header p{font-size:12px;color:var(--muted);margin-top:3px}
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:18px}
  .metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px}
  .metric-card .val{font-size:22px;font-weight:700;line-height:1.2}
  .metric-card .lbl{font-size:11px;color:var(--muted);margin-top:3px}
  .metric-card .sub{font-size:10px;margin-top:2px}
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:14px}
  .card-title{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .chart-wrap{position:relative;width:100%}
  .badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600}
  .b-blue{background:#dbeafe;color:#1e40af}.b-green{background:#dcfce7;color:#166534}
  .b-red{background:#fee2e2;color:#991b1b}.b-amber{background:#fef3c7;color:#92400e}
  .prog-row{display:flex;flex-direction:column;gap:4px;margin:6px 0}
  .prog-bar{height:6px;border-radius:3px;background:#f1f5f9;overflow:hidden}
  .prog-fill{height:100%;border-radius:3px;transition:width .6s ease}
  .status-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px}
  .status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
</style>
</head>
<body>

<nav class="sidebar">
  <div class="brand">
    <h1>N2X 3D</h1>
    <p>3D Polymer Dashboard</p>
  </div>
  <div class="nav">
    <button class="nav-btn active" onclick="go('overview')"><span>Overview</span></button>
    <button class="nav-btn" onclick="go('dataset')"><span>Dataset</span></button>
    <button class="nav-btn" onclick="go('training')"><span>Training</span></button>
    <button class="nav-btn" onclick="go('evaluation')"><span>Evaluation</span></button>
    <button class="nav-btn" onclick="go('trajectory')"><span>Trajectories</span></button>
    <button class="nav-btn" onclick="go('stress')"><span>Stress</span></button>
  </div>
</nav>

<main class="main">
<div class="page active" id="pg-overview">
  <div class="page-header">
    <h2>N2X 3D Project Overview</h2>
    <p>3D Neural Network Closure Model for FENE Polymer Fluids</p>
  </div>
  <div class="metric-grid" id="met-grid"></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">3D Architecture</div>
      <div style="font-size:12px;padding:10px;background:#f8fafc;border-radius:7px">
        <strong>Input</strong>: 13 features [I1, I2, I3, I4, κxx, κxy, κxz, κyx, κyy, κyz, κzx, κzy, κzz]<br>
        <strong>Hidden</strong>: 64 → 64 → 32 (tanh activations)<br>
        <strong>Output</strong>: 6 targets [Ω_xx, Ω_xy, Ω_xz, Ω_yy, Ω_yz, Ω_zz]
      </div>
    </div>
    <div class="card">
      <div class="card-title">Pipeline Status</div>
      <div id="pipe-status"></div>
    </div>
  </div>
</div>

<div class="page" id="pg-dataset">
  <div class="page-header"><h2>3D Dataset Explorer</h2></div>
  <div class="card"><div class="card-title">Cxx Distribution</div><div class="chart-wrap" style="height:200px"><canvas id="cCxx"></canvas></div></div>
</div>

<div class="page" id="pg-training">
  <div class="page-header"><h2>3D Training Monitor</h2></div>
  <div class="card"><div class="card-title">Train vs Val MSE</div><div class="chart-wrap" style="height:240px"><canvas id="cTrain"></canvas></div></div>
</div>

<div class="page" id="pg-evaluation">
  <div class="page-header"><h2>3D Model Evaluation</h2></div>
  <div class="card"><div class="card-title">Omega_xx Parity</div><div class="chart-wrap" style="height:220px"><canvas id="cParity0"></canvas></div></div>
</div>

<div class="page" id="pg-trajectory">
  <div class="page-header"><h2>3D Tensor Trajectories</h2></div>
  <div class="card"><div class="card-title">tr(C) = Cxx + Cyy + Czz</div><div class="chart-wrap" style="height:260px"><canvas id="cTraj"></canvas></div></div>
</div>

<div class="page" id="pg-stress">
  <div class="page-header"><h2>3D Stress Evolution</h2></div>
  <div class="card"><div class="card-title">Shear Stress tau_xy</div><div class="chart-wrap" style="height:240px"><canvas id="cTauXY"></canvas></div></div>
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
    options:{responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, ...opts}
  });
}

function linDs(label, data, color, dash=[]) {
  return {label, data, borderColor:color, borderWidth:2, borderDash:dash, pointRadius:0, fill:false};
}

function barDs(label, data, color) {
  return {label, data, backgroundColor:color+'cc', borderColor:color, borderWidth:1};
}

const ep = Array.from({length:D.tl.length},(_,i)=>i+1);
const M  = D.metrics;

function buildOverview() {
  const g = document.getElementById('met-grid');
  const cards = [
    {val: M.n_samples?.toLocaleString()||'—', lbl:'Training samples', color:'#2563eb'},
    {val: M.r2?.toFixed(4)||'—',             lbl:'R² score (3D)',     color:'#16a34a'},
    {val: M.mse?.toFixed(4)||'—',            lbl:'Test MSE',          color:'#d97706'},
    {val: (M.frob?.toFixed(1)||'—')+'%',     lbl:'Frobenius error',   color:'#7c3aed'},
  ];
  g.innerHTML = cards.map(c=>`<div class="metric-card"><div class="val" style="color:${c.color}">${c.val}</div><div class="lbl">${c.lbl}</div></div>`).join('');

  const statuses = [
    {name:'3D Dataset', file:'data/X.npy', done: D.feat_std.length>0},
    {name:'3D Training', file:'results/history.json', done: D.tl.length>0},
    {name:'3D Evaluation', file:'results/Y_pred.npy', done: D.parity.x0?.length>0},
    {name:'3D Trajectories', file:'results/C_nn.npy', done: D.nn_tr.length>0},
  ];
  document.getElementById('pipe-status').innerHTML = statuses.map(s=>`
    <div class="status-row">
      <span>${s.name}</span><span class="badge ${s.done?'b-green':'b-red'}">${s.done?'loaded':'missing'}</span>
    </div>`).join('');
}

window.onload = function() {
  buildOverview();
  mkChart('cCxx','bar', D.cxx_hist.x, [barDs('Cxx',D.cxx_hist.y,'#2563eb')]);
  mkChart('cTrain','line', ep, [linDs('Train',D.tl,'#7c3aed'), linDs('Val',D.vl,'#f59e0b',[5,3])]);
  mkChart('cTraj','line', D.t, [linDs('BD ref',D.ref_tr,'#374151'), linDs('N2X 3D',D.nn_tr,'#2563eb'), linDs('UCM',D.ucm_tr,'#dc2626',[5,3])]);
  mkChart('cTauXY','line', D.t, [linDs('BD ref',D.txy_ref,'#374151'), linDs('N2X 3D',D.txy_nn,'#2563eb'), linDs('UCM',D.txy_ucm,'#dc2626',[5,3])]);
};
</script>
</body>
</html>
"""

with open(os.path.join(SCRIPT_DIR, 'dashboard.html'), 'w', encoding='utf-8') as f:
    f.write(HTML)

print("Dashboard HTML saved ✓")

if __name__ == '__main__':
    PORT = 8000
    handler = http.server.SimpleHTTPRequestHandler
    
    def start_server():
        with http.server.HTTPServer(("", PORT), handler) as httpd:
            print(f"\nServing 3D dashboard at http://localhost:{PORT}/dashboard.html")
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
