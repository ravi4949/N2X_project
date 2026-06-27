"""
================================================================================
N2X Model — MASTER PIPELINE
================================================================================
Runs the complete pipeline end-to-end:
    Step 1: Generate dataset (Brownian dynamics simulation)
    Step 2: Train neural network (N2X closure model)
    Step 3: Evaluate & compare with UCM / FENE-P
    Step 4: Generate all plots

Run: python master_pipeline.py
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import os, sys, time, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*70)
print("  N2X — Machine Learning Based Modeling of Non-Newtonian Fluids")
print("="*70)

os.makedirs('data',    exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class P:   # DumbbellParams
    H=1.0; b=50.0; kT=1.0; zeta=1.0
    tau=zeta/(4*H); D=kT/zeta

def fene_force(r):
    r2=np.sum(r**2,axis=1,keepdims=True)
    return P.H*r/np.maximum(1-r2/P.b, 1e-4)

def euler_step(r, kappa, dt, rng):
    F=fene_force(r)
    drift=(r@kappa.T)-F/P.zeta
    noise=np.sqrt(2*P.kT/P.zeta*dt)*rng.standard_normal(r.shape)
    return r+drift*dt+noise

def conf_tensor(r):
    return np.einsum('ni,nj->ij',r,r)/r.shape[0]

def invariants(C):
    return np.array([np.trace(C), np.linalg.det(C), np.trace(C@C)])

def ucm_closure(C):
    return (C - np.eye(2)) / P.tau

def fenep_closure(C):
    trC=np.trace(C); d=max(1-trC/P.b, 1e-6)
    return (4*P.H/P.zeta)*C/d - 2*P.kT/P.zeta*np.eye(2)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  STEP 1: Generating dataset from Brownian dynamics …")
print("─"*60)
t0=time.time()

N_TRAJ      = 200          # number of trajectories
N_DUMBBELLS = 1000         # ensemble size (reduce for speed)
N_STEPS     = 100          # time-steps per trajectory
DT          = 0.01
SEED        = 42

rng_main = np.random.default_rng(SEED)

features_list, targets_list  = [], []
C_now_list,  C_next_list     = [], []
kappa_list                   = []

for traj in range(N_TRAJ):
    # Random flow type
    flow = rng_main.choice(['shear','extension','random'])
    scale= float(rng_main.uniform(0.05, 1.5))
    if   flow=='shear':
        kappa=np.array([[0.,scale],[0.,0.]])
    elif flow=='extension':
        kappa=np.array([[scale,0.],[0.,-scale]])
    else:
        kappa=rng_main.uniform(-scale*0.5, scale*0.5,(2,2))

    # Initialise
    r = rng_main.normal(0, np.sqrt(P.kT/P.H),(N_DUMBBELLS,2))
    C = conf_tensor(r)

    for step in range(N_STEPS):
        C_old = conf_tensor(r)
        r     = euler_step(r, kappa, DT, rng_main)
        C_new = conf_tensor(r)

        # closure from evolution equation
        transport = kappa@C_old + C_old@kappa.T
        dC_dt     = (C_new - C_old)/DT
        Omega     = transport - dC_dt

        inv    = invariants(C_old)
        kflat  = kappa.flatten()
        features_list.append(np.concatenate([inv, kflat]).astype(np.float32))
        targets_list.append(np.array([Omega[0,0],Omega[0,1],Omega[1,1]],dtype=np.float32))
        C_now_list.append(C_old.flatten().astype(np.float32))
        C_next_list.append(C_new.flatten().astype(np.float32))
        kappa_list.append(kflat.astype(np.float32))

    if (traj+1) % 50 == 0:
        print(f"    Trajectory {traj+1}/{N_TRAJ}  ({len(features_list):,} samples so far)")

X = np.array(features_list)
Y = np.array(targets_list)
np.save('data/X.npy', X)
np.save('data/Y.npy', Y)
np.save('data/C_now.npy',  np.array(C_now_list))
np.save('data/C_next.npy', np.array(C_next_list))
np.save('data/kappa.npy',  np.array(kappa_list))

print(f"  Dataset: {X.shape[0]:,} samples  | Time: {time.time()-t0:.1f}s")
print(f"  X shape: {X.shape}  |  Y shape: {Y.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — NEURAL NETWORK TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  STEP 2: Training N2X neural network …")
print("─"*60)

# ── Preprocessing ─────────────────────────────────────────────────────────────
X64 = X.astype(np.float64)
Y64 = Y.astype(np.float64)

sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X64)
Ys = sY.fit_transform(Y64)

np.save('models/sX_mean.npy', sX.mean_);  np.save('models/sX_std.npy', sX.scale_)
np.save('models/sY_mean.npy', sY.mean_);  np.save('models/sY_std.npy', sY.scale_)

X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(Xs, Ys, test_size=0.20, random_state=7)
X_val, X_te,  Y_val, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=7)
print(f"  Train:{len(X_tr):,}  Val:{len(X_val):,}  Test:{len(X_te):,}")

# ── Model Definition ──────────────────────────────────────────────────────────
class Layer:
    def __init__(self, ni, no, act, rng):
        sc=np.sqrt(2./(ni+no))
        self.W=rng.normal(0,sc,(ni,no)); self.b=np.zeros(no)
        self.act=act
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
        self.dW=np.zeros_like(self.W); self.db=np.zeros_like(self.b)

    def fwd(self, x):
        self._x=x; self._z=x@self.W+self.b
        self._a=np.tanh(self._z) if self.act=='tanh' else self._z
        return self._a

    def bwd(self, g):
        d=(1-self._a**2)*g if self.act=='tanh' else g
        self.dW=self._x.T@d/len(self._x)
        self.db=d.mean(0)
        return d@self.W.T

class Net:
    def __init__(self):
        rng=np.random.default_rng(42)
        self.layers=[Layer(7,64,'tanh',rng),
                     Layer(64,64,'tanh',rng),
                     Layer(64,32,'tanh',rng),
                     Layer(32,3,'none',rng)]
        self.t=0

    def fwd(self,x):
        for l in self.layers: x=l.fwd(x)
        return x

    def bwd(self,g):
        for l in reversed(self.layers): g=l.bwd(g)

    def adam(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t+=1
        for l in self.layers:
            for p,g,m,v in [(l.W,l.dW,l.mW,l.vW),(l.b,l.db,l.mb,l.vb)]:
                m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
                p -= lr*(m/(1-b1**self.t))/(np.sqrt(v/(1-b2**self.t))+eps)

    def save(self):
        d={f'l{i}W':l.W for i,l in enumerate(self.layers)}
        d.update({f'l{i}b':l.b for i,l in enumerate(self.layers)})
        np.savez('models/N2X.npz',**d)

# ── Training Loop ─────────────────────────────────────────────────────────────
model   = Net()
LR      = 5e-4
EPOCHS  = 250
BATCH   = 512
PATIENCE= 25
history = {'train':[], 'val':[]}
best_val= np.inf; best_w=None; wait=0
rng_tr  = np.random.default_rng(99)

t0=time.time()
print(f"\n  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}")
print(f"  {'-'*35}")

for epoch in range(EPOCHS):
    idx=rng_tr.permutation(len(X_tr))
    batch_losses=[]
    for s in range(0,len(X_tr),BATCH):
        bi=idx[s:s+BATCH]
        xb,yb=X_tr[bi],Y_tr[bi]
        pred=model.fwd(xb)
        loss=float(np.mean((pred-yb)**2))
        model.bwd(2*(pred-yb)/len(xb))
        model.adam(LR)
        batch_losses.append(loss)

    if (epoch+1)%10==0: LR*=0.97

    val_pred=model.fwd(X_val)
    val_mse =float(np.mean((val_pred-Y_val)**2))
    tr_mse  =float(np.mean(batch_losses))
    history['train'].append(tr_mse)
    history['val'].append(val_mse)

    if val_mse<best_val:
        best_val=val_mse; best_e=epoch
        best_w=[(l.W.copy(),l.b.copy()) for l in model.layers]
        wait=0
    else:
        wait+=1
        if wait>=PATIENCE:
            print(f"\n  Early stop epoch {epoch+1}  best_val={best_val:.6f} @ epoch {best_e+1}")
            break

    if (epoch+1)%20==0 or epoch==0:
        print(f"  {epoch+1:>6}  {tr_mse:>10.6f}  {val_mse:>10.6f}")

# Restore best
for i,(W,b) in enumerate(best_w):
    model.layers[i].W=W; model.layers[i].b=b
model.save()
print(f"\n  Training time: {time.time()-t0:.1f}s  |  Best val MSE: {best_val:.6f}")

with open('results/history.json','w') as f:
    json.dump({'train_loss':history['train'],'val_loss':history['val']}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  STEP 3: Evaluation …")
print("─"*60)

Y_pred_s = model.fwd(X_te)
Y_pred   = sY.inverse_transform(Y_pred_s)
Y_true   = sY.inverse_transform(Y_te)

mse  = float(np.mean((Y_pred-Y_true)**2))
r2   = float(1-np.sum((Y_true-Y_pred)**2)/np.sum((Y_true-Y_true.mean(0))**2))
frob = float(np.mean(np.linalg.norm(Y_true-Y_pred,axis=1)/(np.linalg.norm(Y_true,axis=1)+1e-12)))

print(f"  MSE              : {mse:.6f}")
print(f"  R²               : {r2:.6f}")
print(f"  Rel. Frob. Error : {frob*100:.3f}%")

np.save('results/Y_pred.npy', Y_pred)
np.save('results/Y_true.npy', Y_true)
np.save('results/X_test.npy', sX.inverse_transform(X_te))


# ── Forward integration comparison ───────────────────────────────────────────
def nn_closure(C, kappa):
    inv  =invariants(C); kf=kappa.flatten()
    feat =np.concatenate([inv,kf]).reshape(1,-1)
    feats=(feat-sX.mean_)/sX.scale_
    out  =model.fwd(feats)[0]
    return out*sY.scale_+sY.mean_

def integrate(clf, kappa, C0, dt, steps):
    traj=np.zeros((steps+1,2,2)); traj[0]=C0; C=C0.copy()
    for s in range(steps):
        Omega=clf(C,kappa)
        if len(Omega)==3:
            Om=np.array([[Omega[0],Omega[1]],[Omega[1],Omega[2]]])
        else:
            Om=Omega
        dC=kappa@C+C@kappa.T-Om
        C=0.5*(C+C.T)+dC*dt
        C=0.5*(C+C.T)
        traj[s+1]=C
    return traj

# Reference simulation
kappa_test = np.array([[0.,1.],[0.,0.]])   # shear γ̇=1
rng_ref    = np.random.default_rng(777)
r_ref      = rng_ref.normal(0,np.sqrt(P.kT/P.H),(3000,2))
N_EVAL     = 400
DT_EVAL    = 0.005
C_ref_traj = np.zeros((N_EVAL+1,2,2))
C_ref_traj[0]=conf_tensor(r_ref)
for s in range(N_EVAL):
    r_ref=euler_step(r_ref,kappa_test,DT_EVAL,rng_ref)
    C_ref_traj[s+1]=conf_tensor(r_ref)

C0_eval   = np.eye(2)
C_nn_traj = integrate(nn_closure,  kappa_test, C0_eval, DT_EVAL, N_EVAL)
C_ucm_traj= integrate(lambda C,k: ucm_closure(C), kappa_test, C0_eval, DT_EVAL, N_EVAL)
C_fp_traj = integrate(lambda C,k: fenep_closure(C), kappa_test, C0_eval, DT_EVAL, N_EVAL)

time_arr = np.arange(N_EVAL+1)*DT_EVAL
H        = P.H
tau_ref  = H*C_ref_traj  - np.eye(2)
tau_nn   = H*C_nn_traj   - np.eye(2)
tau_ucm  = H*C_ucm_traj  - np.eye(2)

np.save('results/time.npy',    time_arr)
np.save('results/C_ref.npy',   C_ref_traj)
np.save('results/C_nn.npy',    C_nn_traj)
np.save('results/C_ucm.npy',   C_ucm_traj)
np.save('results/C_fenep.npy', C_fp_traj)
np.save('results/tau_ref.npy', tau_ref)
np.save('results/tau_nn.npy',  tau_nn)
np.save('results/tau_ucm.npy', tau_ucm)

# Trajectory metrics
def tr_mse(Ca, Cb):
    return float(np.mean((np.trace(Ca,axis1=1,axis2=2)
                          - np.trace(Cb,axis1=1,axis2=2))**2))

print(f"\n  Trajectory tr(C) MSE vs Reference:")
print(f"    N2X NN : {tr_mse(C_nn_traj,  C_ref_traj):.5f}")
print(f"    UCM       : {tr_mse(C_ucm_traj, C_ref_traj):.5f}")
print(f"    FENE-P    : {tr_mse(C_fp_traj,  C_ref_traj):.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  STEP 4: Generating plots …")
print("─"*60)

plt.rcParams.update({
    'font.family':'DejaVu Serif','font.size':11,'axes.labelsize':12,
    'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':2.0,
    'legend.framealpha':0.8,'figure.dpi':110,'savefig.dpi':140,
    'savefig.bbox':'tight',
})
COL={'nn':'#2196F3','ucm':'#F44336','fenep':'#4CAF50',
     'ref':'#212121','train':'#9C27B0','val':'#FF9800'}

epochs_arr = np.arange(1,len(history['train'])+1)


# ── Fig 1: Training loss ───────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,scale in zip(axes,['linear','log']):
    fn=ax.semilogy if scale=='log' else ax.plot
    fn(epochs_arr,history['train'],color=COL['train'],lw=2,label='Train')
    fn(epochs_arr,history['val'],  color=COL['val'],  lw=2,ls='--',label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title(f'Training Loss ({scale} scale)'); ax.legend()
    best_e2=int(np.argmin(history['val']))
    ax.axvline(best_e2+1,color='gray',ls=':',lw=1.2)
fig.suptitle('N2X — Training Curve',fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig('results/plots/fig1_training_loss.png'); plt.close()

# ── Fig 2: Parity plots ────────────────────────────────────────────────────
lbls=[r'$\Omega_{xx}$',r'$\Omega_{xy}$',r'$\Omega_{yy}$']
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for i,(ax,lb) in enumerate(zip(axes,lbls)):
    yt,yp=Y_true[:,i],Y_pred[:,i]
    lim=[min(yt.min(),yp.min()),max(yt.max(),yp.max())]
    ax.scatter(yt,yp,alpha=0.1,s=3,color=COL['nn'],rasterized=True)
    ax.plot(lim,lim,'k--',lw=1.5,label='y=x')
    r2_i=1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
    ax.text(0.06,0.90,f'R²={r2_i:.4f}',transform=ax.transAxes,fontsize=10)
    ax.set_xlabel(f'True {lb}'); ax.set_ylabel(f'Pred {lb}')
    ax.set_title(f'Parity — {lb}'); ax.legend(fontsize=9)
fig.suptitle('N2X — Parity Plots',fontweight='bold',y=1.01)
plt.tight_layout(); plt.savefig('results/plots/fig2_parity.png'); plt.close()

# ── Fig 3: Tensor trace ────────────────────────────────────────────────────
def tr(C): return np.trace(C,axis1=1,axis2=2)
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(time_arr,tr(C_ref_traj), color=COL['ref'],  lw=2.5, label='BD Reference')
ax.plot(time_arr,tr(C_nn_traj),  color=COL['nn'],   lw=2,   label='N2X NN')
ax.plot(time_arr,tr(C_ucm_traj), color=COL['ucm'],  lw=2,   ls='--',label='UCM')
ax.plot(time_arr,tr(C_fp_traj),  color=COL['fenep'],lw=2,   ls=':',  label='FENE-P')
ax.set_xlabel('Time'); ax.set_ylabel('tr(C)')
ax.set_title(r'Conformation Tensor Trace  tr$(C)$  — Shear $\dot\gamma=1$'); ax.legend()
plt.tight_layout(); plt.savefig('results/plots/fig3_tensor_trace.png'); plt.close()

# ── Fig 4 & 5: Stresses ────────────────────────────────────────────────────
for comp,idx_,nm in [((0,1),r'\tau_{xy}','shear'),((0,0),r'\tau_{xx}','normal')]:
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(time_arr,tau_ref[:,comp[0],comp[1]], color=COL['ref'],lw=2.5,label='BD Reference')
    ax.plot(time_arr,tau_nn[:,comp[0],comp[1]],  color=COL['nn'], lw=2,  label='N2X NN')
    ax.plot(time_arr,tau_ucm[:,comp[0],comp[1]], color=COL['ucm'],lw=2,  ls='--',label='UCM')
    ax.set_xlabel('Time'); ax.set_ylabel(f'{nm} stress')
    ax.set_title(f'Polymer Stress ({nm})  gdot=1'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/fig4_stress_{nm}.png'); plt.close()

# ── Fig 5: Error histogram ────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
err=(Y_pred-Y_true).flatten()
axes[0].hist(err,bins=80,color=COL['nn'],alpha=0.8,edgecolor='white')
axes[0].axvline(0,color='black',lw=1.5,ls='--')
axes[0].set_xlabel('Error'); axes[0].set_ylabel('Count')
axes[0].set_title('Closure Error Distribution')
mu,sig=err.mean(),err.std()
axes[0].text(0.72,0.90,f'μ={mu:.4f}\nσ={sig:.4f}',
             transform=axes[0].transAxes,fontsize=10,
             bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))
frob_v=np.linalg.norm(Y_true-Y_pred,axis=1)/(np.linalg.norm(Y_true,axis=1)+1e-12)
axes[1].hist(frob_v*100,bins=60,color=COL['fenep'],alpha=0.8,edgecolor='white')
axes[1].set_xlabel('Relative Frobenius Error (%)'); axes[1].set_ylabel('Count')
med_frob=np.median(frob_v)*100
axes[1].axvline(med_frob,color='red',lw=1.5,ls='--',label=f'Median={med_frob:.1f}%')
axes[1].legend(); axes[1].set_title('Frobenius Error Distribution')
fig.suptitle('N2X — Error Analysis',fontweight='bold',y=1.01)
plt.tight_layout(); plt.savefig('results/plots/fig5_errors.png'); plt.close()

# ── Fig 6: Dashboard ─────────────────────────────────────────────────────
import matplotlib.gridspec as gridspec
fig=plt.figure(figsize=(16,10))
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35)

ax=fig.add_subplot(gs[0,0])
ax.semilogy(epochs_arr,history['train'],color=COL['train'],lw=2,label='Train')
ax.semilogy(epochs_arr,history['val'],  color=COL['val'],  lw=2,ls='--',label='Val')
ax.set_title('A — Training Loss'); ax.set_xlabel('Epoch'); ax.legend(fontsize=9)

ax=fig.add_subplot(gs[0,1])
yt0,yp0=Y_true[:,0],Y_pred[:,0]
lm=[min(yt0.min(),yp0.min()),max(yt0.max(),yp0.max())]
ax.scatter(yt0,yp0,s=2,alpha=0.1,color=COL['nn'],rasterized=True)
ax.plot(lm,lm,'k--',lw=1.2)
r2_0=1-np.sum((yt0-yp0)**2)/np.sum((yt0-yt0.mean())**2)
ax.text(0.06,0.90,f'R²={r2_0:.3f}',transform=ax.transAxes,fontsize=10)
ax.set_title(r'B — Parity $\Omega_{xx}$')

ax=fig.add_subplot(gs[0,2])
ax.hist(err,bins=60,color=COL['nn'],alpha=0.8,edgecolor='white')
ax.axvline(0,color='k',lw=1.2,ls='--')
ax.set_title('C — Error Histogram'); ax.set_xlabel('Error')

ax=fig.add_subplot(gs[1,:2])
ax.plot(time_arr,tr(C_ref_traj), color=COL['ref'],  lw=2.5,label='BD Ref')
ax.plot(time_arr,tr(C_nn_traj),  color=COL['nn'],   lw=2,  label='N2X NN')
ax.plot(time_arr,tr(C_ucm_traj), color=COL['ucm'],  lw=2,  ls='--',label='UCM')
ax.plot(time_arr,tr(C_fp_traj),  color=COL['fenep'],lw=2,  ls=':',label='FENE-P')
ax.set_title(r'D — tr(C) Shear $\dot\gamma=1$'); ax.legend(fontsize=9)

ax=fig.add_subplot(gs[1,2])
ax.plot(time_arr,tau_ref[:,0,1],color=COL['ref'],lw=2.5,label='BD Ref')
ax.plot(time_arr,tau_nn[:,0,1], color=COL['nn'], lw=2,  label='NN')
ax.plot(time_arr,tau_ucm[:,0,1],color=COL['ucm'],lw=2,  ls='--',label='UCM')
ax.set_title(r'E — Stress $\tau_{xy}$'); ax.legend(fontsize=9)

fig.suptitle('N2X — Complete Results Dashboard',fontsize=14,fontweight='bold')
plt.savefig('results/plots/fig6_dashboard.png'); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(f"  Dataset samples   : {X.shape[0]:,}")
print(f"  Network size      : 7 → 64 → 64 → 32 → 3")
print(f"  Training epochs   : {len(history['train'])}")
print(f"  Best val MSE      : {best_val:.6f}")
print(f"  Test MSE          : {mse:.6f}")
print(f"  R²                : {r2:.6f}")
print(f"  Rel. Frob. Error  : {frob*100:.2f}%")
print(f"  tr(C) MSE (NN)    : {tr_mse(C_nn_traj, C_ref_traj):.5f}")
print(f"  tr(C) MSE (UCM)   : {tr_mse(C_ucm_traj, C_ref_traj):.5f}")

summary = dict(dataset_size=int(X.shape[0]), test_mse=mse, r2=r2,
               rel_frob_pct=float(frob*100),
               nn_trace_mse=tr_mse(C_nn_traj,C_ref_traj),
               ucm_trace_mse=tr_mse(C_ucm_traj,C_ref_traj))
with open('results/final_summary.json','w') as f:
    json.dump(summary,f,indent=2)

print("\n  All outputs saved to: data/  models/  results/  results/plots/")
print("  PIPELINE COMPLETE ✓")
