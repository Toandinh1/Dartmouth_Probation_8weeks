import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")  # no GUI, no Tkinter
import matplotlib.pyplot as plt
# ================================================================
# 0) Config
# ================================================================
SCARE_CSV  = "scare_dataset_check.csv"
NORMAL_CSV = "normal_dataset.csv"

TUS_COL = "t_us"
IR_COL  = "ir"

WIN_SEC     = 4
OVERLAP_SEC = 1
STRIDE_SEC  = WIN_SEC - OVERLAP_SEC   # 3 seconds (derived)
SKIP_SEC    = 0

RANDOM_SEED = 0
N_FOLDS     = 5

# Feature list used for classical ML
FEAT_NAMES = ["ptp", "absmean", "bp_0p1_0p5", "bp_0p5_1p5", "spec_entropy_0_5"]

# ================================================================
# 1) Helpers
# ================================================================
def infer_fs_from_tus(df, tus_col="t_us"):
    t = df[tus_col].to_numpy().astype(np.float64)
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1e7)]  # keep sane deltas (<10s)
    if len(dt) < 10:
        raise ValueError("Not enough valid timestamp deltas to infer FS.")
    dt_med = np.median(dt)          # microseconds
    fs = 1e6 / dt_med
    return fs, dt_med

def make_windows(x, win, stride):
    n = len(x)
    if n < win:
        return np.empty((0, win), dtype=np.float32), np.array([], dtype=np.int64)
    starts = np.arange(0, n - win + 1, stride, dtype=np.int64)
    W = np.stack([x[s:s+win] for s in starts], axis=0).astype(np.float32)
    return W, starts

def zscore_windows(W, eps=1e-6):
    W = W - W.mean(axis=1, keepdims=True)
    W = W / (W.std(axis=1, keepdims=True) + eps)
    return W

def bandpower(x, fs, f_lo, f_hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    P = (np.abs(X) ** 2)
    m = (f >= f_lo) & (f < f_hi)
    return float(P[m].sum())

def spectral_entropy(x, fs, fmax=5.0, eps=1e-12):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    P = (np.abs(X) ** 2)
    m = (f >= 0.0) & (f <= fmax)
    Pm = P[m] + eps
    Pm = Pm / Pm.sum()
    H = -np.sum(Pm * np.log(Pm))
    return float(H)

def features_from_windows(W, fs):
    feats = {}
    feats["rms"]     = np.sqrt((W**2).mean(axis=1))
    feats["ptp"]     = W.max(axis=1) - W.min(axis=1)
    feats["absmean"] = np.abs(W).mean(axis=1)

    bp01_05, bp05_15, sent = [], [], []
    for w in W:
        bp01_05.append(bandpower(w, fs, 0.1, 0.5))
        bp05_15.append(bandpower(w, fs, 0.5, 1.5))
        sent.append(spectral_entropy(w, fs, fmax=5.0))
    feats["bp_0p1_0p5"]       = np.array(bp01_05, dtype=np.float32)
    feats["bp_0p5_1p5"]       = np.array(bp05_15, dtype=np.float32)
    feats["spec_entropy_0_5"] = np.array(sent, dtype=np.float32)
    return feats

def cohens_d(x, y, eps=1e-12):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1)*vx + (ny - 1)*vy) / (nx + ny - 2 + eps)
    return float((x.mean() - y.mean()) / (np.sqrt(pooled) + eps))

def plot_overlay_examples(Wa, Wb, fs, win_sec, n_each=5, seed=42, name_a="Scare", name_b="Normal"):
    rng = np.random.default_rng(seed)
    na = min(n_each, len(Wa))
    nb = min(n_each, len(Wb))
    ia = rng.choice(len(Wa), size=na, replace=False) if na > 0 else []
    ib = rng.choice(len(Wb), size=nb, replace=False) if nb > 0 else []
    t = np.arange(Wa.shape[1]) / fs

    plt.figure(figsize=(10, 4))
    for i in ia:
        plt.plot(t, Wa[i], alpha=0.7)
    plt.title(f"{name_a}: {na} random windows (z-scored), win={win_sec}s @ {fs:.1f}Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for i in ib:
        plt.plot(t, Wb[i], alpha=0.7)
    plt.title(f"{name_b}: {nb} random windows (z-scored), win={win_sec}s @ {fs:.1f}Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hist_compare(a, b, name, bins=35, win_sec=4, fs=64.0):
    plt.figure(figsize=(7, 4))
    plt.hist(a, bins=bins, alpha=0.5, density=True, label="Scare")
    plt.hist(b, bins=bins, alpha=0.5, density=True, label="Normal")
    plt.title(f"Distribution: {name} (win={win_sec}s, fs≈{fs:.1f}Hz)")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Blocked CV split to reduce temporal leakage ----
# We split each class into N_FOLDS contiguous chunks and test on chunk k.
def blocked_cv_indices(n_scare, n_norm, n_folds=5):
    s_idx = np.arange(n_scare)
    n_idx = np.arange(n_norm)

    s_folds = np.array_split(s_idx, n_folds)
    n_folds_arr = np.array_split(n_idx, n_folds)

    for k in range(n_folds):
        te_s = s_folds[k]
        te_n = n_folds_arr[k]

        tr_s = np.concatenate([s_folds[i] for i in range(n_folds) if i != k]) if n_folds > 1 else np.array([], dtype=int)
        tr_n = np.concatenate([n_folds_arr[i] for i in range(n_folds) if i != k]) if n_folds > 1 else np.array([], dtype=int)

        # Map to global indices in concatenated X = [scare; normal]
        te = np.concatenate([te_s, n_scare + te_n])
        tr = np.concatenate([tr_s, n_scare + tr_n])
        yield tr, te

def _score_positive_class(model, X_te):
    """
    Continuous score for ROC-AUC:
      - prefer predict_proba[:,1]
      - else decision_function scaled to [0,1]
      - else predicted labels as 0/1
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_te)[:, 1]
        return p.astype(np.float64)

    if hasattr(model, "decision_function"):
        s = model.decision_function(X_te).astype(np.float64)
        s_min, s_max = np.min(s), np.max(s)
        if s_max - s_min < 1e-12:
            return np.full_like(s, 0.5, dtype=np.float64)
        return (s - s_min) / (s_max - s_min)

    return model.predict(X_te).astype(np.float64)

def eval_model_blocked_cv(model, X, y, n_scare, n_norm, n_folds=5, thr=0.5):
    aucs, accs, f1s = [], [], []
    cm_sum = np.zeros((2, 2), dtype=int)  # [[TN,FP],[FN,TP]] with labels=[0,1]

    for tr, te in blocked_cv_indices(n_scare, n_norm, n_folds=n_folds):
        if len(tr) == 0 or len(te) == 0:
            continue

        m = clone(model)
        m.fit(X[tr], y[tr])

        score = _score_positive_class(m, X[te])
        pred = (score >= thr).astype(int)

        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], score))
        else:
            aucs.append(np.nan)

        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, pos_label=1))
        cm_sum += confusion_matrix(y[te], pred, labels=[0, 1])

    aucs = np.array(aucs, dtype=np.float64)
    accs = np.array(accs, dtype=np.float64)
    f1s  = np.array(f1s, dtype=np.float64)

    aucs = aucs[~np.isnan(aucs)]  # drop invalid folds (rare)

    return aucs, accs, f1s, cm_sum

# ================================================================
# 2) Load + infer FS
# ================================================================
scare  = pd.read_csv(SCARE_CSV)
normal = pd.read_csv(NORMAL_CSV)

FS_s, dt_med_s = infer_fs_from_tus(scare,  TUS_COL)
FS_n, dt_med_n = infer_fs_from_tus(normal, TUS_COL)
FS = (FS_s + FS_n) / 2.0

print(f"Inferred FS: scare={FS_s:.2f}Hz (dt_med={dt_med_s:.0f}us), "
      f"normal={FS_n:.2f}Hz (dt_med={dt_med_n:.0f}us)")
print(f"Using FS={FS:.2f}Hz")

# ================================================================
# 3) Windowing (4s win, 1s overlap => 3s stride)
# ================================================================
WIN    = int(round(FS * WIN_SEC))
STRIDE = int(round(FS * STRIDE_SEC))
SKIP   = int(round(FS * SKIP_SEC))

scare_ir  = scare[IR_COL].to_numpy().astype(np.float32)[SKIP:]
normal_ir = normal[IR_COL].to_numpy().astype(np.float32)[SKIP:]

Ws, starts_s = make_windows(scare_ir,  WIN, STRIDE)
Wn, starts_n = make_windows(normal_ir, WIN, STRIDE)

print("Window samples:", WIN, "Stride samples:", STRIDE)
print("Windows:", "scare", Ws.shape, "normal", Wn.shape)
if len(Ws) == 0 or len(Wn) == 0:
    raise RuntimeError("Not enough data to form windows for at least one class.")

# Normalize per-window (remove DC + scale)
Ws_n = zscore_windows(Ws)
Wn_n = zscore_windows(Wn)

# Quick qualitative check
plot_overlay_examples(Ws_n, Wn_n, FS, WIN_SEC, n_each=5, seed=42, name_a="Scare", name_b="Normal")

# ================================================================
# 4) Feature extraction
# ================================================================
Fs = features_from_windows(Ws_n, FS)
Fn = features_from_windows(Wn_n, FS)

# Optional: show distributions
for k in FEAT_NAMES:
    plot_hist_compare(Fs[k], Fn[k], k, bins=35, win_sec=WIN_SEC, fs=FS)

# 2D scatter (intuition)
plt.figure(figsize=(7, 5))
plt.scatter(Fs["bp_0p5_1p5"], Fs["spec_entropy_0_5"], s=18, alpha=0.65, label="Scare")
plt.scatter(Fn["bp_0p5_1p5"], Fn["spec_entropy_0_5"], s=18, alpha=0.65, label="Normal")
plt.title("Feature scatter: Bandpower 0.5–1.5 Hz vs Spectral entropy (0–5Hz)")
plt.xlabel("Bandpower 0.5–1.5 Hz")
plt.ylabel("Spectral entropy (0–5 Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# 5) Univariate evidence: effect sizes + MWU + FDR
# ================================================================
print("\n=== Distribution shift: Cohen's d + Mann–Whitney + FDR ===")
pvals, rows = [], []
for k in FEAT_NAMES:
    d = cohens_d(Fs[k], Fn[k])
    u = mannwhitneyu(Fs[k], Fn[k], alternative="two-sided")
    pvals.append(u.pvalue)
    rows.append((k, d, u.pvalue))

rej, p_fdr = fdrcorrection(np.array(pvals), alpha=0.05)
for i, (k, d, p) in enumerate(rows):
    print(f"{k:16s} d={d:+.3f}  p={p:.3e}  p_fdr={p_fdr[i]:.3e}  sig={bool(rej[i])}")

# ================================================================
# 6) Build dataset X, y (one window = one sample)
# ================================================================
Xs = np.stack([Fs[k] for k in FEAT_NAMES], axis=1)
Xn = np.stack([Fn[k] for k in FEAT_NAMES], axis=1)

X = np.vstack([Xs, Xn]).astype(np.float32)
y = np.array([1]*len(Xs) + [0]*len(Xn), dtype=np.int32)  # 1=scare, 0=normal

n_scare = len(Xs)
n_norm  = len(Xn)
print(f"\nSamples (windows): scare={n_scare}, normal={n_norm}, total={len(y)}")

# ================================================================
# 7) Classical ML: Linear SVM + RF (blocked CV)
# ================================================================

# LinearSVC does NOT support probability=True, so calibrate for ROC-AUC.
svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        estimator=LinearSVC(
            C=3.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=20000
        ),
        method="sigmoid",
        cv=3
    ))
])

# RF in pipeline for consistent interface
rf = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

models = {
    "LinearSVM (calibrated)": svm_linear,
    "RandomForest": rf
}

# ----------------------------
# Evaluation (cached for boxplot)
# ----------------------------
print("\n=== Blocked CV (time-chunked) results ===")
cached_aucs = {}

for name, model in models.items():
    aucs, accs, f1s, cm = eval_model_blocked_cv(
        model, X, y, n_scare, n_norm, n_folds=N_FOLDS, thr=0.5
    )
    cached_aucs[name] = aucs

    print(f"\n{name}")
    if len(aucs) > 0:
        print(f"ROC-AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
    else:
        print("ROC-AUC: (skipped) not enough valid folds with both classes")

    print(f"ACC:     {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"F1:      {f1s.mean():.3f} ± {f1s.std():.3f}")
    print("Confusion matrix (sum over folds) [[TN FP],[FN TP]]:")
    print(cm)

# Optional: visualize AUCs (no re-training)
plt.figure(figsize=(7, 4))
data = [cached_aucs[k] for k in cached_aucs.keys() if len(cached_aucs[k]) > 0]
labels = [k for k in cached_aucs.keys() if len(cached_aucs[k]) > 0]
plt.boxplot(data, labels=labels, vert=True)
plt.axhline(0.5, linestyle="--", label="Chance AUC=0.5")
plt.ylabel("ROC-AUC")
plt.title("Blocked CV ROC-AUC (classical ML)")
plt.grid(True, axis="y")
plt.legend()
plt.tight_layout()
plt.show()
