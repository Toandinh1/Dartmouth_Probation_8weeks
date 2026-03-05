import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

# =========================
# Config (MATCH MCU)
# =========================
SCARE_CSV  = "focus_dataset.csv"
NORMAL_CSV = "normal_dataset.csv"

TUS_COL = "t_us"
IR_COL  = "ir"

FS_TARGET = 100
N = 512
HOP = 384

RANDOM_SEED = 0
N_FOLDS = 5

EPS_Z   = 1e-6
EPS_BP  = 1e-6
EPS_ENT = 1e-12

# Wider grid helps
C_GRID = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
CW_GRID = [None, "balanced"]  # try both

# Feature order MUST match MCU feat5[0..4]
FEAT_NAMES = ["ptp", "absmean", "log_bp_0p1_0p5", "log_bp_0p5_1p5", "spec_entropy_0_5"]

# Choose thresholding objective:
# "balanced" -> maximize (TPR + TNR)/2
# "youden"   -> maximize (TPR - FPR) = TPR + TNR - 1
THR_MODE = "balanced"

# Optional: after selecting threshold, shift it upward a bit to reduce FP (improve normal).
# Set to 0.0 to disable. Typical: 0.0 ~ 0.3 * std(scores_train)
THR_SHIFT_SIGMA = 0.25

# =========================
# Helpers
# =========================
def infer_fs_from_tus(df, tus_col="t_us"):
    t = df[tus_col].to_numpy().astype(np.float64)
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1e7)]
    if len(dt) < 10:
        raise ValueError("Not enough valid timestamp deltas to infer FS.")
    fs = 1e6 / np.median(dt)
    return fs

def resample_to_fs(x, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    # deterministic output length (helps reproducibility)
    n_out = int(round(n_in * fs_out / fs_in))
    if n_out < 2:
        return x.astype(np.float32)
    t_in = np.arange(n_in, dtype=np.float64) / fs_in
    t_out = np.arange(n_out, dtype=np.float64) / fs_out
    y = np.interp(t_out, t_in, x.astype(np.float64)).astype(np.float32)
    return y

def make_windows(x, win, hop):
    n = len(x)
    if n < win:
        return np.empty((0, win), dtype=np.float32)
    starts = np.arange(0, n - win + 1, hop, dtype=np.int64)
    W = np.stack([x[s:s+win] for s in starts], axis=0).astype(np.float32)
    return W

def zscore_windows(W, eps=1e-6):
    W = W - W.mean(axis=1, keepdims=True)
    W = W / (W.std(axis=1, keepdims=True) + eps)
    return W

def features_mcu_matched(Wz, fs):
    f = np.fft.rfftfreq(Wz.shape[1], d=1/fs)
    m01_05 = (f >= 0.1) & (f < 0.5)
    m05_15 = (f >= 0.5) & (f < 1.5)
    m0_5   = (f >= 0.0) & (f <= 5.0)

    ptp = (Wz.max(axis=1) - Wz.min(axis=1)).astype(np.float32)
    absmean = np.abs(Wz).mean(axis=1).astype(np.float32)

    log_bp01 = np.zeros(len(Wz), dtype=np.float32)
    log_bp05 = np.zeros(len(Wz), dtype=np.float32)
    ent05    = np.zeros(len(Wz), dtype=np.float32)

    # FFT per window (rectangle window, MCU friendly)
    for i, w in enumerate(Wz):
        X = np.fft.rfft(w)
        P = (np.abs(X) ** 2).astype(np.float64)

        bp01 = float(P[m01_05].sum())
        bp05 = float(P[m05_15].sum())

        log_bp01[i] = np.float32(np.log(bp01 + EPS_BP))
        log_bp05[i] = np.float32(np.log(bp05 + EPS_BP))

        Pm = P[m0_5] + EPS_ENT
        s = Pm.sum()
        if s <= 0:
            ent05[i] = np.float32(0.0)
        else:
            Pm = Pm / s
            ent05[i] = np.float32(-np.sum(Pm * np.log(Pm)))

    Xfeat = np.stack([ptp, absmean, log_bp01, log_bp05, ent05], axis=1).astype(np.float32)
    return Xfeat

def blocked_cv_indices(n_pos, n_neg, n_folds=5):
    # assumes windows are time-ordered within each class
    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg)
    pos_folds = np.array_split(pos_idx, n_folds)
    neg_folds = np.array_split(neg_idx, n_folds)

    for k in range(n_folds):
        te_p = pos_folds[k]
        te_n = neg_folds[k]
        tr_p = np.concatenate([pos_folds[i] for i in range(n_folds) if i != k])
        tr_n = np.concatenate([neg_folds[i] for i in range(n_folds) if i != k])

        te = np.concatenate([te_p, n_pos + te_n])
        tr = np.concatenate([tr_p, n_pos + tr_n])
        yield tr, te

def metrics_from_cm(cm):
    # cm = [[TN, FP], [FN, TP]]
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    tpr = TP / (TP + FN + 1e-12)  # scare recall
    tnr = TN / (TN + FP + 1e-12)  # normal recall (your target)
    bal = 0.5 * (tpr + tnr)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    return tpr, tnr, bal, acc

def choose_threshold(scores, y_true, mode="balanced"):
    # choose threshold on TRAIN fold only
    ts = np.quantile(scores, np.linspace(0.01, 0.99, 199))
    best_t = float(ts[0])
    best_val = -1e18

    for t in ts:
        pred = (scores >= t).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tpr, tnr, bal, _ = metrics_from_cm(cm)

        if mode == "balanced":
            val = bal
        elif mode == "youden":
            # TPR + TNR - 1
            val = tpr + tnr - 1.0
        else:
            raise ValueError("mode must be 'balanced' or 'youden'")

        if val > best_val:
            best_val = val
            best_t = float(t)

    # optional upward shift to reduce FP (better normal)
    if THR_SHIFT_SIGMA != 0.0:
        sigma = float(np.std(scores))
        best_t = best_t + THR_SHIFT_SIGMA * sigma

    return best_t

def blocked_cv_eval(model_template, X, y, n_pos, n_neg, n_folds=5, thr_mode="balanced"):
    aucs, accs, f1s = [], [], []
    tprs, tnrs, bals = [], [], []
    cm_sum = np.zeros((2, 2), dtype=int)
    thrs = []

    for tr, te in blocked_cv_indices(n_pos, n_neg, n_folds=n_folds):
        m = clone(model_template)
        m.fit(X[tr], y[tr])

        s_tr = m.decision_function(X[tr]).astype(np.float64)
        thr = choose_threshold(s_tr, y[tr], mode=thr_mode)
        thrs.append(thr)

        s_te = m.decision_function(X[te]).astype(np.float64)
        pred = (s_te >= thr).astype(int)

        cm = confusion_matrix(y[te], pred, labels=[0, 1])
        cm_sum += cm

        tpr, tnr, bal, acc = metrics_from_cm(cm)
        tprs.append(tpr); tnrs.append(tnr); bals.append(bal)
        accs.append(acc)
        f1s.append(f1_score(y[te], pred, pos_label=1))

        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], s_te))

    return (np.array(aucs), np.array(accs), np.array(f1s),
            cm_sum, np.array(thrs, dtype=np.float64),
            np.array(tprs), np.array(tnrs), np.array(bals))

def c_float_array(name, arr):
    elems = ", ".join([f"{x:.8e}f" for x in arr])
    return f"static const float {name}[{len(arr)}] = {{{elems}}};\n"

# =========================
# Load -> resample -> windows -> features
# =========================
scare  = pd.read_csv(SCARE_CSV)
normal = pd.read_csv(NORMAL_CSV)

# safety: ensure time is monotonic (helps blocked split correctness)
ts_s = scare[TUS_COL].to_numpy()
ts_n = normal[TUS_COL].to_numpy()
if np.any(np.diff(ts_s) <= 0):
    print("WARNING: scare timestamps not strictly increasing. Blocked CV may be invalid.")
if np.any(np.diff(ts_n) <= 0):
    print("WARNING: normal timestamps not strictly increasing. Blocked CV may be invalid.")

fs_s = infer_fs_from_tus(scare,  TUS_COL)
fs_n = infer_fs_from_tus(normal, TUS_COL)
print(f"Inferred FS: scare={fs_s:.2f}Hz normal={fs_n:.2f}Hz; resample to {FS_TARGET:.2f}Hz")

scare_ir  = resample_to_fs(scare[IR_COL].to_numpy().astype(np.float32),  fs_s, FS_TARGET)
normal_ir = resample_to_fs(normal[IR_COL].to_numpy().astype(np.float32), fs_n, FS_TARGET)

Ws = make_windows(scare_ir,  N, HOP)
Wn = make_windows(normal_ir, N, HOP)

if len(Ws) == 0 or len(Wn) == 0:
    raise RuntimeError("Not enough data to form windows for at least one class.")

Ws_z = zscore_windows(Ws, eps=EPS_Z)
Wn_z = zscore_windows(Wn, eps=EPS_Z)

Xs = features_mcu_matched(Ws_z, FS_TARGET)
Xn = features_mcu_matched(Wn_z, FS_TARGET)

X = np.vstack([Xs, Xn]).astype(np.float32)
y = np.array([1]*len(Xs) + [0]*len(Xn), dtype=np.int32)

n_pos, n_neg = len(Xs), len(Xn)
print(f"Windows: scare={n_pos} normal={n_neg} total={len(y)}")
print("Feature order:", FEAT_NAMES)

# =========================
# Model search: maximize NORMAL accuracy (TNR)
# =========================
best = None

print("\n=== Blocked CV sweep (optimize NORMAL/TNR) ===")
for cw in CW_GRID:
    for C in C_GRID:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=C, class_weight=cw, random_state=RANDOM_SEED, max_iter=50000))
        ])

        aucs, accs, f1s, cm, thrs, tprs, tnrs, bals = blocked_cv_eval(
            model, X, y, n_pos, n_neg, n_folds=N_FOLDS, thr_mode=THR_MODE
        )

        res = {
            "C": C,
            "cw": cw,
            "tnr": float(tnrs.mean()) if len(tnrs) else float("nan"),
            "tpr": float(tprs.mean()) if len(tprs) else float("nan"),
            "bal": float(bals.mean()) if len(bals) else float("nan"),
            "acc": float(accs.mean()) if len(accs) else float("nan"),
            "auc": float(np.nanmean(aucs)) if len(aucs) else float("nan"),
            "thr": float(np.median(thrs)) if len(thrs) else 0.0,
            "cm": cm,
        }

        print(f"[cw={cw} C={C:<5}] "
              f"TNR={res['tnr']:.3f}  TPR={res['tpr']:.3f}  "
              f"BAL={res['bal']:.3f}  ACC={res['acc']:.3f}  AUC={res['auc']:.3f}  "
              f"thr_med={res['thr']:.6f}")

        # choose best by NORMAL accuracy first, tie-break by balanced accuracy then AUC
        if best is None:
            best = res
        else:
            if (res["tnr"] > best["tnr"] + 1e-9) or \
               (abs(res["tnr"] - best["tnr"]) < 1e-9 and res["bal"] > best["bal"] + 1e-9) or \
               (abs(res["tnr"] - best["tnr"]) < 1e-9 and abs(res["bal"] - best["bal"]) < 1e-9 and res["auc"] > best["auc"] + 1e-9):
                best = res

print("\nBest (opt NORMAL/TNR):")
print(best)
BEST_C  = best["C"]
BEST_CW = best["cw"]
DEPLOY_THR = best["thr"]

print("\nConfusion matrix sum over folds [[TN FP],[FN TP]]:")
print(best["cm"])
print(f"Chosen deploy threshold (median over folds): {DEPLOY_THR:.8f}")

# =========================
# Train final model on ALL data, export MCU header
# =========================
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC(C=BEST_C, class_weight=BEST_CW, random_state=RANDOM_SEED, max_iter=50000))
])
final_model.fit(X, y)

# Export params
scaler = final_model.named_steps["scaler"]
clf    = final_model.named_steps["clf"]

MEAN  = scaler.mean_.astype(np.float32)
SCALE = scaler.scale_.astype(np.float32)
W     = clf.coef_.reshape(-1).astype(np.float32)
B     = float(clf.intercept_[0])
D     = len(W)

header = ""
header += "#pragma once\n"
header += "// Auto-generated Linear SVM params (MCU-matched)\n"
header += f"// BEST_C={BEST_C}  CLASS_WEIGHT={'NONE' if BEST_CW is None else 'BALANCED'}\n"
header += f"// THR_MODE={THR_MODE}  THR_SHIFT_SIGMA={THR_SHIFT_SIGMA}\n"
header += f"#define SVM_D {D}\n\n"
header += c_float_array("SVM_MEAN", MEAN)
header += c_float_array("SVM_SCALE", SCALE)
header += c_float_array("SVM_W", W)
header += f"static const float SVM_B = {B:.8e}f;\n"
header += f"static const float SVM_THR = {DEPLOY_THR:.8e}f;\n\n"

header += r"""
static inline float svm_score(const float *x_raw) {
  float score = SVM_B;
  for (int i = 0; i < SVM_D; i++) {
    float xn = (x_raw[i] - SVM_MEAN[i]) / (SVM_SCALE[i] + 1e-12f);
    score += SVM_W[i] * xn;
  }
  return score;
}

static inline int svm_predict(const float *x_raw) {
  float score = svm_score(x_raw);
  return (score >= SVM_THR) ? 1 : 0; // 1=scare, 0=normal
}
"""

with open("linear_svm_params.h", "w") as f:
    f.write(header)

print("\nWrote linear_svm_params.h")
