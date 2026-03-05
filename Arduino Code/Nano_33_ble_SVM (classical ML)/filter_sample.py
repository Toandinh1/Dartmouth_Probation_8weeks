import numpy as np
import pandas as pd

# =========================
# Inputs
# =========================
RAW_CSV = "scare_dataset1.csv"                # your original raw dataset
KEEP_WINDOWS_CSV = "scare_filtered_windows.csv"  # contains start_sample for each kept window

# Window params (must match how you created windows)
FS_TARGET = 100.0   # same as training run
N = 256
HOP = 192

# Output
OUT_RAW_WINDOWS = "scare_raw_kept_windows_exact.csv"
OUT_RAW_SEGMENTS = "scare_raw_kept_segments_merged.csv"

# =========================
# Helpers
# =========================
def infer_fs_from_tus(df, tus_col="t_us"):
    t = df[tus_col].to_numpy().astype(np.float64)
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1e7)]
    if len(dt) < 10:
        raise ValueError("Not enough valid timestamp deltas to infer FS.")
    return 1e6 / np.median(dt)

def resample_df_to_fs(df, fs_in, fs_out, tus_col="t_us", cols=("ir","red","green","finger","rb_used","flags","idx")):
    """
    Resample a raw dataframe to fs_out using interpolation on time.
    This produces a new dataframe with ~uniform sampling.
    Only numeric columns are interpolated; others are carried by nearest or dropped.
    """
    # Build time in seconds from t_us (relative)
    t_us = df[tus_col].to_numpy(np.float64)
    t0 = t_us[0]
    t = (t_us - t0) / 1e6  # seconds

    # Target timeline
    duration = t[-1]
    n_out = int(np.floor(duration * fs_out)) + 1
    t_out = np.arange(n_out, dtype=np.float64) / fs_out

    out = pd.DataFrame()
    out[tus_col] = (t_out * 1e6 + t0).astype(np.int64)

    # Interpolate key signals if present
    for c in ["ir", "red", "green"]:
        if c in df.columns:
            y = df[c].to_numpy(np.float64)
            out[c] = np.interp(t_out, t, y).astype(np.float32)

    # Carry non-interpolated columns (optional)
    # For labels/flags, nearest neighbor makes more sense than linear
    for c in ["finger", "rb_used", "flags", "idx"]:
        if c in df.columns:
            y = df[c].to_numpy(np.float64)
            out[c] = np.round(np.interp(t_out, t, y)).astype(np.int64)

    return out

def merge_intervals(intervals):
    """intervals: list of (start,end) inclusive-exclusive indices on resampled signal."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s,e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:  # overlap / touch
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s,e))
    return merged

# =========================
# Load files
# =========================
raw = pd.read_csv(RAW_CSV)
keep = pd.read_csv(KEEP_WINDOWS_CSV)

if "start_sample" not in keep.columns:
    raise ValueError("KEEP_WINDOWS_CSV must contain a 'start_sample' column (from window creation).")

# =========================
# Resample raw to FS_TARGET (so start_sample indices match)
# =========================
fs_in = infer_fs_from_tus(raw, "t_us")
print(f"Raw inferred fs_in={fs_in:.2f}Hz -> resample to {FS_TARGET:.2f}Hz")

raw_rs = resample_df_to_fs(raw, fs_in, FS_TARGET, tus_col="t_us")
print("Resampled length:", len(raw_rs))

# =========================
# Build intervals for kept windows
# =========================
starts = keep["start_sample"].to_numpy(np.int64)
intervals = [(int(s), int(s + N)) for s in starts]  # emphasize: [s, s+N)

# A) exact windows (duplicates allowed)
rows_exact = []
for s,e in intervals:
    chunk = raw_rs.iloc[s:e].copy()
    chunk["win_start_sample"] = s
    rows_exact.append(chunk)

raw_exact = pd.concat(rows_exact, ignore_index=True)
raw_exact.to_csv(OUT_RAW_WINDOWS, index=False)
print("Wrote exact window raw export:", OUT_RAW_WINDOWS, "rows=", len(raw_exact))

# B) merged segments (no duplicates)
merged = merge_intervals(intervals)
rows_seg = []
for s,e in merged:
    chunk = raw_rs.iloc[s:e].copy()
    chunk["seg_start_sample"] = s
    chunk["seg_end_sample"] = e
    rows_seg.append(chunk)

raw_seg = pd.concat(rows_seg, ignore_index=True)
raw_seg.to_csv(OUT_RAW_SEGMENTS, index=False)
print("Wrote merged segment raw export:", OUT_RAW_SEGMENTS, "rows=", len(raw_seg))

print("\nIntervals:")
print(" windows kept:", len(intervals))
print(" merged segments:", len(merged))
