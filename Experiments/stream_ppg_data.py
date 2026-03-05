# import time
# import numpy as np
# import serial

# PORT = "COM7"
# BAUD = 115200
# FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"

# fs = 32  # try 64, 32, 16
# dt = 1.0 / fs

# with open(FILE, "r") as f:
#     lines = [line.strip() for line in f if line.strip()]
# vals = np.array([float(x) for x in lines], dtype=np.float32)
# ppg = vals[2:]

# ser = serial.Serial(PORT, BAUD, timeout=0.1)
# time.sleep(2)
# ser.reset_input_buffer()

# t0 = time.perf_counter()
# next_t = t0

# for i, v in enumerate(ppg[:5000]):
#     ser.write(f"{float(v)}\n".encode())

#     # Read MCU output without blocking too long
#     while ser.in_waiting:
#         s = ser.readline().decode(errors="ignore").strip()
#         if s:
#             print(s)

#     # schedule next sample time precisely
#     next_t += dt
#     while True:
#         now = time.perf_counter()
#         if now >= next_t:
#             break
#         time.sleep(0.0005)  # 0.5ms yield


# import time
# import re
# import numpy as np
# import serial
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ----------------------------
# # Config
# # ----------------------------
# PORT = "COM7"
# BAUD = 115200
# FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"

# FS_LIST = [64, 32, 16]      # compare these
# N_SAMPLES = 8000            # how many samples to stream per run
# WARMUP_SEC = 2.0            # ignore first few seconds of PWR logs (startup transient)
# OUTDIR = Path("ppg_runs")
# OUTDIR.mkdir(exist_ok=True)

# # If your Arduino supports "RST\n" command, it helps reset counters between runs
# SEND_RST = True

# # ----------------------------
# # Parse [PWR] lines from MCU
# # Example:
# # [PWR] dt=15.00s  Fs=32.0Hz  samp=31.87/s  inf=0.0667/s  active=23.8%  infer_ms/s=78.627  fill_ms/s=0.027  rx_ms/s=96.826  Eproxy_ms/s=175.480
# # ----------------------------
# PWR_RE = re.compile(
#     r"\[PWR\]\s*dt=(?P<dt>[\d.]+)s\s+"
#     r"Fs=(?P<Fs>[\d.]+)Hz\s+"
#     r"samp=(?P<samp>[\d.]+)/s\s+"
#     r"inf=(?P<inf>[\d.]+)/s\s+"
#     r"active=(?P<active>[\d.]+)%\s+"
#     r"infer_ms/s=(?P<infer_ms_s>[\d.]+)\s+"
#     r"fill_ms/s=(?P<fill_ms_s>[\d.]+)\s+"
#     r"rx_ms/s=(?P<rx_ms_s>[\d.]+)\s+"
#     r"Eproxy_ms/s=(?P<eproxy_ms_s>[\d.]+)"
# )

# def load_ppg_csv(file_path: str) -> np.ndarray:
#     with open(file_path, "r") as f:
#         lines = [line.strip() for line in f if line.strip()]
#     vals = np.array([float(x) for x in lines], dtype=np.float32)
#     return vals[2:]  # skip header lines used in your data

# def drain_serial(ser: serial.Serial, max_lines=5000, print_all=False):
#     """Drain anything currently in the serial buffer."""
#     lines = []
#     n = 0
#     while ser.in_waiting and n < max_lines:
#         s = ser.readline().decode(errors="ignore").strip()
#         if s:
#             lines.append(s)
#             if print_all:
#                 print(s)
#         n += 1
#     return lines

# def run_one_fs(ser: serial.Serial, ppg: np.ndarray, fs: int, n_samples: int):
#     dt = 1.0 / fs

#     # Reset device state (optional but recommended for clean counters)
#     if SEND_RST:
#         ser.write(b"RST\n")
#         time.sleep(0.2)
#         drain_serial(ser, print_all=False)

#     # Time scheduling: perf_counter has good resolution on Windows
#     t0 = time.perf_counter()
#     next_t = t0

#     rows = []
#     start_wall = time.time()

#     # stream
#     for i, v in enumerate(ppg[:n_samples]):
#         ser.write(f"{float(v)}\n".encode())

#         # read any MCU output (non-blocking-ish)
#         # keep this tight so you don't slip timing too much
#         while ser.in_waiting:
#             s = ser.readline().decode(errors="ignore").strip()
#             if not s:
#                 continue

#             m = PWR_RE.search(s)
#             if m:
#                 d = m.groupdict()
#                 d = {k: float(v) for k, v in d.items()}
#                 # record wall time relative to start
#                 d["t_sec"] = time.time() - start_wall
#                 d["fs_run"] = fs
#                 rows.append(d)
#             # uncomment if you want to see everything
#             # else: print(s)

#         # schedule next sample time precisely
#         next_t += dt
#         # Busy-wait with small sleeps; adaptively reduce CPU burn
#         while True:
#             now = time.perf_counter()
#             remain = next_t - now
#             if remain <= 0:
#                 break
#             # sleep most of the remaining time, keep a tiny margin
#             if remain > 0.003:
#                 time.sleep(remain - 0.002)
#             else:
#                 time.sleep(0.0003)

#     # final drain
#     time.sleep(0.2)
#     drain_serial(ser, print_all=False)

#     df = pd.DataFrame(rows)
#     return df

# def summarize_df(df: pd.DataFrame, warmup_sec: float):
#     if df.empty:
#         return None

#     # remove warmup transient
#     df2 = df[df["t_sec"] >= warmup_sec].copy()
#     if df2.empty:
#         df2 = df.copy()

#     # summary stats
#     metrics = ["active", "infer_ms_s", "fill_ms_s", "rx_ms_s", "eproxy_ms_s", "samp", "inf", "dt"]
#     summary = df2[metrics].agg(["mean", "std", "count"]).T.reset_index()
#     summary.rename(columns={"index": "metric"}, inplace=True)
#     return df2, summary

# def plot_bars(all_summaries: dict, outpath: Path):
#     """
#     all_summaries: {fs: summary_df}
#     """
#     metrics_to_plot = ["active", "infer_ms_s", "rx_ms_s", "eproxy_ms_s", "inf", "samp"]
#     fs_list = sorted(all_summaries.keys())

#     fig, ax = plt.subplots(figsize=(11, 5))
#     width = 0.12
#     x = np.arange(len(fs_list))

#     for j, metric in enumerate(metrics_to_plot):
#         means = []
#         stds = []
#         for fs in fs_list:
#             s = all_summaries[fs]
#             row = s[s["metric"] == metric].iloc[0]
#             means.append(row["mean"])
#             stds.append(row["std"] if not np.isnan(row["std"]) else 0.0)

#         ax.bar(x + j * width, means, width=width, yerr=stds, capsize=3, label=metric)

#     ax.set_xticks(x + (len(metrics_to_plot)-1)*width/2)
#     ax.set_xticklabels([str(fs) for fs in fs_list])
#     ax.set_xlabel("Streaming Fs (Hz)")
#     ax.set_ylabel("Value (mean ± std over PWR windows)")
#     ax.set_title("MCU resource proxies vs Fs (from [PWR] logs)")
#     ax.legend(ncol=3, fontsize=9)
#     fig.tight_layout()
#     fig.savefig(outpath, dpi=200)
#     plt.show()

# def plot_timeseries(df: pd.DataFrame, fs: int, outpath: Path):
#     if df.empty:
#         return
#     fig, ax = plt.subplots(figsize=(11, 4))
#     ax.plot(df["t_sec"], df["active"], label="active%")
#     ax.plot(df["t_sec"], df["eproxy_ms_s"], label="Eproxy_ms/s")
#     ax.plot(df["t_sec"], df["rx_ms_s"], label="rx_ms/s")
#     ax.plot(df["t_sec"], df["infer_ms_s"], label="infer_ms/s")
#     ax.set_title(f"Timeseries (Fs={fs}Hz) from [PWR] logs")
#     ax.set_xlabel("t (sec)")
#     ax.set_ylabel("value")
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(outpath, dpi=200)
#     plt.show()

# def main():
#     ppg = load_ppg_csv(FILE)
#     print("Loaded samples:", len(ppg))

#     ser = serial.Serial(PORT, BAUD, timeout=0.1)
#     time.sleep(2)
#     ser.reset_input_buffer()
#     drain_serial(ser, print_all=False)

#     all_runs = []
#     all_summaries = {}

#     for fs in FS_LIST:
#         print(f"\n=== Running Fs={fs} Hz ===")
#         df = run_one_fs(ser, ppg, fs, N_SAMPLES)

#         raw_csv = OUTDIR / f"raw_pwr_fs{fs}.csv"
#         df.to_csv(raw_csv, index=False)
#         print("Saved:", raw_csv)

#         df2, summary = summarize_df(df, WARMUP_SEC)
#         if summary is None:
#             print("No [PWR] lines captured. Check Arduino printing / regex / serial.")
#             continue

#         sum_csv = OUTDIR / f"summary_fs{fs}.csv"
#         summary.to_csv(sum_csv, index=False)
#         print("Saved:", sum_csv)

#         all_runs.append(df2)
#         all_summaries[fs] = summary

#         # optional timeseries per run
#         plot_timeseries(df2, fs, OUTDIR / f"timeseries_fs{fs}.png")

#     # combined bar plot
#     if all_summaries:
#         plot_bars(all_summaries, OUTDIR / "compare_bar.png")

#     ser.close()

# if __name__ == "__main__":
#     main()


# import time
# import re
# import numpy as np
# import serial
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ----------------------------
# # Config
# # ----------------------------
# PORT = "COM7"
# BAUD = 115200
# FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"

# FS_FIXED = 64                 # keep fixed
# HOP_LIST = [512, 384, 256, 128]
# N_SAMPLES = 20000             # increase to get enough [PWR] windows per hop
# WARMUP_SEC = 5.0              # ignore first windows after changing hop
# OUTDIR = Path("ppg_hop_runs")
# OUTDIR.mkdir(exist_ok=True)

# SEND_RST = True

# # ----------------------------
# # Parse [PWR] lines from MCU
# # Expect (recommended):
# # [PWR] dt=5.00s  Fs=64.0Hz  HOP=256  samp=64.00/s  inf=0.2500/s  active=...
# # ----------------------------
# PWR_RE = re.compile(
#     r"\[PWR\]\s*dt=(?P<dt>[\d.]+)s\s+"
#     r"Fs=(?P<Fs>[\d.]+)Hz\s+"
#     r"(?:HOP=(?P<HOP>\d+)\s+)?"
#     r"samp=(?P<samp>[\d.]+)/s\s+"
#     r"inf=(?P<inf>[\d.]+)/s\s+"
#     r"active=(?P<active>[\d.]+)%\s+"
#     r"infer_ms/s=(?P<infer_ms_s>[\d.]+)\s+"
#     r"fill_ms/s=(?P<fill_ms_s>[\d.]+)\s+"
#     r"rx_ms/s=(?P<rx_ms_s>[\d.]+)\s+"
#     r"Eproxy_ms/s=(?P<eproxy_ms_s>[\d.]+)"
# )

# def load_ppg_csv(file_path: str) -> np.ndarray:
#     with open(file_path, "r") as f:
#         lines = [line.strip() for line in f if line.strip()]
#     vals = np.array([float(x) for x in lines], dtype=np.float32)
#     return vals[2:]  # skip header lines used in your data

# def drain_serial(ser: serial.Serial, max_lines=5000, print_all=False):
#     lines = []
#     n = 0
#     while ser.in_waiting and n < max_lines:
#         s = ser.readline().decode(errors="ignore").strip()
#         if s:
#             lines.append(s)
#             if print_all:
#                 print(s)
#         n += 1
#     return lines

# def send_cmd(ser: serial.Serial, cmd: str, settle=0.2, verbose=True):
#     if verbose:
#         print(">>", cmd.strip())
#     ser.write(cmd.encode())
#     time.sleep(settle)
#     drain_serial(ser, print_all=False)

# def run_one_hop(ser: serial.Serial, ppg: np.ndarray, fs: int, hop: int, n_samples: int):
#     dt = 1.0 / fs

#     # reset counters/state for clean run
#     if SEND_RST:
#         send_cmd(ser, "RST\n", settle=0.3, verbose=False)

#     # set hop
#     send_cmd(ser, f"HOP {hop}\n", settle=0.3, verbose=True)

#     # time scheduling
#     t0 = time.perf_counter()
#     next_t = t0

#     rows = []
#     start_wall = time.time()

#     for i, v in enumerate(ppg[:n_samples]):
#         ser.write(f"{float(v)}\n".encode())

#         # non-blocking-ish read
#         while ser.in_waiting:
#             s = ser.readline().decode(errors="ignore").strip()
#             if not s:
#                 continue

#             m = PWR_RE.search(s)
#             if m:
#                 d = m.groupdict()
#                 # coerce to floats (HOP might be missing)
#                 out = {}
#                 for k, val in d.items():
#                     if val is None:
#                         out[k] = np.nan
#                     else:
#                         out[k] = float(val)
#                 out["t_sec"] = time.time() - start_wall
#                 out["fs_run"] = fs
#                 out["hop_run"] = hop
#                 out["latency_s_theory"] = hop / fs
#                 rows.append(out)

#         # schedule next sample time precisely
#         next_t += dt
#         while True:
#             now = time.perf_counter()
#             remain = next_t - now
#             if remain <= 0:
#                 break
#             if remain > 0.003:
#                 time.sleep(remain - 0.002)
#             else:
#                 time.sleep(0.0003)

#     # final drain
#     time.sleep(0.3)
#     drain_serial(ser, print_all=False)

#     return pd.DataFrame(rows)

# def summarize_df(df: pd.DataFrame, warmup_sec: float):
#     if df.empty:
#         return None, None

#     df2 = df[df["t_sec"] >= warmup_sec].copy()
#     if df2.empty:
#         df2 = df.copy()

#     metrics = ["active", "infer_ms_s", "fill_ms_s", "rx_ms_s", "eproxy_ms_s", "samp", "inf", "dt"]
#     summary = df2[metrics].agg(["mean", "std", "count"]).T.reset_index()
#     summary.rename(columns={"index": "metric"}, inplace=True)
#     return df2, summary

# def plot_bars_hop(all_summaries: dict, fs_fixed: int, outpath: Path):
#     metrics_to_plot = ["active", "infer_ms_s", "rx_ms_s", "eproxy_ms_s", "inf"]
#     hops = sorted(all_summaries.keys())

#     fig, ax = plt.subplots(figsize=(11, 5))
#     width = 0.16
#     x = np.arange(len(hops))

#     for j, metric in enumerate(metrics_to_plot):
#         means, stds = [], []
#         for hop in hops:
#             s = all_summaries[hop]
#             row = s[s["metric"] == metric].iloc[0]
#             means.append(row["mean"])
#             stds.append(row["std"] if not np.isnan(row["std"]) else 0.0)

#         ax.bar(x + j * width, means, width=width, yerr=stds, capsize=3, label=metric)

#     ax.set_xticks(x + (len(metrics_to_plot) - 1) * width / 2)
#     ax.set_xticklabels([f"{h}" for h in hops])
#     ax.set_xlabel(f"HOP (samples)  [Fs fixed = {fs_fixed} Hz]")
#     ax.set_ylabel("Value (mean ± std over PWR windows)")
#     ax.set_title("MCU resource proxies vs HOP (from [PWR] logs)")
#     ax.legend(ncol=3, fontsize=9)
#     fig.tight_layout()
#     fig.savefig(outpath, dpi=200)
#     plt.show()

# def plot_latency_tradeoff(summary_table: pd.DataFrame, outpath: Path):
#     # x = latency, y = eproxy, annotate hop
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.scatter(summary_table["latency_s"], summary_table["eproxy_ms_s_mean"])
#     for _, r in summary_table.iterrows():
#         ax.annotate(str(int(r["hop"])), (r["latency_s"], r["eproxy_ms_s_mean"]))
#     ax.set_xlabel("Decision latency (s) = HOP/Fs")
#     ax.set_ylabel("Eproxy_ms/s (mean)")
#     ax.set_title("Latency vs Energy-proxy tradeoff (annotated by HOP)")
#     fig.tight_layout()
#     fig.savefig(outpath, dpi=200)
#     plt.show()

# def main():
#     ppg = load_ppg_csv(FILE)
#     print("Loaded samples:", len(ppg))

#     ser = serial.Serial(PORT, BAUD, timeout=0.1)
#     time.sleep(2)
#     ser.reset_input_buffer()
#     drain_serial(ser, print_all=False)

#     all_summaries = {}
#     summary_rows = []

#     for hop in HOP_LIST:
#         print(f"\n=== Running HOP={hop} (Fs={FS_FIXED}) ===")
#         df = run_one_hop(ser, ppg, FS_FIXED, hop, N_SAMPLES)

#         raw_csv = OUTDIR / f"raw_pwr_fs{FS_FIXED}_hop{hop}.csv"
#         df.to_csv(raw_csv, index=False)
#         print("Saved:", raw_csv)

#         df2, summary = summarize_df(df, WARMUP_SEC)
#         if summary is None:
#             print("No [PWR] lines captured. Check Arduino printing / regex / serial.")
#             continue

#         sum_csv = OUTDIR / f"summary_fs{FS_FIXED}_hop{hop}.csv"
#         summary.to_csv(sum_csv, index=False)
#         print("Saved:", sum_csv)

#         all_summaries[hop] = summary

#         # Build a compact table row for tradeoff plot + reporting
#         def get_mean(metric):
#             return float(summary[summary["metric"] == metric]["mean"].iloc[0])

#         row = {
#             "hop": hop,
#             "fs": FS_FIXED,
#             "latency_s": hop / FS_FIXED,
#             "active_mean": get_mean("active"),
#             "infer_ms_s_mean": get_mean("infer_ms_s"),
#             "rx_ms_s_mean": get_mean("rx_ms_s"),
#             "eproxy_ms_s_mean": get_mean("eproxy_ms_s"),
#             "inf_s_mean": get_mean("inf"),
#             "samp_s_mean": get_mean("samp"),
#             "n_pwr_windows": int(summary[summary["metric"] == "active"]["count"].iloc[0]),
#         }
#         summary_rows.append(row)

#     # plots
#     if all_summaries:
#         plot_bars_hop(all_summaries, FS_FIXED, OUTDIR / "compare_hop_bar.png")

#         trade = pd.DataFrame(summary_rows).sort_values("hop")
#         trade.to_csv(OUTDIR / "tradeoff_table.csv", index=False)
#         print("Saved:", OUTDIR / "tradeoff_table.csv")

#         plot_latency_tradeoff(trade, OUTDIR / "latency_vs_eproxy.png")

#         print("\nTradeoff table:\n", trade)

#     ser.close()

# if __name__ == "__main__":
#     main()

# import time
# import numpy as np
# import serial
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ============================
# # Config
# # ============================
# PORT = "COM7"
# BAUD = 115200
# FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"

# FS_FIXED = 64
# HOP_LIST = [384]
# BATCH_LIST = [1, 2, 4, 8]
# N_SAMPLES = 20000
# WARMUP_SEC = 6.0

# OUTDIR = Path("ppg_A2_runs")
# OUTDIR.mkdir(exist_ok=True)

# SEND_RST = True

# # ============================
# # Load PPG
# # ============================
# def load_ppg_csv(file_path: str) -> np.ndarray:
#     with open(file_path, "r") as f:
#         lines = [line.strip() for line in f if line.strip()]
#     vals = np.array([float(x) for x in lines], dtype=np.float32)
#     return vals[2:]

# # ============================
# # Robust Serial line reader (accumulator)
# # ============================
# class LineAccumulator:
#     def __init__(self):
#         self.buf = bytearray()

#     def feed(self, data: bytes):
#         self.buf.extend(data)

#     def pop_lines(self):
#         lines = []
#         while True:
#             idx = self.buf.find(b"\n")
#             if idx < 0:
#                 break
#             raw = self.buf[:idx].decode(errors="ignore").strip()
#             del self.buf[:idx+1]
#             if raw:
#                 lines.append(raw)
#         return lines

# def read_available_lines(ser: serial.Serial, acc: LineAccumulator, max_bytes: int = 4096):
#     n = ser.in_waiting
#     if n <= 0:
#         return []
#     data = ser.read(min(n, max_bytes))
#     acc.feed(data)
#     return acc.pop_lines()

# # ============================
# # Parse [PWR] line (tolerant)
# # ============================
# def parse_pwr_line(s: str):
#     if not s.startswith("[PWR]"):
#         return None

#     tokens = s.replace("[PWR]", "").strip().split()
#     d = {}
#     for tok in tokens:
#         if "=" not in tok:
#             continue
#         k, v = tok.split("=", 1)

#         # IMPORTANT: order matters!
#         v = (v.replace("/s", "")   # do this first
#                .replace("Hz", "")
#                .replace("%", "")
#                .replace("s", ""))  # do this last

#         try:
#             if k in ("HOP", "BATCH", "pend_peak"):
#                 d[k] = int(float(v))
#             else:
#                 d[k] = float(v)
#         except:
#             pass

#     if ("dt" not in d) or ("Fs" not in d) or ("active" not in d):
#         return None

#     # normalize keys
#     if "Eproxy_ms/s" in d:
#         d["eproxy_ms_s"] = d.pop("Eproxy_ms/s")
#     if "infer_ms/s" in d:
#         d["infer_ms_s"] = d.pop("infer_ms/s")
#     if "fill_ms/s" in d:
#         d["fill_ms_s"] = d.pop("fill_ms/s")
#     if "rx_ms/s" in d:
#         d["rx_ms_s"] = d.pop("rx_ms/s")

#     return d


# # ============================
# # Commands
# # ============================
# def send_cmd(ser: serial.Serial, cmd: str, acc: LineAccumulator, settle: float = 0.25, verbose: bool = True):
#     if verbose:
#         print(">>", cmd.strip())
#     ser.write(cmd.encode())
#     ser.flush()

#     # IMPORTANT: DO NOT "drain and discard" blindly.
#     # Instead, read lines for a short time and PRINT them (so you can confirm OK / errors).
#     t0 = time.time()
#     while time.time() - t0 < settle:
#         lines = read_available_lines(ser, acc)
#         for s in lines:
#             # show responses like "OK HOP 384"
#             if s.startswith("OK") or s.startswith("ERR") or s.startswith("[PWR]"):
#                 print("<<", s)
#         time.sleep(0.01)

# # ============================
# # Run one condition
# # ============================
# def run_one_condition(ser: serial.Serial, acc: LineAccumulator, ppg: np.ndarray,
#                       fs: int, hop: int, batch: int, n_samples: int):

#     dt = 1.0 / fs
#     rows = []
#     start_wall = time.time()

#     if SEND_RST:
#         send_cmd(ser, "RST\n", acc, settle=0.40, verbose=False)

#     send_cmd(ser, f"HOP {hop}\n", acc, settle=0.30, verbose=True)
#     send_cmd(ser, f"BATCH {batch}\n", acc, settle=0.30, verbose=True)

#     # small extra settle so the first [PWR] window is clean
#     time.sleep(0.2)

#     next_t = time.perf_counter()

#     for v in ppg[:n_samples]:
#         ser.write(f"{float(v)}\n".encode())

#         # read any output lines
#         for s in read_available_lines(ser, acc):
#             d = parse_pwr_line(s)
#             if d is not None:
#                 d["t_sec"] = time.time() - start_wall
#                 d["fs_run"] = fs
#                 d["hop_run"] = hop
#                 d["batch_run"] = batch
#                 d["latency_s_theory"] = (batch * hop) / fs
#                 rows.append(d)

#         # timing
#         next_t += dt
#         while True:
#             remain = next_t - time.perf_counter()
#             if remain <= 0:
#                 break
#             if remain > 0.003:
#                 time.sleep(remain - 0.002)
#             else:
#                 time.sleep(0.0003)

#     # final flush
#     time.sleep(0.5)
#     for s in read_available_lines(ser, acc):
#         d = parse_pwr_line(s)
#         if d is not None:
#             d["t_sec"] = time.time() - start_wall
#             d["fs_run"] = fs
#             d["hop_run"] = hop
#             d["batch_run"] = batch
#             d["latency_s_theory"] = (batch * hop) / fs
#             rows.append(d)

#     return pd.DataFrame(rows)

# # ============================
# # Summarize
# # ============================
# def summarize_df(df: pd.DataFrame, warmup_sec: float):
#     if df.empty:
#         return None, None

#     df2 = df[df["t_sec"] >= warmup_sec].copy()
#     if df2.empty:
#         df2 = df.copy()

#     metrics = ["active", "infer_ms_s", "fill_ms_s", "rx_ms_s", "eproxy_ms_s", "samp", "inf", "dt", "pend_peak"]
#     metrics = [m for m in metrics if m in df2.columns]

#     summary = df2[metrics].agg(["mean", "std", "count"]).T.reset_index()
#     summary.rename(columns={"index": "metric"}, inplace=True)
#     return df2, summary

# def _summary_get(summary: pd.DataFrame, metric: str, stat: str = "mean", default=np.nan):
#     s = summary[summary["metric"] == metric]
#     if s.empty:
#         return default
#     return float(s[stat].iloc[0])

# # ============================
# # Main
# # ============================
# def main():
#     ppg = load_ppg_csv(FILE)
#     print("Loaded samples:", len(ppg))

#     ser = serial.Serial(PORT, BAUD, timeout=0.0)  # non-blocking reads
#     time.sleep(2.0)
#     ser.reset_input_buffer()
#     ser.reset_output_buffer()

#     acc = LineAccumulator()

#     summary_rows = []

#     for hop in HOP_LIST:
#         for batch in BATCH_LIST:
#             print(f"\n=== Running HOP={hop}, BATCH={batch} (Fs={FS_FIXED}) ===")
#             df = run_one_condition(ser, acc, ppg, FS_FIXED, hop, batch, N_SAMPLES)

#             raw_csv = OUTDIR / f"raw_pwr_fs{FS_FIXED}_hop{hop}_batch{batch}.csv"
#             df.to_csv(raw_csv, index=False)
#             print("Saved:", raw_csv, "rows=", len(df))

#             df2, summary = summarize_df(df, WARMUP_SEC)
#             if summary is None:
#                 print("No [PWR] lines captured.")
#                 print("If you saw [PWR] in Serial Monitor, then Python wasn't receiving it.")
#                 continue

#             sum_csv = OUTDIR / f"summary_fs{FS_FIXED}_hop{hop}_batch{batch}.csv"
#             summary.to_csv(sum_csv, index=False)
#             print("Saved:", sum_csv)

#             row = {
#                 "hop": hop,
#                 "batch": batch,
#                 "fs": FS_FIXED,
#                 "latency_s": (batch * hop) / FS_FIXED,
#                 "active_mean": _summary_get(summary, "active", "mean"),
#                 "eproxy_ms_s_mean": _summary_get(summary, "eproxy_ms_s", "mean"),
#                 "inf_s_mean": _summary_get(summary, "inf", "mean"),
#                 "samp_s_mean": _summary_get(summary, "samp", "mean"),
#                 "pend_peak_mean": _summary_get(summary, "pend_peak", "mean"),
#                 "n_pwr_windows": int(_summary_get(summary, "active", "count", default=0)),
#             }
#             summary_rows.append(row)

#     trade = pd.DataFrame(summary_rows).sort_values(["hop", "batch"])
#     trade_csv = OUTDIR / "tradeoff_table_A2.csv"
#     trade.to_csv(trade_csv, index=False)
#     print("\nSaved:", trade_csv)
#     print("\nTradeoff table:\n", trade)

#     ser.close()

# if __name__ == "__main__":
#     main()



import time
import re
import struct
import numpy as np
import serial
import pandas as pd
from pathlib import Path
from typing import Literal, Optional

# ============================
# Config
# ============================
PORT = "COM7"
BAUD = 115200
FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"

FS_FIXED = 64
HOP = 384
BATCH = 1

BURST_LIST = [1, 8, 16, 64, 256]
#BURST_LIST = [256]

N_SAMPLES = 20000
WARMUP_SEC = 6.0

OUTDIR = Path("ppg_A1_uart_burst")
OUTDIR.mkdir(exist_ok=True)

SEND_RST = True
PRINT_NONPWR_DEBUG = False
DEBUG_PRINT_RAW_PWRS = False

# Choose streaming mode for samples:
#   "TEXT"  -> send ASCII "float\n"
#   "BIN_F32"-> send framed binary float32 (0xAA 0x55 0x01 0x04 payload chk)
#   "BIN_I8" -> send framed binary int8   (0xAA 0x55 0x02 0x01 payload chk)
STREAM_MODE: Literal["TEXT", "BIN_F32", "BIN_I8"] = "BIN_F32"

# If using BIN_I8, these must match the Arduino I8Q dequant config
# Arduino: float = (q - zp) * scale
I8_SCALE = 0.01
I8_ZP = 0

# ============================
# Arduino binary framing (must match your sketch)
# ============================
SYNC1 = 0xAA
SYNC2 = 0x55
MODE_F32 = 0x01
MODE_I8 = 0x02

# ============================
# Regex for Arduino [PWR] line
# ============================
PWR_RE = re.compile(
    r"\[PWR\]\s*dt=(?P<dt>[\d.]+)s\s+"
    r"Fs=(?P<Fs>[\d.]+)Hz\s+"
    r"HOP=(?P<HOP>\d+)\s+"
    r"BATCH=(?P<BATCH>\d+)\s+"
    r"pend_peak=(?P<pend_peak>\d+)\s+"
    r"samp=(?P<samp>[\d.]+)/s\s+"
    r"inf=(?P<inf>[\d.]+)/s\s+"
    r"active=(?P<active>[\d.]+)%\s+"
    r"infer_ms/s=(?P<infer_ms_s>[\d.]+)\s+"
    r"fill_ms/s=(?P<fill_ms_s>[\d.]+)\s+"
    r"rx_ms/s=(?P<rx_ms_s>[\d.]+)\s+"
    r"Eproxy_ms/s=(?P<eproxy_ms_s>[\d.]+)"
)


def load_ppg_csv(file_path: str) -> np.ndarray:
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    vals = []
    for x in lines:
        try:
            vals.append(float(x))
        except ValueError:
            continue

    vals = np.array(vals, dtype=np.float32)
    if len(vals) <= 2:
        raise ValueError("BVP file parsing produced too few numeric samples.")
    return vals[2:]


# ============================
# Robust serial line reader
# ============================
class SerialLineReader:
    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.buf = bytearray()

    def poll_lines(self, max_bytes: int = 16384):
        lines = []
        n = self.ser.in_waiting
        if n <= 0:
            return lines
        n = min(n, max_bytes)
        chunk = self.ser.read(n)
        if not chunk:
            return lines
        self.buf.extend(chunk)

        while True:
            idx = self.buf.find(b"\n")
            if idx < 0:
                break
            raw = self.buf[: idx + 1]
            del self.buf[: idx + 1]
            s = raw.decode(errors="ignore").strip()
            if s:
                lines.append(s)
        return lines


def drain_serial(reader: SerialLineReader, seconds: float = 0.3):
    t0 = time.time()
    while time.time() - t0 < seconds:
        _ = reader.poll_lines()
        time.sleep(0.01)


def send_cmd(ser: serial.Serial, reader: SerialLineReader, cmd: str, settle: float = 0.25, verbose: bool = True):
    if verbose:
        print(">>", cmd.strip())
    ser.write(cmd.encode())
    ser.flush()
    time.sleep(settle)
    drain_serial(reader, seconds=0.2)


def parse_pwr_lines(lines):
    rows = []
    for s in lines:
        m = PWR_RE.search(s)
        if not m:
            if PRINT_NONPWR_DEBUG and (s.startswith("[PWR]") or s.startswith("OK") or s.startswith("READY") or s.startswith("[MEM]")):
                print("DBG:", s)
            continue

        if DEBUG_PRINT_RAW_PWRS:
            print("RAW_PWR:", s)

        d = m.groupdict()
        row = {}
        for k, v in d.items():
            if k in ["HOP", "BATCH", "pend_peak"]:
                row[k] = int(v)
            else:
                row[k] = float(v)
        rows.append(row)
    return rows


def summarize_df(df: pd.DataFrame, warmup_sec: float):
    if df.empty:
        return None, None

    df2 = df[df["t_sec"] >= warmup_sec].copy()
    if df2.empty:
        df2 = df.copy()

    metrics = ["active", "infer_ms_s", "fill_ms_s", "rx_ms_s", "eproxy_ms_s", "samp", "inf", "dt", "pend_peak"]
    metrics = [m for m in metrics if m in df2.columns]

    summary = df2[metrics].agg(["mean", "std", "count"]).T.reset_index()
    summary.rename(columns={"index": "metric"}, inplace=True)
    return df2, summary


def _summary_get(summary: pd.DataFrame, metric: str, stat: str = "mean", default=np.nan):
    s = summary[summary["metric"] == metric]
    if s.empty:
        return default
    return float(s[stat].iloc[0])


def poll_and_append_pwr(reader: SerialLineReader, rows: list, start_wall: float, fs: int, hop: int, batch: int, burst: int):
    lines = reader.poll_lines()
    parsed = parse_pwr_lines(lines)
    if not parsed:
        return
    now_sec = time.time() - start_wall
    for r in parsed:
        r["t_sec"] = now_sec
        r["fs_run"] = fs
        r["hop_run"] = hop
        r["batch_run"] = batch
        r["burst_run"] = burst
        rows.append(r)


def wait_until_with_poll(reader: SerialLineReader, target_perf: float, rows: list, start_wall: float, fs: int, hop: int, batch: int, burst: int):
    while True:
        now = time.perf_counter()
        remain = target_perf - now
        if remain <= 0:
            break

        poll_and_append_pwr(reader, rows, start_wall, fs, hop, batch, burst)

        if remain > 0.01:
            time.sleep(0.008)
        elif remain > 0.003:
            time.sleep(remain - 0.001)
        else:
            time.sleep(0.0005)

    poll_and_append_pwr(reader, rows, start_wall, fs, hop, batch, burst)


# ============================
# Sample sending helpers (TEXT / BIN_F32 / BIN_I8)
# ============================
def _frame_bytes(mode: int, payload: bytes) -> bytes:
    ln = len(payload)
    chk = mode ^ ln
    for b in payload:
        chk ^= b
    return bytes([SYNC1, SYNC2, mode, ln]) + payload + bytes([chk])


def send_sample_text(ser: serial.Serial, v: float):
    ser.write(f"{float(v)}\n".encode())


def send_sample_bin_f32(ser: serial.Serial, v: float):
    payload = struct.pack("<f", float(v))
    ser.write(_frame_bytes(MODE_F32, payload))


def quantize_to_i8(v: float, scale: float, zp: int) -> int:
    # q = round(v/scale) + zp
    q = int(np.round(float(v) / float(scale))) + int(zp)
    return int(np.clip(q, -128, 127))


def send_sample_bin_i8(ser: serial.Serial, v: float, scale: float, zp: int):
    q = quantize_to_i8(v, scale, zp)
    payload = struct.pack("<b", q)  # signed int8
    ser.write(_frame_bytes(MODE_I8, payload))


def configure_stream_mode(ser: serial.Serial, reader: SerialLineReader, mode: Literal["TEXT", "BIN_F32", "BIN_I8"]):
    """
    IMPORTANT: your Arduino only parses commands in TEXT mode.
    So we:
      1) MODE TEXT
      2) (optional) I8Q for BIN_I8
      3) switch to desired mode
    """
    send_cmd(ser, reader, "MODE TEXT\n", settle=0.20, verbose=True)

    if mode == "BIN_I8":
        send_cmd(ser, reader, f"I8Q {I8_SCALE} {I8_ZP}\n", settle=0.20, verbose=True)

    if mode == "TEXT":
        # already in TEXT
        return

    if mode == "BIN_F32":
        send_cmd(ser, reader, "MODE BIN_F32\n", settle=0.20, verbose=True)
        return

    if mode == "BIN_I8":
        send_cmd(ser, reader, "MODE BIN_I8\n", settle=0.20, verbose=True)
        return

    raise ValueError(f"Unknown mode: {mode}")


# ============================
# A1 sender (absolute schedule)
# ============================
def run_a1_uart_burst(
    ser: serial.Serial,
    reader: SerialLineReader,
    ppg: np.ndarray,
    fs: int,
    burst: int,
    n_samples: int,
    stream_mode: Literal["TEXT", "BIN_F32", "BIN_I8"],
):
    # Safety: ensure enough runtime (>= ~12s) to see [PWR] after warmup
    min_samples = int(12 * fs)
    if n_samples < min_samples:
        print(f"WARNING: N_SAMPLES={n_samples} < {min_samples}; may not capture [PWR]. Increase N_SAMPLES.")

    if SEND_RST:
        send_cmd(ser, reader, "RST\n", settle=0.35, verbose=False)

    # 1) Always go through TEXT to set parameters
    send_cmd(ser, reader, "MODE TEXT\n", settle=0.20, verbose=True)

    # Optional: profiling-friendly output gating (if your Arduino supports it)
    # send_cmd(ser, reader, "PRED OFF\n", settle=0.10, verbose=False)
    # send_cmd(ser, reader, "MEM OFF\n", settle=0.10, verbose=False)

    send_cmd(ser, reader, f"HOP {HOP}\n", settle=0.25, verbose=True)
    send_cmd(ser, reader, f"BATCH {BATCH}\n", settle=0.25, verbose=True)

    # 2) Switch to requested stream mode (TEXT / BIN_F32 / BIN_I8)
    configure_stream_mode(ser, reader, stream_mode)

    drain_serial(reader, seconds=0.5)

    start_wall = time.time()
    start_perf = time.perf_counter()

    rows = []
    sent = 0
    n_samples = min(n_samples, len(ppg))

    i = 0
    while i < n_samples:
        k = min(burst, n_samples - i)
        chunk = ppg[i : i + k]

        # Send burst fast
        if stream_mode == "TEXT":
            for v in chunk:
                send_sample_text(ser, v)
        elif stream_mode == "BIN_F32":
            for v in chunk:
                send_sample_bin_f32(ser, v)
        elif stream_mode == "BIN_I8":
            for v in chunk:
                send_sample_bin_i8(ser, v, I8_SCALE, I8_ZP)
        else:
            raise ValueError(stream_mode)

        ser.flush()

        sent += k
        i += k

        # poll immediately
        poll_and_append_pwr(reader, rows, start_wall, fs, HOP, BATCH, burst)

        # enforce average Fs via absolute schedule
        target_t = start_perf + (sent / fs)
        wait_until_with_poll(reader, target_t, rows, start_wall, fs, HOP, BATCH, burst)

    # Final drain
    end_t = time.time() + 1.2
    while time.time() < end_t:
        poll_and_append_pwr(reader, rows, start_wall, fs, HOP, BATCH, burst)
        time.sleep(0.02)

    return pd.DataFrame(rows)


def main():
    ppg = load_ppg_csv(FILE)
    print("Loaded samples:", len(ppg))
    print("STREAM_MODE =", STREAM_MODE)

    ser = serial.Serial(PORT, BAUD, timeout=0.0)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    reader = SerialLineReader(ser)
    drain_serial(reader, seconds=0.6)

    trade_rows = []

    for burst in BURST_LIST:
        print(f"\n=== A1 UART burst: burst={burst} samples (avg Fs={FS_FIXED}) [{STREAM_MODE}] ===")
        df = run_a1_uart_burst(
            ser, reader, ppg, FS_FIXED, burst, N_SAMPLES, stream_mode=STREAM_MODE
        )

        raw_csv = OUTDIR / f"raw_pwr_mode{STREAM_MODE}_fs{FS_FIXED}_hop{HOP}_batch{BATCH}_burst{burst}.csv"
        df.to_csv(raw_csv, index=False)
        print("Saved:", raw_csv)

        _, summary = summarize_df(df, WARMUP_SEC)
        if summary is None:
            print("No [PWR] lines captured.")
            print("Likely causes: Arduino not printing [PWR] (TX blocked), regex mismatch, or run too short.")
            continue

        sum_csv = OUTDIR / f"summary_mode{STREAM_MODE}_fs{FS_FIXED}_hop{HOP}_batch{BATCH}_burst{burst}.csv"
        summary.to_csv(sum_csv, index=False)
        print("Saved:", sum_csv)

        trade_rows.append({
            "mode": STREAM_MODE,
            "burst": burst,
            "fs": FS_FIXED,
            "hop": HOP,
            "batch": BATCH,
            "active_mean": _summary_get(summary, "active"),
            "rx_ms_s_mean": _summary_get(summary, "rx_ms_s"),
            "infer_ms_s_mean": _summary_get(summary, "infer_ms_s"),
            "fill_ms_s_mean": _summary_get(summary, "fill_ms_s"),
            "eproxy_ms_s_mean": _summary_get(summary, "eproxy_ms_s"),
            "samp_s_mean": _summary_get(summary, "samp"),
            "inf_s_mean": _summary_get(summary, "inf"),
            "pend_peak_mean": _summary_get(summary, "pend_peak"),
            "n_pwr_windows": int(_summary_get(summary, "active", "count", default=0)),
        })

    trade = pd.DataFrame(trade_rows).sort_values("burst")
    trade_csv = OUTDIR / f"tradeoff_table_A1_uart_burst_mode{STREAM_MODE}.csv"
    trade.to_csv(trade_csv, index=False)

    print("\nSaved:", trade_csv)
    print("\nTradeoff table:\n", trade)

    ser.close()


if __name__ == "__main__":
    main()







