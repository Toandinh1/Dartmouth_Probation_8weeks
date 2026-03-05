import time, threading, struct
import numpy as np
import serial

PORT = "COM7"
BAUD = 115200
FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"
fs = 64
NSEND = 5000

# -------- load data --------
with open(FILE, "r") as f:
    vals = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
ppg = vals[2:]
print("fs:", fs, "samples:", len(ppg))

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)
ser.reset_input_buffer()
ser.reset_output_buffer()

# -------- reader thread --------
stop = False
def reader():
    while not stop:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(line)
        except Exception:
            pass

t = threading.Thread(target=reader, daemon=True)
t.start()

# -------- accurate scheduler (no drift) --------
dt = 1.0 / fs
t0 = time.perf_counter()
next_t = t0

for i, v in enumerate(ppg[:NSEND]):
    # send one sample (ASCII)
    ser.write(f"{float(v)}\n".encode())

    # schedule next send time (drift-free)
    next_t += dt
    now = time.perf_counter()
    remaining = next_t - now
    if remaining > 0:
        # sleep most of it, then small spin for accuracy
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        while time.perf_counter() < next_t:
            pass

stop = True
time.sleep(0.2)
