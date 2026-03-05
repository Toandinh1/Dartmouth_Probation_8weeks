import time
import numpy as np
import serial

PORT = "COM7"
BAUD = 115200
FILE = r"WESAD\S2\S2_E4_Data\BVP.csv"

WIN_SEC = 8
OVERLAP_SEC = 2

# Load WESAD BVP
with open(FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
vals = np.array([float(x) for x in lines], dtype=np.float32)

fs = int(vals[1])
ppg = vals[2:]

N = fs * WIN_SEC
STEP = fs * (WIN_SEC - OVERLAP_SEC)

w1 = ppg[0:N]
w2 = ppg[STEP:STEP+N]

print("fs:", fs, "N:", N, "STEP:", STEP)

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

def read_lines(seconds=2.0):
    t0 = time.time()
    out = []
    while time.time() - t0 < seconds:
        s = ser.readline().decode(errors="ignore").strip()
        if s:
            out.append(s)
    return out

def send_window(win):
    # Send exactly N samples
    for v in win:
        ser.write(f"{float(v)}\n".encode())
    # Trigger
    ser.write(b"RUN\n")
    return read_lines(seconds=2.0)

print("\n--- Window 1 ---")
out1 = send_window(w1)
print("\n".join(out1))

print("\n--- Window 2 ---")
out2 = send_window(w2)
print("\n".join(out2))
