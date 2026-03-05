import time
import numpy as np
import pandas as pd
import serial


PORT = "COM7"         # Arduino port
BAUD = 115200
N = 512
FILE = "WESAD/S2/S2_E4_Data/BVP.csv"     
# Load data (assumes first column is PPG)
df = pd.read_csv(FILE)
ppg = df.iloc[:, 0].to_numpy(dtype=float)

if len(ppg) < N:
    raise ValueError(f"Not enough samples: {len(ppg)} < {N}")

window = ppg[:N]

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# Send 512 samples
for v in window:
    ser.write(f"{v}\n".encode())

# Trigger inference
ser.write(b"RUN\n")

# Read board output
t0 = time.time()
while time.time() - t0 < 5:
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print(line)
