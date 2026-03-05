import serial
import struct
import csv
import time
from collections import deque

FRAME_START = 0xAA
FRAME_END   = 0x55

FT_DATA  = 0x01
FT_START = 0x10
FT_STOP  = 0x11
FT_READY = 0x12

# DataPayload in Arduino:
# uint32 idx, uint32 t_us, uint32 red, uint32 ir, uint32 green, uint16 rb_used, uint8 flags
PAYLOAD_FMT = "<IIIIIHB"   # little-endian
PAYLOAD_LEN = struct.calcsize(PAYLOAD_FMT)

def checksum8(type_byte: int, len_byte: int, payload: bytes) -> int:
    x = 0
    x ^= type_byte & 0xFF
    x ^= len_byte & 0xFF
    for b in payload:
        x ^= b
    return x & 0xFF

def read_exact(ser: serial.Serial, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError("Serial read timeout.")
        buf += chunk
    return buf

def sync_to_start(ser: serial.Serial) -> None:
    # Find 0xAA in stream
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("Timeout syncing to start byte.")
        if b[0] == FRAME_START:
            return

def main(port="COM7", baud=921600, out_csv="ppg_log.csv",
         log_only_when_recording=False, flush_every_rows=200):
    """
    log_only_when_recording=False  -> ALWAYS write FT_DATA rows to CSV (recommended for debugging)
    log_only_when_recording=True   -> write rows only between START/STOP frames
    """
    ser = serial.Serial(port, baudrate=baud, timeout=1)
    ser.reset_input_buffer()
    print(f"[OK] Opened {port} @ {baud}")

    f = open(out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "pc_time_s", "recording",
        "idx", "t_us", "red", "ir", "green",
        "rb_used", "flags",
        "finger", "underflow", "overflow", "catchup"
    ])
    f.flush()

    # stats
    last_print = time.time()
    frames_ok = 0
    rows_written = 0
    bad_chk = 0
    bad_end = 0
    bad_len = 0
    timeouts = 0
    hz_window = deque(maxlen=300)

    recording = False

    try:
        while True:
            try:
                sync_to_start(ser)
                hdr = read_exact(ser, 2)
                ftype = hdr[0]
                flen  = hdr[1]

                payload = read_exact(ser, flen) if flen > 0 else b""
                chk = read_exact(ser, 1)[0]
                end = read_exact(ser, 1)[0]
            except TimeoutError:
                timeouts += 1
                continue

            if end != FRAME_END:
                bad_end += 1
                continue

            if chk != checksum8(ftype, flen, payload):
                bad_chk += 1
                continue

            # control frames
            if ftype == FT_READY:
                print("[ARDUINO] READY")
                continue
            if ftype == FT_START:
                recording = True
                print("[ARDUINO] START (recording=ON)")
                continue
            if ftype == FT_STOP:
                recording = False
                print("[ARDUINO] STOP (recording=OFF)")
                continue

            if ftype != FT_DATA:
                continue

            if flen != PAYLOAD_LEN:
                bad_len += 1
                continue

            idx, t_us, red, ir, green, rb_used, flags = struct.unpack(PAYLOAD_FMT, payload)

            finger    = (flags >> 0) & 1
            underflow = (flags >> 1) & 1
            overflow  = (flags >> 2) & 1
            catchup   = (flags >> 3) & 1

            pc_time = time.time()
            frames_ok += 1
            hz_window.append(t_us)

            # WRITE ROWS
            if (not log_only_when_recording) or recording:
                w.writerow([
                    pc_time, int(recording),
                    idx, t_us, red, ir, green,
                    rb_used, flags,
                    finger, underflow, overflow, catchup
                ])
                rows_written += 1

                if rows_written % flush_every_rows == 0:
                    f.flush()

            # periodic stats
            now = time.time()
            if now - last_print >= 1.0:
                hz = 0.0
                if len(hz_window) >= 20:
                    dt_us = hz_window[-1] - hz_window[0]
                    if dt_us > 0:
                        hz = (len(hz_window) - 1) * 1e6 / dt_us

                print(
                    f"[STATS] ok_frames={frames_ok}/s  rows_written={rows_written}  "
                    f"rec={'ON' if recording else 'OFF'}  hz~{hz:.2f}  "
                    f"bad_chk={bad_chk} bad_end={bad_end} bad_len={bad_len} timeouts={timeouts}"
                )

                frames_ok = 0
                bad_chk = 0
                bad_end = 0
                bad_len = 0
                timeouts = 0
                last_print = now
                f.flush()

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    # For debugging: always log data even if START never arrives
    main(port="COM7", baud=921600, out_csv="ppg_log.csv",
         log_only_when_recording=False, flush_every_rows=200)
