import json, mmap, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Optional but recommended for high-quality anti-aliasing
try:
    from scipy.signal import resample_poly
except:
    resample_poly = None

# ---------- Config ----------
SIDE_CAR = r"D:\Pipeline RUL Data\data\raw\B.json"
RAW_FILE = r"D:\Pipeline RUL Data\data\raw\B.wfs"
OUT_DIR = r"D:\Pipeline RUL Data\out_parquet_20k"
TIME_COL = "t"

# Streaming / memory settings
FRAMES_PER_CHUNK = 10_000_000   # 10M frames @ 1MHz ~ 0.01 s of 8ch data? (No: 10M frames = 10 s)
# 10M frames = 10 seconds worth of data; adjust up/down by your RAM/SSD speed
DOWNSAMPLE = 50                 # 1,000,000 Hz -> 20,000 Hz
SCALE_INT16 = 32768.0           # map int16 -> [-1,1]

# FIR edge handling: keep a small overlap so the filter at chunk edges is clean
EDGE_OVERLAP_FRAMES = 10_000    # 10k frames ~ 10 ms at 1MHz
# ----------------------------

def load_sidecar(path):
    sc = json.loads(Path(path).read_text())
    dtype = np.dtype('<i2') if sc.get("endianness","little")=="little" else np.dtype('>i2')
    return dict(
        dtype=dtype,
        channels=sc["channels"],
        interleaved=bool(sc["interleaved"]),
        data_offset=int(sc.get("data_offset",0)),
        sample_rate=int(sc["sample_rate"])
    )

def main():
    sc = load_sidecar(SIDE_CAR)
    C = len(sc["channels"])
    sr = sc["sample_rate"]
    assert sr == 1_000_000, f"Sidecar sample_rate={sr}, expected 1_000_000."

    outp = Path(OUT_DIR)
    outp.mkdir(parents=True, exist_ok=True)

    bpp = np.dtype(sc["dtype"]).itemsize
    frame_size = bpp * C

    total_bytes = Path(RAW_FILE).stat().st_size
    data_bytes = total_bytes - sc["data_offset"]
    total_frames = data_bytes // frame_size

    print(f"[INFO] total_frames={total_frames:,}  (~{total_frames/sr:.2f} s), channels={C}, dtype={sc['dtype']}")
    print(f"[INFO] Writing Parquet to: {OUT_DIR}")

    # For chunking, include overlap so polyphase filtering doesn't distort edges
    step = FRAMES_PER_CHUNK
    part = 0
    t0 = 0.0
    dec_sr = sr // DOWNSAMPLE
    carry_tail = None  # tail from previous chunk for edge continuity

    with open(RAW_FILE, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        pos = sc["data_offset"]
        frames_done = 0

        while frames_done < total_frames:
            take = min(step, total_frames - frames_done)
            # Extend read for overlap at end (except final)
            take_with_overlap = take + (EDGE_OVERLAP_FRAMES if (frames_done + take) < total_frames else 0)

            # Read bytes
            mm.seek(pos)
            buf = mm.read(take_with_overlap * frame_size)
            arr = np.frombuffer(buf, dtype=sc["dtype"], count=take_with_overlap * C)

            # Reshape to (frames, C)
            if sc["interleaved"]:
                arr = arr.reshape(-1, C)
            else:
                arr = arr.reshape(C, -1).T

            # If we have a carry_tail from the previous chunk, prepend it
            if carry_tail is not None:
                arr = np.vstack([carry_tail, arr])

            # Convert to float [-1,1]
            arr = arr.astype(np.float32) / SCALE_INT16

            # Anti-alias + decimate
            if resample_poly is None:
                # Fallback: pick every Nth sample (no anti-alias). Prefer to install SciPy!
                print("[WARN] scipy not installed; decimating without anti-alias. Run: pip install scipy")
                arr_dec = arr[::DOWNSAMPLE, :]
            else:
                # up=1, down=DOWNSAMPLE; axis=0 over time
                arr_dec = resample_poly(arr, up=1, down=DOWNSAMPLE, axis=0, window=('kaiser', 5.0))

            # Compute time vector for the *valid, non-overlap* region of the current chunk
            # Determine what portion corresponds to the "true current chunk" (exclude the end-overlap we added)
            # Original (pre-decimation) frames from this read:
            frames_in_this_block = (carry_tail.shape[0] if carry_tail is not None else 0) + take_with_overlap
            # But only 'take' of those belong to this part (the rest EDGE_OVERLAP_FRAMES are overlap for the next part)
            # Ratio to decimated index:
            dec_len = arr_dec.shape[0]
            # Simple proportional split:
            dec_for_real = int(round(dec_len * ( ( (carry_tail.shape[0] if carry_tail is not None else 0) + take ) / frames_in_this_block )))
            arr_dec_real = arr_dec[:dec_for_real, :]

            # Prepare DataFrame
            t = np.arange(arr_dec_real.shape[0], dtype=np.float64) / dec_sr + t0
            df = pd.DataFrame(arr_dec_real, columns=sc["channels"])
            df.insert(0, "t", t)

            # Write parquet
            out_file = outp / f"part_{part:04d}.parquet"
            df.to_parquet(out_file, index=False)
            print(f"[WRITE] {out_file.name}  rows={len(df):,}   t=[{t[0]:.3f} .. {t[-1]:.3f}] s")

            # Prepare carry_tail for NEXT iteration: the last EDGE_OVERLAP_FRAMES of *this* chunk (pre-decimation)
            # Take the last EDGE_OVERLAP_FRAMES of the raw-after-prepend array to keep continuity.
            if EDGE_OVERLAP_FRAMES > 0:
                carry_tail = arr[-EDGE_OVERLAP_FRAMES:, :].copy()
            else:
                carry_tail = None

            # Update counters
            part += 1
            frames_done += take
            pos += take * frame_size
            t0 = t[-1] + 1.0/dec_sr  # next start time continues from here

    # Write metadata
    meta = {
        "source": RAW_FILE,
        "format": "raw_binary_int16_interleaved",
        "channels": sc["channels"],
        "sample_rate_in": sr,
        "sample_rate_out": dec_sr,
        "downsample": DOWNSAMPLE,
        "parts": part,
        "time_col": TIME_COL
    }
    Path(OUT_DIR, "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Ingest complete. Parts={part}. Out SR={dec_sr} Hz. Meta written to _meta.json")

if __name__ == "__main__":
    main()
