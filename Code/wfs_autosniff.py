import mmap, os, json, math, sys
from pathlib import Path
import numpy as np

FILE_PATH = r"D:\Pipeline RUL Data\data\raw\B.wfs"  # change if needed

# combos to try
DTYPES = ["<f4", ">f4", "<i2", ">i2", "<i4", ">i4"]  # little/big endian float32, int16, int32
CHANNELS_TRY = [1, 2, 3, 4, 8]
INTERLEAVED_TRY = [True, False]
OFFSETS = [0, 512, 1024, 4096, 65536]
SAMPLE_RATE_CANDIDATES = [1000000]

# how many frames to score (keep small to be fast)
FRAMES = 5_000_000  # reduce if RAM limited
MIN_FRAMES = 200_000

def human(x):
    units = ["B","KB","MB","GB","TB"]; i=0; f=float(x)
    while f>=1024 and i<len(units)-1: f/=1024; i+=1
    return f"{f:.2f} {units[i]}"

def score_candidate(buf, dtype, C, interleaved) -> dict:
    bpp = np.dtype(dtype).itemsize
    nvals = len(buf) // bpp
    if nvals < C*MIN_FRAMES: 
        return {"ok": False, "reason": "too short"}

    take = min(FRAMES, nvals//C)
    arr = np.frombuffer(buf[:take*C*bpp], dtype=dtype, count=take*C)
    try:
        if interleaved:
            arr = arr.reshape(-1, C)
        else:
            arr = arr.reshape(C, -1).T
    except Exception:
        return {"ok": False, "reason": "reshape fail"}

    # basic stats per channel
    ch_stats = []
    pen_nan = 0.0
    pen_sat = 0.0
    pen_zero = 0.0
    reward_auto = 0.0
    for k in range(C):
        x = arr[:,k]
        finite = np.isfinite(x)
        frac_finite = float(np.mean(finite))
        if frac_finite < 0.999:
            pen_nan += (1.0 - frac_finite)*100

        # saturation check for integer types
        if np.issubdtype(np.dtype(dtype), np.integer):
            info = np.iinfo(np.dtype(dtype))
            near_sat = np.mean((x<=info.min+1) | (x>=info.max-1))
            pen_sat += near_sat * 50.0

        std = float(np.std(x)) + 1e-12
        mean = float(np.mean(x))
        if std < 1e-9:
            pen_zero += 20.0  # constant channel
        # lag-1 autocorrelation (expected positive for physical signals)
        x0 = x[:-1]; x1 = x[1:]
        if len(x0) > 1_000:
            corr = float(np.corrcoef(x0, x1)[0,1])
        else:
            corr = 0.0
        reward_auto += max(0.0, min(corr, 0.999)) * 2.0

        ch_stats.append(dict(mean=mean, std=std, corr=corr))

    # inter-channel correlation penalty (too similar could mean wrong C)
    inter_corr_pen = 0.0
    if C >= 2:
        from itertools import combinations
        for i,j in combinations(range(C),2):
            vi = arr[:200000,i]; vj = arr[:200000,j]
            try:
                cc = float(np.corrcoef(vi, vj)[0,1])
                if cc > 0.98:  # suspiciously duplicated
                    inter_corr_pen += (cc-0.98)*20
            except Exception:
                pass

    # spectral sanity: strong single-bin spikes suggest endian/dtype error
    spec_pen = 0.0
    try:
        x = arr[:1_000_000,0]
        x = x - np.mean(x)
        X = np.fft.rfft(x)
        mag = np.abs(X) + 1e-12
        peak = float(np.max(mag))
        med = float(np.median(mag))
        if med == 0: med = 1e-12
        peak_to_med = peak/med
        if peak_to_med > 1e4:  # too spiky likely garbage parse
            spec_pen += math.log10(peak_to_med)
    except Exception:
        pass

    # final heuristic score
    score = reward_auto - (pen_nan + pen_sat + pen_zero + inter_corr_pen + spec_pen)
    return {"ok": True, "score": score, "stats": ch_stats}

def guess_sample_rate(arr_first_ch):
    # try to detect 50/60 Hz line if present
    x = arr_first_ch[:1_000_000].astype(float)
    x -= x.mean()
    if len(x) < 4000:
        return None
    X = np.fft.rfft(x)
    mag = np.abs(X) + 1e-12
    # For each candidate sr, compute bins for 50/60Hz and see if energy is high
    best_sr = None; best_val = -1
    for sr in SAMPLE_RATE_CANDIDATES:
        freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
        # find bin indices closest to 50 and 60 Hz
        i50 = int(np.argmin(np.abs(freqs-50.0)))
        i60 = int(np.argmin(np.abs(freqs-60.0)))
        # look at a small neighborhood
        rng = 3
        e50 = float(np.max(mag[max(0,i50-rng):i50+rng+1]))
        e60 = float(np.max(mag[max(0,i60-rng):i60+rng+1]))
        baseline = float(np.median(mag))
        val = max(e50, e60) / (baseline + 1e-12)
        if val > best_val:
            best_val = val
            best_sr = sr
    # if nothing stands out, fall back to 10000
    if best_val < 5.0:
        return 10000
    return best_sr

def main():
    fp = Path(FILE_PATH)
    size = fp.stat().st_size
    print(f"[INFO] File: {fp}  size={human(size)}")

    best = None
    best_details = None

    with open(fp, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        for off in OFFSETS:
            if off >= size: 
                continue
            for dt in DTYPES:
                bpp = np.dtype(dt).itemsize
                tail = size - off
                if tail < bpp * MIN_FRAMES:
                    continue
                # read a small slab for quick rejection
                mm.seek(off)
                slab = mm.read(min(tail, bpp * FRAMES * 2))
                for C in CHANNELS_TRY:
                    for inter in INTERLEAVED_TRY:
                        res = score_candidate(slab, dt, C, inter)
                        if not res["ok"]:
                            continue
                        sc = res["score"]
                        if (best is None) or (sc > best):
                            best = sc
                            best_details = dict(offset=off, dtype=dt, channels=C, interleaved=inter, stats=res["stats"])

    if best_details is None:
        print("[FAIL] Could not find a plausible layout. Try increasing FRAMES or adding other dtypes.")
        sys.exit(1)

    # sample rate guess using first channel under best layout
    with open(fp, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        off = best_details["offset"]; dt = best_details["dtype"]; C = best_details["channels"]; inter = best_details["interleaved"]
        bpp = np.dtype(dt).itemsize
        mm.seek(off)
        take_vals = min((size-off)//bpp, FRAMES*C)
        buf = mm.read(take_vals*bpp)
        arr = np.frombuffer(buf, dtype=dt, count=take_vals)
        try:
            if inter:
                arr = arr.reshape(-1, C)
            else:
                arr = arr.reshape(C, -1).T
        except Exception:
            pass
        sr_guess = guess_sample_rate(arr[:,0])

    # pretty channel names
    chans = [f"ch{k}" for k in range(best_details["channels"])]

    sidecar = {
        "dtype": {"<f4":"float32", ">f4":"float32", "<i2":"int16", ">i2":"int16", "<i4":"int32", ">i4":"int32"}[best_details["dtype"]],
        "endianness": "little" if best_details["dtype"].startswith("<") else "big",
        "channels": chans,
        "interleaved": best_details["interleaved"],
        "data_offset": int(best_details["offset"]),
        "sample_rate": int(sr_guess)
    }

    print("\n[RECOMMENDED SIDECAR JSON]\n" + json.dumps(sidecar, indent=2))
    out_path = Path(fp.with_suffix(".json"))
    out_path.write_text(json.dumps(sidecar, indent=2))
    print(f"\n[OK] Wrote sidecar to {out_path}")
    print("\n[NOTE] You can rename channels later (e.g., ch0->pressure, ch1->flow).")

if __name__ == "__main__":
    main()
