import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

IN_DIR  = r"D:\Pipeline RUL Data\out_parquet_20k"
OUT_DIR = r"D:\Pipeline RUL Data\windows_parquet_20k"
TIME_COL = "t"

WINDOW_SECONDS = 1.0
STRIDE_SECONDS = 0.5
T_FAILURE = None
KEEP_CHANNELS = None
SPLIT = (0.60, 0.20, 0.20)

def load_meta():
    return json.loads(Path(IN_DIR, "_meta.json").read_text())

def iter_parts():
    return sorted(Path(IN_DIR).glob("part_*.parquet"))

def feats_for_window(w_df, sr, chans):
    feats = {}
    t_end = float(w_df[TIME_COL].iloc[-1]); feats["t_end"] = t_end
    for c in chans:
        x = w_df[c].values.astype(np.float32)
        m = float(np.mean(x)); s = float(np.std(x)) + 1e-12
        feats[f"{c}_mean"] = m
        feats[f"{c}_std"]  = s
        feats[f"{c}_min"]  = float(np.min(x))
        feats[f"{c}_max"]  = float(np.max(x))
        feats[f"{c}_rms"]  = float(np.sqrt(np.mean(x*x)))
        thr = 5.0*np.median(np.abs(x)) + 1e-12
        feats[f"{c}_count_over_thr"] = int(np.sum(np.abs(x) > thr))
        X = np.fft.rfft(x - m); mag = np.abs(X) + 1e-12
        freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
        feats[f"{c}_spec_centroid"] = float(np.sum(freqs*mag)/np.sum(mag))
        b1 = (freqs <= 5000); b2 = (freqs > 5000) & (freqs <= 10000)
        feats[f"{c}_E_0_5k"]  = float(np.sum(mag[b1]))
        feats[f"{c}_E_5_10k"] = float(np.sum(mag[b2]))
        feats[f"{c}_kurt"] = float(np.mean(((x-m)/s)**4))
    return feats

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    sr = int(meta["sample_rate_out"])
    chans = meta["channels"] if KEEP_CHANNELS is None else KEEP_CHANNELS

    if T_FAILURE is None:
        last_t = None
        for p in iter_parts():
            df = pd.read_parquet(p, columns=[TIME_COL])
            last_t = df[TIME_COL].iloc[-1]
        t_fail = float(last_t)
    else:
        t_fail = float(T_FAILURE)

    t_start = 0.0
    t_end_global = t_fail
    t_train_end = t_start + SPLIT[0]*(t_end_global - t_start)
    t_val_end   = t_train_end + SPLIT[1]*(t_end_global - t_start)

    win_len = int(round(WINDOW_SECONDS * sr))
    stride  = int(round(STRIDE_SECONDS * sr))

    carry = pd.DataFrame()
    rows_train, rows_val, rows_test = [], [], []

    for p in tqdm(iter_parts(), desc="Windowing"):
        df = pd.read_parquet(p)
        cols = [TIME_COL] + [c for c in df.columns if c in chans]
        df = df[cols]
        df = pd.concat([carry, df], axis=0, ignore_index=True)
        n = len(df)

        local = []
        start = 0
        while start + win_len <= n:
            w = df.iloc[start:start+win_len]
            feats = feats_for_window(w, sr, chans)
            t_e = feats["t_end"]
            rul = max(t_fail - t_e, 0.0)
            feats["RUL"] = float(rul)
            local.append(feats)
            start += stride

        if local:
            out_df = pd.DataFrame(local)
            # split by time
            rows_train.append(out_df[out_df["t_end"] <= t_train_end])
            rows_val.append(out_df[(out_df["t_end"] > t_train_end) & (out_df["t_end"] <= t_val_end)])
            rows_test.append(out_df[out_df["t_end"] > t_val_end])

        carry = df.iloc[max(0, n - (win_len - stride)):].reset_index(drop=True)

    # Concatenate and write once
    train_df = pd.concat(rows_train, ignore_index=True)
    val_df   = pd.concat(rows_val, ignore_index=True)
    test_df  = pd.concat(rows_test, ignore_index=True)

    train_df.to_parquet(Path(OUT_DIR, "train.parquet"), index=False)
    val_df.to_parquet(Path(OUT_DIR, "val.parquet"), index=False)
    test_df.to_parquet(Path(OUT_DIR, "test.parquet"), index=False)

    meta2 = {
        "source_meta": str(Path(IN_DIR, "_meta.json")),
        "sample_rate": sr,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "t_failure": t_fail,
        "channels": chans,
        "splits": {"train_end": t_train_end, "val_end": t_val_end},
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    }
    Path(OUT_DIR, "_meta_windows.json").write_text(json.dumps(meta2, indent=2))
    print(f"[OK] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"[INFO] Saved under {OUT_DIR}")

if __name__ == "__main__":
    main()
