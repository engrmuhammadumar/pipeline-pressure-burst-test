import os, sys, struct, mmap, json, math
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

# Optional imports guarded
try:
    import h5py
except:
    h5py = None

try:
    from scipy.io import loadmat
except:
    loadmat = None

# ---------- Config you can edit later if needed ----------
DEFAULT_SAMPLE_RATE = None      # e.g., 10000 (Hz) if you know it
DEFAULT_CHANNELS = None         # e.g., ["pressure", "flow", "strain"]
DEFAULT_DTYPE = "float32"       # fallback for raw binary
DEFAULT_INTERLEAVED = True      # for raw binary: samples interleaved by channel
CHUNK_ROWS = 5_000_000          # ~5M samples per chunk; adjust to your RAM
DOWNSAMPLE_FACTOR = 10          # keep every Nth sample after anti-alias
TIME_COL = "t"
OUT_DIR = "out_parquet"
META_PATH = "out_parquet/_meta.json"
# ---------------------------------------------------------

def ensure_out():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def write_meta(meta: dict):
    ensure_out()
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def read_meta() -> Optional[dict]:
    p = Path(META_PATH)
    if not p.exists(): return None
    return json.loads(p.read_text())

def human(x):
    units = ["B","KB","MB","GB","TB"]
    i=0; f=float(x)
    while f>=1024 and i<len(units)-1:
        f/=1024; i+=1
    return f"{f:.2f} {units[i]}"

# ---------- Probing -----------------------------------------------------------

def looks_like_hdf5(fp: Path) -> bool:
    # HDF5 starts with \x89HDF\r\n\x1a\n
    with open(fp, "rb") as f:
        sig = f.read(8)
    return sig == b"\x89HDF\r\n\x1a\n"

def looks_like_mat(fp: Path) -> bool:
    # MATLAB v5 starts with "MATLAB 5.0 MAT-file"
    with open(fp, "rb") as f:
        sig = f.read(32)
    return b"MATLAB" in sig and b"MAT-file" in sig

def looks_like_text_csv(fp: Path, sample_bytes=2048) -> bool:
    with open(fp, "rb") as f:
        chunk = f.read(sample_bytes)
    # Heuristic: many commas/semicolons/newlines and printable ASCII fraction
    textish = sum(32 <= b <= 126 or b in (9,10,13) for b in chunk)/max(1,len(chunk))
    separators = chunk.count(b',') + chunk.count(b';') + chunk.count(b'\t')
    return textish > 0.9 and separators > 10

def probe(file_path: str) -> dict:
    fp = Path(file_path)
    size = fp.stat().st_size
    info = {"file": str(fp), "size_bytes": size, "size_human": human(size), "type": "unknown"}
    if looks_like_hdf5(fp):
        info["type"] = "hdf5"
        return info
    if looks_like_mat(fp):
        info["type"] = "matlab"
        return info
    if looks_like_text_csv(fp):
        info["type"] = "csv"
        return info
    # Try to sniff simple binary header with small ASCII header (first 4KB)
    with open(fp, "rb") as f:
        head = f.read(4096)
    if any(k in head for k in [b"CHANNELS", b"SAMPLERATE", b"SAMPLE_RATE", b"DTYPE"]):
        info["type"] = "bin+ascii_header"
    else:
        info["type"] = "raw_binary"
    return info

# ---------- Parsers -----------------------------------------------------------

def parse_hdf5(file_path: str) -> Tuple[List[str], int]:
    if h5py is None:
        raise RuntimeError("h5py not installed. pip install h5py")
    chans = []
    sr = DEFAULT_SAMPLE_RATE
    with h5py.File(file_path, "r") as h:
        # heuristic: datasets of shape (N,) or (N,C)
        def collect(ds_path, obj):
            nonlocal chans
            if isinstance(obj, h5py.Dataset):
                chans.append(ds_path)
        h.visititems(collect)
        # Try to read sampling rate from attrs if exists
        for k,v in h.attrs.items():
            if "sample" in k.lower() and "rate" in k.lower():
                try: sr = float(v)
                except: pass
    return chans, sr

def parse_matlab(file_path: str) -> Tuple[List[str], int]:
    if loadmat is None:
        raise RuntimeError("scipy not installed. pip install scipy")
    md = loadmat(file_path, simplify_cells=True)
    chans = []
    sr = DEFAULT_SAMPLE_RATE
    for k,v in md.items():
        if k.startswith("__"): continue
        if isinstance(v, (np.ndarray, list)):
            chans.append(k)
    # try to find sample rate key
    for k in md.keys():
        if "sample" in k.lower() and "rate" in k.lower():
            try: sr = float(md[k])
            except: pass
    return chans, sr

def parse_csv_chunked(file_path: str, sample_rate: Optional[float]):
    ensure_out()
    # Infer header and delimiter automatically with pandas
    it = pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False)
    total_rows = 0
    part = 0
    for df in it:
        total_rows += len(df)
        df = normalize_dataframe(df, sample_rate)
        outp = Path(OUT_DIR)/f"part_{part:04d}.parquet"
        df.to_parquet(outp, index=False)
        part += 1
    meta = {
        "source": file_path, "format": "csv",
        "sample_rate": sample_rate, "chunks": part, "rows": total_rows
    }
    write_meta(meta)
    print(f"[OK] CSV → Parquet. Rows={total_rows}, Parts={part}. Saved to {OUT_DIR}")

def parse_bin_ascii_header(file_path: str, sample_rate: Optional[float]):
    # Read first 64KB to find header lines
    with open(file_path, "rb") as f:
        head = f.read(65536)
    header_end = head.find(b"\n\n")
    if header_end == -1: header_end = head.find(b"\r\n\r\n")
    if header_end == -1: header_end = 4096
    header = head[:header_end].decode(errors="ignore")
    # Attempt to extract params
    sr = sample_rate
    dtype = DEFAULT_DTYPE
    channels = DEFAULT_CHANNELS
    interleaved = DEFAULT_INTERLEAVED
    for line in header.splitlines():
        L = line.strip().lower()
        if "sample_rate" in L or "samplerate" in L:
            try: sr = float(L.split("=")[-1])
            except: pass
        if "dtype" in L:
            dtype = L.split("=")[-1].strip()
        if "channels" in L and ":" in L:
            chraw = line.split(":")[-1].strip()
            channels = [c.strip() for c in chraw.split(",")]
        if "layout" in L and "interleaved" in L:
            interleaved = True
        if "layout" in L and "planar" in L:
            interleaved = False
    data_offset = header_end+2
    parse_raw_binary(file_path, sr, channels, dtype, interleaved, data_offset)

def parse_raw_binary(file_path: str,
                     sample_rate: Optional[float],
                     channels: Optional[List[str]],
                     dtype: str,
                     interleaved: bool,
                     data_offset: int = 0):
    ensure_out()
    fp = Path(file_path)
    bpp = np.dtype(dtype).itemsize
    file_size = fp.stat().st_size
    data_bytes = file_size - data_offset
    if channels is None:
        # assume 2 channels as a safe default; user can override
        channels = ["ch0", "ch1"]
    C = len(channels)
    frame_size = bpp * C
    n_samples_total = data_bytes // frame_size
    print(f"[INFO] raw binary: dtype={dtype}, C={C}, interleaved={interleaved}, samples={n_samples_total:,}")

    with open(fp, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        mm.seek(data_offset)

        part = 0
        rows_done = 0
        while rows_done < n_samples_total:
            take = min(CHUNK_ROWS, n_samples_total - rows_done)
            buf = mm.read(take * frame_size)
            arr = np.frombuffer(buf, dtype=dtype, count=take * C)
            if interleaved:
                arr = arr.reshape(-1, C)
            else:
                arr = arr.reshape(C, -1).T
            df = pd.DataFrame(arr, columns=channels)
            df = normalize_dataframe(df, sample_rate)
            outp = Path(OUT_DIR)/f"part_{part:04d}.parquet"
            df.to_parquet(outp, index=False)
            part += 1
            rows_done += take
            print(f"  wrote part {part}, rows {take:,} (total {rows_done:,}/{n_samples_total:,})")

        meta = {
            "source": file_path, "format": "raw_binary",
            "sample_rate": sample_rate, "channels": channels,
            "dtype": dtype, "interleaved": interleaved,
            "data_offset": data_offset, "chunks": part,
            "rows": int(n_samples_total)
        }
        write_meta(meta)
        print(f"[OK] Binary → Parquet complete. Parts={part}. Saved to {OUT_DIR}")

# ---------- Normalization / downsample ---------------------------------------

def normalize_dataframe(df: pd.DataFrame, sample_rate: Optional[float]) -> pd.DataFrame:
    # If a time column already exists, keep it. Otherwise synthesize if sample_rate known.
    df = df.copy()
    cols = list(df.columns)
    # Try to detect time column names
    time_candidates = [c for c in cols if c.lower() in ("t","time","timestamp")]
    if time_candidates:
        tcol = time_candidates[0]
        df.rename(columns={tcol: TIME_COL}, inplace=True)
    else:
        if sample_rate and TIME_COL not in df.columns:
            n = len(df)
            dt = 1.0/float(sample_rate)
            t = np.arange(n) * dt
            df.insert(0, TIME_COL, t)
    # Basic cleaning: drop completely constant junk columns
    nunique = df.nunique()
    drop_cols = [c for c in df.columns if nunique.get(c, 2) <= 1 and c != TIME_COL]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Downsample if too dense and time exists
    if DOWNSAMPLE_FACTOR and DOWNSAMPLE_FACTOR > 1:
        df = df.iloc[::DOWNSAMPLE_FACTOR, :].reset_index(drop=True)
        if TIME_COL in df.columns and sample_rate:
            # adjust time so it remains correct (keep absolute times from slicing)
            pass
    return df

# ---------- Orchestrator ------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python wfs_ingest.py \"D:\\Pipeline RUL Data\\data\\raw\\B.wfs\"")
        sys.exit(1)
    file_path = sys.argv[1]
    info = probe(file_path)
    print("[PROBE]", json.dumps(info, indent=2))

    sr = DEFAULT_SAMPLE_RATE
    if sr is None:
        # You can pass sample rate as 2nd arg if you know it
        if len(sys.argv) >= 3:
            try:
                sr = float(sys.argv[2])
            except:
                sr = None

    if info["type"] == "hdf5":
        chans, sr2 = parse_hdf5(file_path)
        if sr is None: sr = sr2
        print(f"[HDF5] Found datasets (first 10): {chans[:10]}")
        # Minimal export: user-specific; you can open datasets and stream to Parquet here
        print("For HDF5, adapt: open desired dataset(s), convert in chunks to Parquet using normalize_dataframe().")
        write_meta({"source": file_path, "format": "hdf5", "sample_rate": sr, "datasets": chans})
    elif info["type"] == "matlab":
        chans, sr2 = parse_matlab(file_path)
        if sr is None: sr = sr2
        print(f"[MAT] Variables (first 10): {chans[:10]}")
        print("For large MAT, use mat73 (HDF5-based) or split variables and write chunked Parquet.")
        write_meta({"source": file_path, "format": "matlab", "sample_rate": sr, "variables": chans})
    elif info["type"] == "csv":
        parse_csv_chunked(file_path, sr)
    elif info["type"] == "bin+ascii_header":
        parse_bin_ascii_header(file_path, sr)
    else:
        # raw_binary
        # If you know specifics, edit DEFAULT_* at top or pass via a sidecar JSON
        sidecar = Path(file_path).with_suffix(".json")
        if sidecar.exists():
            cfg = json.loads(sidecar.read_text())
            dtype = cfg.get("dtype", DEFAULT_DTYPE)
            channels = cfg.get("channels", DEFAULT_CHANNELS)
            inter = cfg.get("interleaved", DEFAULT_INTERLEAVED)
            data_offset = int(cfg.get("data_offset", 0))
            sr2 = cfg.get("sample_rate", sr)
            parse_raw_binary(file_path, sr2, channels, dtype, inter, data_offset)
        else:
            print("[ACTION REQUIRED] Unknown binary layout.")
            print("Create a sidecar JSON next to the file (same name, .json) with fields like:")
            print(json.dumps({
                "dtype": "float32",
                "channels": ["pressure","flow","strain"],
                "interleaved": True,
                "data_offset": 0,
                "sample_rate": 10000
            }, indent=2))
            print("Then rerun the script.")

if __name__ == "__main__":
    main()
