"""
shse_tools.py - read lab .shse leak files and run the week-1 feasibility check.

File layout (from the lab's .shse format sheet):
  header : 14 B char datetime | 8 B double FS | 1 B uint8 N channels | 8 B double samples/channel
  data   : 1-second blocks; inside each block, channel 1..N, each FS doubles (little-endian)

Usage:
  python shse_tools.py path/to/file.shse                 # summary + feasibility check
  python shse_tools.py path/to/file.shse --seconds 0 10   # only read 0-10 s
"""
import argparse
import itertools
import struct

import numpy as np
from scipy import signal

HEADER_BYTES = 14 + 8 + 1 + 8
VERSION = "shse_tools v5 (2026-10-12): + --events burst TDOA"


def read_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_BYTES)
    dt = raw[:14].decode("ascii", errors="replace")
    fs, = struct.unpack("<d", raw[14:22])
    nch = raw[22]
    nspc, = struct.unpack("<d", raw[23:31])
    return {"datetime": dt, "fs": int(fs), "n_channels": nch,
            "samples_per_channel": int(nspc), "seconds": int(nspc // fs)}


def read_shse(path, t0=0, t1=None):
    """Return (x, fs, header); x has shape (n_channels, n_samples). Reads whole seconds [t0, t1)."""
    h = read_header(path)
    fs, nch = h["fs"], h["n_channels"]
    t1 = h["seconds"] if t1 is None else min(t1, h["seconds"])
    n_sec = t1 - t0
    block = nch * fs
    data = np.fromfile(path, dtype="<f8", count=n_sec * block,
                       offset=HEADER_BYTES + t0 * block * 8)
    n_sec = data.size // block                      # guard against a truncated last block
    x = data[: n_sec * block].reshape(n_sec, nch, fs).transpose(1, 0, 2).reshape(nch, -1)
    return x, fs, h


def gcc_phat_tdoa(a, b, fs, max_lag_s=2e-3, band=(50e3, 400e3), min_lag_s=0.0):
    """TDOA of a relative to b (seconds; positive = a arrives later) via GCC-PHAT in a band.

    min_lag_s > 0 masks lags |tau| < min_lag_s, so a zero-lag common-mode (electrical) component
    cannot win the peak search.
    """
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    R[(f < band[0]) | (f > band[1])] = 0
    cc = np.fft.irfft(R, n)
    m = int(max_lag_s * fs)
    cc = np.abs(np.concatenate((cc[-m:], cc[: m + 1])))
    k = int(min_lag_s * fs)
    if k > 0:
        cc[m - k: m + k + 1] = 0
    return (np.argmax(cc) - m) / fs


def bandpass(x, fs, band):
    b, a = signal.butter(4, band, btype="band", fs=fs)
    return signal.filtfilt(b, a, x, axis=-1)


def timeline(path, band=(50e3, 400e3), step=1):
    """Per-second band RMS of every channel over the whole file -> shows when the leak is on.
    Writes <file>_timeline.csv (and a .png if matplotlib is available)."""
    h = read_header(path)
    rows = []
    for t in range(0, h["seconds"], step):
        x, fs, _ = read_shse(path, t, t + 1)
        if x.shape[1] == 0:
            break
        rows.append([t] + list(np.sqrt(np.mean(bandpass(x, fs, band) ** 2, axis=1))))
    R = np.array(rows)
    out = path.rsplit(".", 1)[0] + "_timeline.csv"
    np.savetxt(out, R, delimiter=",", header="t_s," + ",".join(f"ch{c+1}" for c in range(R.shape[1] - 1)))
    print(f"\nband RMS per second ({band[0]/1e3:.0f}-{band[1]/1e3:.0f} kHz), every 10th second:")
    for r in R[::10]:
        print(f"  t={int(r[0]):4d}s  " + "  ".join(f"{v:.2e}" for v in r[1:]))
    print("saved:", out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 3.5))
        for c in range(1, R.shape[1]):
            plt.semilogy(R[:, 0], R[:, c], lw=0.8, label=f"ch{c}")
        plt.xlabel("time [s]"); plt.ylabel("band RMS"); plt.legend(); plt.grid(alpha=.3)
        plt.title(path.replace("\\", "/").rsplit("/", 1)[-1]); plt.tight_layout()
        plt.savefig(out.replace(".csv", ".png"), dpi=120)
        print("saved:", out.replace(".csv", ".png"))
    except ImportError:
        pass


def feasibility_check(x, fs, win_s=0.05, n_win=200, band=(50e3, 400e3), min_lag_us=3.0):
    """For every sensor pair: coherence, its strongest frequencies, and window-wise GCC-PHAT TDOAs.

    Reports TDOAs twice: unmasked, and with |lag| < min_lag_us masked. If the unmasked TDOA is
    ~0 for every pair, a common-mode (electrical) component dominates and must be removed first.
    """
    xf = bandpass(x, fs, band)
    L = int(win_s * fs)
    starts = np.linspace(0, xf.shape[1] - L, min(n_win, xf.shape[1] // L)).astype(int)
    print(f"\n{'pair':>8} {'coh':>6}  {'top coherent freqs [kHz]':<30}"
          f"{'TDOA med [us]':>14} {'IQR':>7} | {'masked med':>11} {'IQR':>7}")
    for i, j in itertools.combinations(range(x.shape[0]), 2):
        f, C = signal.coherence(xf[i], xf[j], fs=fs, nperseg=8192)
        inb = (f >= band[0]) & (f <= band[1])
        coh = C[inb].mean()
        top = f[inb][np.argsort(C[inb])[-5:][::-1]] / 1e3
        res = []
        for ml in (0.0, min_lag_us * 1e-6):
            d = np.array([gcc_phat_tdoa(xf[i, s:s + L], xf[j, s:s + L], fs, band=band, min_lag_s=ml)
                          for s in starts]) * 1e6
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            res += [med, q3 - q1]
        tops = ",".join(f"{v:.0f}" for v in top)
        print(f"  ch{i+1}-ch{j+1} {coh:6.3f}  {tops:<30}{res[0]:14.1f} {res[1]:7.1f} | {res[2]:11.1f} {res[3]:7.1f}")

    # common-mode diagnostic: how much of each channel is explained by the mean of all channels
    cm = xf.mean(axis=0)
    share = [np.corrcoef(xf[c], cm)[0, 1] ** 2 for c in range(xf.shape[0])]
    print("\ncommon-mode share of variance per channel:", np.round(share, 3),
          "(near 1/N_channels is normal; much higher = shared interference)")


def batch(folder, t0, t1, band=(50e3, 400e3), win_s=0.05, n_win=200, min_lag_us=3.0):
    """Leak-on TDOA check for every .shse file in a folder -> prints a table and writes batch_tdoa.csv."""
    import glob
    import os
    files = sorted(glob.glob(os.path.join(glob.escape(folder), "*.shse")))
    rows = []
    print(f"window {t0}-{t1} s, band {band[0]/1e3:.0f}-{band[1]/1e3:.0f} kHz, |lag|<{min_lag_us} us masked")
    print(f"{'file':<24}" + "".join(f"{p:>22}" for p in ("ch1-ch2 med/IQR [us]", "ch1-ch3", "ch2-ch3"))
          + f"{'coh 1-2/1-3/2-3':>20}")
    for f in files:
        x, fs, _ = read_shse(f, t0, t1)
        if x.shape[1] < int(win_s * fs) * 2:
            print(f"{os.path.basename(f):<24}  skipped: no data in {t0}-{t1} s")
            continue
        xf = bandpass(x, fs, band)
        L = int(win_s * fs)
        starts = np.linspace(0, xf.shape[1] - L, min(n_win, xf.shape[1] // L)).astype(int)
        name = os.path.basename(f)
        row, cohs, cells = [name], [], []
        for i, j in itertools.combinations(range(x.shape[0]), 2):
            d = np.array([gcc_phat_tdoa(xf[i, s:s + L], xf[j, s:s + L], fs, band=band,
                                        min_lag_s=min_lag_us * 1e-6) for s in starts]) * 1e6
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            fr, C = signal.coherence(xf[i], xf[j], fs=fs, nperseg=8192)
            coh = C[(fr >= band[0]) & (fr <= band[1])].mean()
            row += [med, q3 - q1, coh]
            cells.append(f"{med:9.1f} / {q3 - q1:6.1f}")
            cohs.append(f"{coh:.2f}")
        rows.append(row)
        print(f"{name:<24}" + "".join(f"{c:>22}" for c in cells) + f"{'/'.join(cohs):>20}", flush=True)
    out = os.path.join(os.getcwd(), "batch_tdoa.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("file,ch12_med_us,ch12_iqr_us,ch12_coh,ch13_med_us,ch13_iqr_us,ch13_coh,"
                 "ch23_med_us,ch23_iqr_us,ch23_coh\n")
        for r in rows:
            fh.write(r[0] + "," + ",".join(f"{v:.3f}" for v in r[1:]) + "\n")
    print("saved: batch_tdoa.csv in the current folder")


def _avg_spectra(path, t0, t1, nperseg, pairs, nch, fs):
    """Auto- and cross-spectra averaged over [t0, t1), read one second at a time."""
    S, n_used, f = {}, 0, None
    for t in range(t0, t1):
        x, _, _ = read_shse(path, t, t + 1)
        if x.shape[1] < nperseg:
            break
        for i in range(nch):
            f, S[(i, i)] = _acc(S.get((i, i)), signal.csd(x[i], x[i], fs=fs, nperseg=nperseg))
        for i, j in pairs:
            # scipy csd(a, b) = <conj(A) B>; we want <X_i conj(X_j)> -> csd(x_j, x_i)
            f, S[(i, j)] = _acc(S.get((i, j)), signal.csd(x[j], x[i], fs=fs, nperseg=nperseg))
        n_used += 1
    for k in S:
        S[k] /= max(n_used, 1)
    return f, S, n_used


def _coh(S, i, j):
    return np.abs(S[(i, j)]) ** 2 / (S[(i, i)].real * S[(j, j)].real + 1e-30)


def xspec(path, t0=150, t1=350, nperseg=4096, band=(50e3, 400e3), baseline=(20, 110),
          subbands=((50e3, 100e3), (100e3, 150e3), (150e3, 200e3), (200e3, 300e3), (300e3, 400e3))):
    """Cross-spectra averaged over a long leak-on period (read 1 s at a time).

    For each pair (i, j): R(f) = <X_i X_j*>, so a peak at +tau means ch_i arrives tau LATER than ch_j.
    Frequencies that are already coherent in the no-leak baseline window are electrical
    interference and are masked before any delay estimate.
    Reports the GCC peak under PHAT, SCOT and ML (Hannan-Thomson) weighting, the mean coherence,
    and the group delay -dphi/domega fitted inside each sub-band (= dispersion check).
    Saves <file>_xspec.npz and a .png.
    """
    h = read_header(path)
    fs, nch = h["fs"], h["n_channels"]
    t1 = min(t1, h["seconds"])
    pairs = list(itertools.combinations(range(nch), 2))
    f, S, n_used = _avg_spectra(path, t0, t1, nperseg, pairs, nch, fs)
    n_avg = n_used * (2 * fs // nperseg - 1)              # Welch segments (50 % overlap)
    g2_floor = max(10.0 / n_avg, 0.002)                   # coherence below this is estimation noise
    bad = np.zeros_like(f, bool)
    if baseline:
        _, B, nb = _avg_spectra(path, baseline[0], baseline[1], nperseg, pairs, nch, fs)
        if nb:
            # mask a bin only where the interference cross-spectrum (from the no-leak window) is a
            # non-negligible part of the leak-window cross-spectrum, i.e. where it can bias the delay
            for i, j in pairs:
                bad |= np.abs(B[(i, j)]) > 0.1 * np.abs(S[(i, j)])
            bad = np.convolve(bad, np.ones(3), "same") > 0  # widen by one bin each side
    inb = (f >= band[0]) & (f <= band[1])
    use = inb & ~bad
    lags = (np.arange(nperseg) - nperseg // 2) / fs * 1e6
    print(f"{path.replace(chr(92), '/').rsplit('/', 1)[-1]}  |  averaged {n_used} s ({t0}-{t0 + n_used} s)"
          f"  |  {int((bad & inb).sum())} interference bins masked (baseline {baseline})")
    print(f"  {'pair':<8}{'coh':>6}{'PHAT':>9}{'SCOT':>9}{'ML':>9}   group delay per sub-band [us] "
          + " ".join(f"{lo/1e3:.0f}-{hi/1e3:.0f}k" for lo, hi in subbands))
    out = {"f": f, "lags_us": lags, "masked": bad}
    for i, j in pairs:
        R, Sii, Sjj = S[(i, j)], S[(i, i)].real, S[(j, j)].real
        g2 = np.clip(_coh(S, i, j), 0, 0.999)
        peaks = []
        for name, W in (("PHAT", 1 / (np.abs(R) + 1e-30)),
                        ("SCOT", 1 / np.sqrt(Sii * Sjj + 1e-30)),
                        ("ML", g2 / (1 - g2) / (np.abs(R) + 1e-30))):
            G = np.where(use, R * W, 0)
            cc = np.fft.fftshift(np.fft.irfft(G, nperseg))
            out[f"cc_{name}_{i+1}{j+1}"] = cc
            peaks.append(lags[np.argmax(np.abs(cc))])
        ph = np.unwrap(np.angle(R))
        # group delay from adjacent-bin phase differences (no unwrapping across masked gaps):
        # tau = -angle(sum w_k R[k+1] R[k]^*) / (2 pi df), using only pairs of neighbouring valid bins
        ok = use & (g2 > g2_floor)
        pair_ok = ok[:-1] & ok[1:]
        D = R[1:] * np.conj(R[:-1]) / (np.abs(R[1:]) * np.abs(R[:-1]) + 1e-30)
        wgt = np.sqrt(g2[1:] * g2[:-1])
        df = f[1] - f[0]
        gd = []
        for lo, hi in subbands:
            m = pair_ok & (f[:-1] >= lo) & (f[1:] < hi)
            if m.sum() < 5:
                gd.append("   n/a")
                continue
            gd.append(f"{-np.angle(np.sum(wgt[m] * D[m])) / (2 * np.pi * df) * 1e6:6.0f}")
        out[f"coh_{i+1}{j+1}"], out[f"phase_{i+1}{j+1}"] = g2, ph
        print(f"  ch{i+1}-ch{j+1}{g2[use].mean():6.3f}" + "".join(f"{p:9.1f}" for p in peaks)
              + "   " + "  ".join(gd))
        out[f"top_{i+1}{j+1}"] = _top_peaks(lags, np.abs(out[f"cc_SCOT_{i+1}{j+1}"]))
    print("  top-3 SCOT peaks, lag [us] (relative height) -> two strong peaks = two arrival paths:")
    for i, j in pairs:
        print(f"    ch{i+1}-ch{j+1}: " + "   ".join(f"{l:7.0f} ({h:.2f})" for l, h in out[f"top_{i+1}{j+1}"]))
    base = path.rsplit(".", 1)[0] + "_xspec"
    np.savez_compressed(base + ".npz", **out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 9))
        for i, j in pairs:
            k = f"{i+1}{j+1}"
            ax[0].plot(f / 1e3, out[f"coh_{k}"], lw=.7, label=f"ch{i+1}-ch{j+1}")
            ax[1].plot(lags, np.abs(out[f"cc_ML_{k}"]) / np.abs(out[f"cc_ML_{k}"]).max(), lw=.7)
        ax[0].set_xlim(0, 500); ax[0].set_ylabel("coherence"); ax[0].legend()
        ax[1].set_xlim(-1500, 1500); ax[1].set_xlabel("lag [us] (+ = first channel later)")
        ax[1].set_ylabel("ML-GCC (norm.)")
        m = use & (out["coh_23"] > g2_floor) if nch >= 3 else use
        ax[2].plot(f[m] / 1e3, np.unwrap(np.angle(S[(1, 2)][m])) if nch >= 3 else [], ".", ms=2)
        ax[2].set_xlabel("kHz"); ax[2].set_ylabel("phase ch2-ch3 [rad]")
        plt.tight_layout(); plt.savefig(base + ".png", dpi=120)
    except ImportError:
        pass
    return out


def aic_onset(x):
    """AIC onset picker (same criterion as aic_pick.m), vectorised with cumulative sums."""
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    k = np.arange(2, n - 2)
    c1, c2 = np.cumsum(x), np.cumsum(x ** 2)
    m_pre, v_pre = c1[k - 1] / k, c2[k - 1] / k
    v_pre = v_pre - m_pre ** 2
    n_post = n - k
    s1, s2 = c1[-1] - c1[k - 1], c2[-1] - c2[k - 1]
    v_post = s2 / n_post - (s1 / n_post) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        aic = k * np.log(v_pre) + (n - k - 1) * np.log(v_post)
    aic[~np.isfinite(aic)] = np.inf
    return int(k[np.argmin(aic)])


def events(path, t0, t1, band=(50e3, 400e3), ref=1, k_sigma=5.0, pre_us=300, post_us=1500,
           merge_us=200, min_len_us=50, snr_min=3.0, baseline=(20, 110)):
    """Burst (AE event) TDOAs with ONE fixed setting and no window around any expected answer.

    Trigger on channel `ref` (0-based; default 1 = data Ch2, the loudest). For each burst, AIC-pick the
    onset on every channel in [onset_ref - pre_us, onset_ref + post_us]; a channel's pick is kept only if
    its envelope peak in that window exceeds snr_min x its median envelope in the no-leak baseline.
    Returns per-event TDOAs relative to data Ch3 (index 2), positive = later than Ch3.
    """
    h = read_header(path)
    fs, nch = h["fs"], h["n_channels"]
    b, a = signal.butter(4, band, btype="band", fs=fs)
    env_len = 50
    # baseline envelope level per channel (no leak)
    base = []
    for t in range(baseline[0], min(baseline[1], baseline[0] + 10)):
        xb, _, _ = read_shse(path, t, t + 1)
        eb = np.abs(signal.hilbert(signal.filtfilt(b, a, xb, axis=1), axis=1))
        base.append(np.median(eb, axis=1))
    base = np.median(np.array(base), axis=0)
    pre, post = int(pre_us * fs / 1e6), int(post_us * fs / 1e6)
    rows = []
    for t in range(t0, min(t1, h["seconds"])):
        x, _, _ = read_shse(path, t, t + 1)
        xf = signal.filtfilt(b, a, x, axis=1)
        env = np.abs(signal.hilbert(xf, axis=1))
        er = np.convolve(env[ref], np.ones(env_len) / env_len, "same")
        th = er.mean() + k_sigma * er.std()
        above = np.diff(np.concatenate(([0], (er > th).astype(int), [0])))
        st, en = np.where(above == 1)[0], np.where(above == -1)[0] - 1
        if st.size == 0:
            continue
        ms, me = [st[0]], [en[0]]
        for s_, e_ in zip(st[1:], en[1:]):
            if s_ - me[-1] < merge_us * fs / 1e6:
                me[-1] = e_
            else:
                ms.append(s_); me.append(e_)
        for s_, e_ in zip(ms, me):
            if e_ - s_ < min_len_us * fs / 1e6:
                continue
            w0 = max(0, s_ - 500)
            o_ref = w0 + aic_onset(xf[ref, w0:min(xf.shape[1], s_ + 500)])
            lo, hi = o_ref - pre, o_ref + post
            if lo < 0 or hi > xf.shape[1]:
                continue
            on = np.full(nch, np.nan)
            for c in range(nch):
                ec = np.convolve(env[c, lo:hi], np.ones(env_len) / env_len, "same")
                if ec.max() < snr_min * base[c]:
                    continue
                # AIC on a window that ends just after this channel's own envelope peak
                pk = lo + int(np.argmax(ec))
                a0, a1 = max(lo, pk - int(600 * fs / 1e6)), min(hi, pk + int(20 * fs / 1e6))
                if a1 - a0 < 50:
                    continue
                on[c] = a0 + aic_onset(xf[c, a0:a1])
            rows.append([t + o_ref / fs] + list((on - on[2]) / fs * 1e6))
    return np.array(rows).reshape(-1, 1 + nch)


def _hist_text(v, lo=-400, hi=1600, step=100):
    v = v[np.isfinite(v)]
    edges = np.arange(lo, hi + step, step)
    c, _ = np.histogram(v, edges)
    return " ".join(f"{int(e)}:{n}" for e, n in zip(edges[:-1], c) if n)


def events_report(paths, t0, t1, **kw):
    print(f"burst TDOAs, window {t0}-{t1} s, ONE fixed setting for all files, trigger = data Ch2")
    print(f"{'file':<24}{'events':>7}  {'Ch1-Ch3 med/IQR [us]':>22}  {'Ch2-Ch3 med/IQR':>18}  "
          f"{'Ch1 kept':>8}")
    allrows = {}
    for p in paths:
        R = events(p, t0, t1, **kw)
        name = p.replace(chr(92), "/").rsplit("/", 1)[-1]
        allrows[name] = R
        if len(R) == 0:
            print(f"{name:<24}{0:>7}")
            continue
        d13, d23 = R[:, 1], R[:, 2]
        f13, f23 = d13[np.isfinite(d13)], d23[np.isfinite(d23)]
        q = lambda v: (f"{np.median(v):8.0f} / {np.subtract(*np.percentile(v, [75, 25])):6.0f}"
                       if v.size else "      n/a")
        print(f"{name:<24}{len(R):>7}  {q(f13):>22}  {q(f23):>18}  {f13.size:>8}")
        print(f"{'':<24}  Ch1-Ch3 histogram (100 us bins, start:count): {_hist_text(f13)}")
    return allrows


def _top_peaks(lags, a, k=3, max_lag=1500, min_sep=30):
    """k largest local maxima of |cc| within |lag| <= max_lag, at least min_sep us apart."""
    m = np.abs(lags) <= max_lag
    L, A = lags[m], a[m]
    idx = signal.find_peaks(A, distance=max(1, int(min_sep / (L[1] - L[0]))))[0]
    idx = idx[np.argsort(A[idx])[::-1][:k]]
    top = A[idx[0]] if len(idx) else 1.0
    return [(L[n], A[n] / top) for n in idx]


def _acc(prev, fr):
    f, P = fr
    return f, (P if prev is None else prev + P)


if __name__ == "__main__":
    import sys
    try:  # Windows consoles (cp1252) cannot print Korean folder names
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    print(VERSION)
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".shse file, or a folder when --batch is used")
    ap.add_argument("--seconds", nargs=2, type=int, default=None)
    ap.add_argument("--timeline", action="store_true", help="per-second RMS over the whole file")
    ap.add_argument("--batch", action="store_true", help="leak-on TDOA table for every file in a folder")
    ap.add_argument("--xspec", action="store_true",
                    help="long-average cross-spectra (file or folder); default window 150-350 s")
    ap.add_argument("--baseline", nargs=2, type=int, default=[20, 110],
                    help="no-leak window used to find interference frequencies (default 20 110)")
    ap.add_argument("--events", action="store_true",
                    help="burst TDOAs with one fixed setting (file or folder); default window 150-180 s")
    args = ap.parse_args()
    if args.events:
        import glob
        import os
        files = (sorted(glob.glob(os.path.join(glob.escape(args.path), "*.shse")))
                 if os.path.isdir(args.path) else [args.path])
        events_report(files, *(args.seconds or [150, 180]), baseline=tuple(args.baseline))
        sys.exit()
    if args.xspec:
        import glob
        import os
        files = (sorted(glob.glob(os.path.join(glob.escape(args.path), "*.shse")))
                 if os.path.isdir(args.path) else [args.path])
        for fpath in files:
            xspec(fpath, *(args.seconds or [150, 350]), baseline=tuple(args.baseline))
        sys.exit()
    if args.batch:
        batch(args.path, *(args.seconds or [200, 220]))
        sys.exit()
    args.seconds = args.seconds or [0, 20]
    if args.timeline:
        print(read_header(args.path))
        timeline(args.path)
    else:
        x, fs, h = read_shse(args.path, *args.seconds)
        print(h)
        print("loaded:", x.shape, "| per-channel std:", np.round(x.std(axis=1), 6))
        feasibility_check(x, fs)