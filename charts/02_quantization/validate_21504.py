#!/usr/bin/env python3
"""Visual validation of one group-3 file (21 504 channels) and of the S/N of set A.

Figure per file: waterfall of the whole band (averaged) with the theoretical dispersion
curve, full-resolution zoom at the bottom of the band, series dedispersed at the true DM
with the measured S/N (optimal boxcar) against the expected one from the manifest, and noise histogram.
It also measures the S/N of every A file in the set (it should give ~8).

Usage (simulator venv): python validate_21504.py SET_DIR [NAME] [--output plots_21504]
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "01_simulator"                   # simulator generator and physics
_spec = importlib.util.spec_from_file_location("gen07", SIM / "07_generate_comparison_set.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

WIDTHS = G.WIDTHS[G.WIDTHS <= 1024]


def read_fil(path):
    with open(path, "rb") as f:
        raw = f.read(4096)
    hlen = raw.index(b"HEADER_END") + len(b"HEADER_END")
    nchan = 21504
    d = np.memmap(path, dtype=np.uint8, mode="r", offset=hlen)
    return d.reshape(-1, nchan)          # (time, descending channel)


def freqs_fil(cfg):
    f = np.linspace(cfg["freq_min_mhz"], cfg["freq_max_mhz"], 21504)
    return f[::-1]                        # .fil order


def dedisperse(x, f, dm, idx, f_ref=501.6):
    """x: (time, channel) in sigma. Sum with integer shifts at the true DM.
    Returns the series and the offset: the arrival at f_ref (sample i) ends up at i + off."""
    sh = np.round((G.delay_s(dm, idx, f) - G.delay_s(dm, idx, f_ref)) / 1e-3).astype(int)
    off = int(sh.min())
    sh = sh - off
    n = x.shape[0] - sh.max()
    out = np.zeros(n, np.float64)
    for c in range(x.shape[1]):
        out += x[sh[c]:sh[c] + n, c]
    return out, off


def boxcar_snr(series, nchan, i_exp, window=300):
    c = np.concatenate([[0.0], np.cumsum(series)])
    best = (-np.inf, 1, 0)
    for w in WIDTHS:
        v = (c[w:] - c[:-w]) / np.sqrt(w * nchan)
        lo, hi = max(0, i_exp - window - w), min(len(v), i_exp + window)
        j = lo + int(np.argmax(v[lo:hi]))
        if v[j] > best[0]:
            best = (float(v[j]), int(w), j)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("name", nargs="?", default="A_001")
    ap.add_argument("--output", type=Path, default=HERE / "plots_21504")
    a = ap.parse_args()
    import json
    cfg = json.loads((a.set_dir / "config.json").read_text())
    f = freqs_fil(cfg)
    a.output.mkdir(exist_ok=True)

    # --- S/N of every A file ---
    man = {r["name"]: r for s in ("A_snr_fijo", "B_amplitud_fija", "negativos", "parcial")
           if (a.set_dir / s / "manifest.csv").exists()
           for r in csv.DictReader(open(a.set_dir / s / "manifest.csv"))}
    print(f"{'file':>8} {'DM':>7} {'expected S/N':>12} {'meas. S/N':>10} {'boxcar ms':>9} {'series noise':>12}")
    measured = []
    for name, r in sorted(man.items()):
        if r["set"] != "A":
            continue
        x = (read_fil(r["fil_path"]).astype(np.float32) - 32.0) / 4.0
        dm, t_top = float(r["dm"]), float(r["t_top_s"])
        s, off = dedisperse(x, f, dm, G.DM_INDEX)
        i_exp = int(round(t_top / 1e-3)) + off
        outside = np.r_[s[:max(1, i_exp - 2000)], s[i_exp + 3000:]]
        snr, w, _ = boxcar_snr(s, x.shape[1], i_exp)
        measured.append((snr, float(r["snr_esperado"])))
        print(f"{name:>8} {dm:7.0f} {float(r['snr_esperado']):12.2f} {snr:10.2f} {w:9d} "
              f"{outside.std() / np.sqrt(x.shape[1]):12.3f}")
    d = np.array([m - e for m, e in measured])
    print(f"Measured - expected S/N (A): {d.mean():+.2f} ± {d.std():.2f}  (n={len(d)}; the maximum over noise biases it by ~+0.5)")

    # --- figure of one file ---
    r = man[a.name]
    dm, t_top, idx = float(r["dm"]), float(r["t_top_s"]), float(r["dm_index"])
    raw = read_fil(r["fil_path"])
    x = (raw.astype(np.float32) - 32.0) / 4.0
    nt = x.shape[0]
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 0.8], hspace=0.35, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    fb, tb = 128, 8                                          # 168 subbands of 1.2 MHz, 8 ms
    nt8 = nt // tb * tb
    wf = x[:nt8].reshape(nt8 // tb, tb, -1, fb).mean(axis=(1, 3))     # (t, descending band)
    fc = f.reshape(-1, fb).mean(axis=1)
    sig = 1 / np.sqrt(tb * fb)
    ax.imshow(wf.T / sig, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
              extent=[0, nt8 * 1e-3, fc[-1], fc[0]])
    tt = t_top + G.delay_s(dm, idx, f) - G.delay_s(dm, idx, 501.6)
    ax.plot(tt, f, "w--", lw=0.8, alpha=0.7, label="theoretical delay (manifest DM)")
    ax.set_xlabel("time from the start of the file (s)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(f"{a.name}: DM {dm:.1f}, width {float(r['width_s']) * 1e3:.2f} ms, "
                 f"scattering(600 MHz) {float(r['scat_600_s']) * 1e3:.2f} ms, expected S/N {float(r.get('snr_esperado') or 0):.1f}, "
                 f"length {nt / 1000:.0f} s\nwaterfall averaged to 168 subbands × 8 ms (in σ of the average)")
    ax.legend(loc="lower left", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    sel = slice(len(f) - 256, len(f))                        # 256 fine channels at the bottom (~300 MHz)
    t_lo = G.delay_s(dm, idx, f[sel][-1]) - G.delay_s(dm, idx, 501.6) + t_top
    t_hi = G.delay_s(dm, idx, f[sel][0]) - G.delay_s(dm, idx, 501.6) + t_top
    i0, i1 = int((t_hi - 0.15) / 1e-3), int((t_lo + 0.15) / 1e-3)
    z = x[i0:i1, sel]
    z4 = z[: (z.shape[0] // 4) * 4].reshape(-1, 4, z.shape[1]).mean(1)
    ax.imshow(z4.T / 0.5, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
              extent=[i0 * 1e-3, (i0 + z4.shape[0] * 4) * 1e-3, f[sel][-1], f[sel][0]])
    ax.plot(tt[sel], f[sel], "w--", lw=0.8, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title("zoom: 256 fine channels (9.39 kHz) at the bottom of the band, 4 ms")

    ax = fig.add_subplot(gs[1, 1])
    s, off = dedisperse(x, f, dm, idx)
    i_exp = int(round(t_top / 1e-3)) + off
    snr, w, j = boxcar_snr(s, x.shape[1], i_exp)
    k = np.arange(max(0, i_exp - 400), min(len(s), i_exp + 400))
    ax.plot((k - i_exp) * 1.0, s[k] / np.sqrt(x.shape[1]), lw=0.6, color="0.4", label="1 ms")
    c = np.concatenate([[0.0], np.cumsum(s)])
    bx = (c[w:] - c[:-w]) / np.sqrt(w * x.shape[1])
    kb = k[k < len(bx)]
    ax.plot((kb - i_exp) + w / 2, bx[kb], color=plt.cm.viridis(0.35), label=f"boxcar {w} ms")
    ax.axvline(0, color=plt.cm.viridis(0.8), ls=":", label="manifest t_top")
    ax.set_xlabel("ms relative to the expected arrival at 501.6 MHz")
    ax.set_ylabel("S/N")
    ax.set_title(f"series dedispersed at the true DM: measured S/N {snr:.2f} (expected {float(r.get('snr_esperado') or 0):.2f})")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    noise = raw[:2000].ravel()
    h = np.bincount(noise, minlength=256)[:80] / noise.size
    ax.bar(np.arange(80), h, width=1.0, color=plt.cm.viridis(0.35), label="first 2 s (no burst)")
    lv = np.arange(80)
    from scipy.stats import norm
    ax.plot(lv, norm.cdf(lv + 1 - 32.5, scale=4) - norm.cdf(lv - 32.5, scale=4), "k.", ms=4,
            label="expected: floor 32, σ = 4 levels")
    ax.set_xlabel("level (8 bits)")
    ax.set_title(f"noise: mean {noise.mean():.3f}, σ {noise.std():.3f} levels")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    if measured:
        m = np.array(measured)
        ax.plot(m[:, 1], m[:, 0], "o", color=plt.cm.viridis(0.35))
        ax.axhline(8, color="k", ls=":")
        ax.set_xlim(6, 10)
        ax.set_ylim(5, 11)
        ax.set_xlabel("expected S/N (manifest)")
        ax.set_ylabel("S/N measured in the .fil")
        ax.set_title(f"set A ({len(m)} files): measured − expected = {d.mean():+.2f} ± {d.std():.2f}")
    out = a.output / f"validation_{a.name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
