#!/usr/bin/env python3
"""Effect of quantization on the S/N, in the group-3 test set (21 504 channels).

For each file the float data are rebuilt EXACTLY (same model and same seed as
01_simulator/12_generate_comparison_set_21504.py; quantizing them is checked to give the same bytes
as the .fil) and the SAME noise + signal realization is compared with different quantizations:

  float          unquantized (reference)
  8 bits         the .fil (floor 32, 1 sigma = 4 levels)
  4b 2lev fixed  from the 8 bits: q = round((x - 32) / 2), clipped to [-7, 7]  (2 levels/sigma, ±3.5 sigma)
  4b 1lev fixed  from the 8 bits: 1 level/sigma, clipped to ±7 sigma
  4b 3lev fixed  from the 8 bits: 3 levels/sigma, clipped to ±2.3 sigma
  4b pirate      pirate's network format: scale/offset per (channel, 256 samples), 2 levels/sigma

S/N: dedispersed at the true DM (integer shifts), best boxcar, noise measured outside the pulse
in the series of each variant.

Usage (simulator venv): python quantization_21504.py SET_DIR [--sets A,B] [--output plots_21504]
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "01_simulator"                   # simulator generator and physics


def load(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, (SIM if filename[:2].isdigit() else HERE) / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GEN = load("gen21504", "12_generate_comparison_set_21504.py")
G = GEN.G
VAL = load("val21504", "validate_21504.py")

WIDTHS = G.WIDTHS[G.WIDTHS <= 1024]


def rebuild_float(r):
    """Data in sigma, (time, ASCENDING channel), equal to those of write() before quantizing."""
    p = {k: float(r[k]) for k in ("dm", "width_s", "scat_ref_s", "sp_idx", "sp_run")}
    idx, t_top, nsamp, scale = float(r["dm_index"]), float(r["t_top_s"]), int(r["nsamp"]), float(r["escala_jy"])
    channels, _ = GEN.base(p, idx, t_top, nsamp, dm=p["dm"])
    rng = np.random.default_rng(G.seed_for(GEN.SEED, r["name"]))
    x = np.empty((nsamp, GEN.FREQS.size), np.float32)
    i0s = np.array([i0 for i0, _ in channels])
    i1s = np.array([i0 + len(v) for i0, v in channels])
    for a in range(0, nsamp, GEN.BLOCK):
        b = min(a + GEN.BLOCK, nsamp)
        blk = rng.standard_normal((b - a, GEN.FREQS.size), dtype=np.float32)
        for c in np.flatnonzero((i1s > a) & (i0s < b)):
            i0, v = channels[c]
            lo, hi = max(a, i0), min(b, i0 + len(v))
            blk[lo - a:hi - a, c] += (scale / GEN.SIGMA) * v[lo - i0:hi - i0]
        x[a:b] = blk
    return x


def variants(xf, u8):
    """All in sigma, (time, descending channel = .fil order)."""
    xf = xf[:, ::-1]
    d8 = (u8.astype(np.float32) - 32.0)                    # in levels (1 sigma = 4)
    out = {"float": xf, "8 bits": d8 / 4.0}
    for lev, name in ((2, "4b 2lev fixed"), (1, "4b 1lev fixed"), (3, "4b 3lev fixed")):
        q = np.clip(np.rint(d8 / 4.0 * lev), -7, 7)
        out[name] = q / lev
    # pirate: scale/offset per (channel, 256 samples), from the 8 bits
    n = (d8.shape[0] // 256) * 256
    blk = (d8[:n] / 4.0).reshape(-1, 256, d8.shape[1])
    mu = blk.mean(axis=1, keepdims=True)
    sd = blk.std(axis=1, keepdims=True)
    scale = np.where(sd > 0, sd / 2.0, 1.0)
    q = np.clip(np.rint((blk - mu) / scale), -7, 7)
    out["4b pirate"] = (q * scale + mu).reshape(n, -1)
    return out


def snr(series, i_exp, nchan):
    off = np.r_[series[:max(1, i_exp - 2000)], series[i_exp + 3000:]]
    s0 = off.std()
    series = series - np.median(off)
    c = np.concatenate([[0.0], np.cumsum(series)])
    best = -np.inf
    for w in WIDTHS:
        v = (c[w:] - c[:-w]) / (s0 * np.sqrt(w))
        lo, hi = max(0, i_exp - 300 - w), min(len(v), i_exp + 300)
        best = max(best, float(v[lo:hi].max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--output", type=Path, default=HERE / "plots_21504")
    a = ap.parse_args()
    cfg = json.loads((a.set_dir / "config.json").read_text())
    f = VAL.freqs_fil(cfg)
    folders = {"A": "A_snr_fijo", "B": "B_amplitud_fija"}
    rows, profiles = [], {}
    for s in a.sets.split(","):
        for r in csv.DictReader(open(a.set_dir / folders[s] / "manifest.csv")):
            u8 = np.asarray(VAL.read_fil(r["fil_path"]))
            xf = rebuild_float(r)
            q8 = np.clip(np.floor(xf * 4 + 32.5), 0, 255).astype(np.uint8)[:, ::-1]
            equal = float((q8 == u8).mean())
            vs = variants(xf, u8)
            del xf
            dm, idx = float(r["dm"]), float(r["dm_index"])
            row = {"name": r["name"], "set": s, "dm": dm, "snr_expected": float(r["snr_esperado"]),
                   "equal_bytes": equal}
            for name, x in vs.items():
                series, off = VAL.dedisperse(x, f, dm, idx)
                i_exp = int(round(float(r["t_top_s"]) / 1e-3)) + off
                row[name] = snr(series, i_exp, x.shape[1])
                if r["name"] in ("A_001", "B_001", "B_005"):
                    profiles.setdefault(r["name"], {})[name] = series[i_exp - 150:i_exp + 150] / series[:i_exp - 2000].std()
            rows.append(row)
            print(f"{r['name']:>6} DM {dm:6.0f}  expected {row['snr_expected']:6.1f}  equal bytes {equal:.6f}  " +
                  "  ".join(f"{k}: {row[k]:6.2f}" for k in vs), flush=True)

    names = list(vs)
    with open(a.output / "quantization_21504.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nS/N relative to the float (median [min, max]):")
    table = {}
    for s in a.sets.split(","):
        sub = [r for r in rows if r["set"] == s and r["float"] > 4]
        for name in names[1:]:
            rel = np.array([r[name] / r["float"] for r in sub])
            table[(s, name)] = rel
            print(f"  {s} {name:>13}: {np.median(rel):.4f}  [{rel.min():.4f}, {rel.max():.4f}]  (n={len(rel)})")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    for k, name in enumerate(names[1:]):
        for j, s in enumerate(a.sets.split(",")):
            rel = table[(s, name)]
            xs = k + (j - 0.5) * 0.3 + np.random.default_rng(k).uniform(-0.06, 0.06, rel.size)
            ax.plot(xs, 100 * (rel - 1), "o", ms=5, color=plt.cm.viridis(0.25 + 0.5 * j),
                    label=f"set {s}" if k == 0 else None, alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(names) - 1), names[1:])
    ax.set_ylabel("S/N relative to the unquantized data (%)")
    ax.set_title("same noise and signal realization; only the quantization changes\n(each point = one file with S/N > 4)")
    ax.legend()
    ax = axs[1]
    name0 = "B_005" if "B_005" in profiles else next(iter(profiles))
    t = np.arange(-150, 150)
    for k, name in enumerate(names):
        ax.plot(t, profiles[name0][name], lw=1.0 if name != "float" else 2.0, label=name,
                color=plt.cm.viridis(k / max(1, len(names) - 1)), alpha=0.9 if name != "float" else 0.5)
    ax.set_xlabel("ms relative to the expected arrival")
    ax.set_ylabel("S/N per 1 ms sample")
    ax.set_title(f"{name0}: dedispersed series with each quantization")
    ax.legend(fontsize=8)
    out = a.output / "quantization_21504.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
