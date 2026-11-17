#!/usr/bin/env python3
"""What an FRB looks like with and without the site RFI, centred on the pulse (in S/N per pixel).

Row 1: dispersed waterfall around the sweep (the dotted curve is the dispersion delay).
Row 2: dedispersed at the FRB's DM, ±250 ms.
Row 3: dedispersed profile (sum of channels), in S/N.
Each channel is normalized with its median and robust σ (MAD) in the window; saturated channels (> 50 % at 255) are set to 0.
When averaging n pixels the result is multiplied by √n: the noise of each pixel stays at σ = 1.
Usage: python 06_fig_frb_with_rfi.py ORIGINAL.fil WITH_RFI.fil DM T_TOP_S output.png [title]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigproc import read_header  # noqa: E402

orig, with_rfi, dm, ttop, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
title = sys.argv[6] if len(sys.argv) > 6 else Path(orig).stem
K = 4.148808e3                                            # s MHz² (delay in s)


def load(p):
    head, h = read_header(p)
    nch = h["nchans"]; n = (Path(p).stat().st_size - len(head)) // nch
    return np.memmap(p, np.uint8, "r", offset=len(head), shape=(n, nch)), h


def norm(x):
    x = x.astype(np.float32)
    med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0)
    z = (x - med) / np.maximum(mad, 1e-3)
    z[:, ((x == 255).mean(0) > 0.5) | (mad < 0.5)] = 0
    return z


def rebin(z, bt, bf):
    nt, nf = z.shape[0] // bt * bt, z.shape[1] // bf * bf
    w = z[:nt, :nf].reshape(nt // bt, bt, nf // bf, bf).mean((1, 3)) * np.sqrt(bt * bf)
    return w - np.median(w, 0)            # removes the small bias of the per-channel median, which binning amplifies


fig, ax = plt.subplots(3, 2, figsize=(16, 14), gridspec_kw=dict(height_ratios=[1.1, 1, 0.6]))
for j, (p, name) in enumerate(((orig, "Without RFI"), (with_rfi, "With site RFI"))):
    d, h = load(p); n, nch = d.shape
    f = h["fch1"] + h["foff"] * np.arange(nch)            # descending
    delay = K * dm * (f ** -2 - f.max() ** -2)             # s
    # row 1: dispersed sweep
    t0, t1 = max(0, ttop - 1.0), min(n / 1000, ttop + delay.max() + 1.0)
    z = norm(np.asarray(d[int(t0 * 1000):int(t1 * 1000)]))
    bt, bf = 20, 256
    w = rebin(z, bt, bf)
    ax[0, j].imshow(w.T, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
                    extent=[t0, t0 + w.shape[0] * bt / 1000, f.min(), f.max()])
    ax[0, j].plot(ttop + delay, f, "w:", lw=1.2, alpha=0.8)
    ax[0, j].set(title=f"{name}: dispersed sweep (dotted: DM {dm:.0f} curve)",
                 xlabel="Time (s)", ylabel="Frequency (MHz)")
    # row 2: dedispersed ±250 ms
    r = np.rint(delay * 1000).astype(int); i0 = int(ttop * 1000) - 250
    win = np.stack([np.asarray(d[i0 + rr:i0 + rr + 500, c]) for c, rr in enumerate(r)], 1)
    z = norm(win)
    w = rebin(z, 4, 448)
    ax[1, j].imshow(w.T, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
                    extent=[-250, 250, f.min(), f.max()])
    ax[1, j].set(title=f"{name}: dedispersed at DM {dm:.0f}",
                 xlabel="Time relative to arrival (ms)", ylabel="Frequency (MHz)")
    # row 3: profile
    prof = z.sum(1) / np.sqrt(max(1, (np.abs(z).sum(0) > 0).sum()))
    prof = np.convolve(prof, np.ones(4) / 2, "same")       # 4 ms boxcar, σ = 1
    off = np.r_[prof[:150], prof[350:]]
    snr = (prof - np.median(off)) / (1.4826 * np.median(np.abs(off - np.median(off))))
    ax[2, j].plot(np.arange(-250, 250), snr, lw=1.2, color=plt.cm.viridis(0.3))
    ax[2, j].axhline(0, color="0.7", lw=0.8)
    ax[2, j].set(title=f"{name}: dedispersed profile (4 ms boxcar) · peak {snr[230:280].max():.1f} σ",
                 xlabel="Time relative to arrival (ms)", ylabel="S/N (σ)")
    print(name, "peak", round(float(snr[230:280].max()), 1))
lim = max(a.get_ylim()[1] for a in ax[2]); [a.set_ylim(min(a.get_ylim()[0], -4), lim) for a in ax[2]]
fig.suptitle(title + "\nPixels in S/N (noise σ = 1): row 1 of 20 ms × 256 channels, row 2 of 4 ms × 448 channels", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=100)
print(out)
