#!/usr/bin/env python3
"""Site RFI model from the dynamic spectrum (01_dynamic_spectrum.py).

For each input a (0–7):
  floor_a(f) = B_a(c) · S_a(k)  B: level of coarse channel c without RFI (running median ±25 coarse channels, iterated 4 times
                                   without the channels > 1.1 B; smoothed ±5)
                                S: PFB scalloping within the coarse channel (median over coarse channels)
  r_a(t,f) = P_a(t,f) / floor_a(f) − 1     RFI power in units of the system power
Beam: r(t,f) = mean of r_a over the inputs. For a beam outside the main lobe (sidelobes with random phases) the
RFI/system ratio is on average the same as in one antenna: attenuation 0 dB.
  r_p(f)   = temporal median of r(t,f)                           → persistent RFI (lines)
  σ_r(f)   = temporal MAD of r(t,f)                              → measurement noise
  e(t,f)   = r − r_p if (r − r_p) > 5 σ_r, plus the connected neighbours > 3 σ_r; otherwise 0   → fast RFI
Output: site_rfi_model.npz (frequencies, r_p, σ_r, e in float16, floors), rfi_summary.json, figure.
Usage: python 02_rfi_model.py [--inputs 0,1,...]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PSYS_COUNTS = 4 * np.sqrt(9.375)       # system power in .fil counts (σ = 4 counts = P_sys/√(Δν τ))

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="0,1,2,3,4,5,6,7")
ap.add_argument("--threshold", type=float, default=5.0)
ap.add_argument("--neighbour-threshold", type=float, default=3.0)
args = ap.parse_args()
inputs = [int(x) for x in args.inputs.split(",")]

D = np.load(HERE / "dynamic_spectrum.npy", mmap_mode="r")
f = np.load(HERE / "frequencies_mhz.npy")
NT, NF = D.shape[1], D.shape[2]


def running(x, w, fn):
    """Window filter ±w with reflected edges."""
    xp = np.pad(x, w, mode="reflect")
    v = np.lib.stride_tricks.sliding_window_view(xp, 2 * w + 1)
    return fn(v, axis=-1)


floors, ra_med, r = {}, {}, np.zeros((NT, NF), np.float32)
for a in inputs:
    P = np.asarray(D[a])
    m = np.median(P, 0).reshape(672, 32)
    S = np.median(m / np.median(m, 1, keepdims=True), 0)
    L = np.median(m / S, 1)
    B = running(L, 25, np.median)
    for _ in range(4):                       # iterative floor: running median only over coarse channels without RFI
        B = running(np.where(L < 1.1 * B, L, np.nan), 25, np.nanmedian)
        B = np.where(np.isfinite(B), B, np.nanmin(B))
    B = running(B, 5, np.mean)
    floor = (B[:, None] * S[None, :]).ravel().astype(np.float32)
    ra = P / floor - 1
    floors[a] = floor; ra_med[a] = np.median(ra, 0)
    r += ra / len(inputs)
    print(f"input {a}: floor done, median r_p {np.median(ra_med[a]):+.3f}", flush=True)

r_p = np.median(r, 0)
sig = 1.4826 * np.median(np.abs(r - r_p), 0)
ex = r - r_p
strong = ex > args.threshold * sig
weak = ex > args.neighbour_threshold * sig
lab, nlab = ndimage.label(weak)
keep = np.zeros(nlab + 1, bool); keep[np.unique(lab[strong])] = True; keep[0] = False
mask = keep[lab]
e = np.where(mask, ex, 0).astype(np.float16)

# --- statistics
lab2, nev = ndimage.label(mask)
obj = ndimage.find_objects(lab2)
dur = np.array([s[0].stop - s[0].start for s in obj]); wid = np.array([s[1].stop - s[1].start for s in obj])
peak = ndimage.maximum(ex, lab2, np.arange(1, nev + 1))
dnu = 0.009375
band_frac_ms = mask.mean(1)
res = dict(
    inputs=inputs, duration_s=NT / 1000,
    expected_sigma_over_mean=float(1 / np.sqrt(9.375)), median_sigma_r=float(np.median(sig)),
    channels_rp_above_0p1=float((r_p > 0.1).mean()), channels_rp_above_1=float((r_p > 1).mean()),
    channels_rp_above_10dB=float((r_p > 9).mean()),
    fast_pixels=float(mask.mean()), fast_events=int(nev), events_per_s=float(nev / (NT / 1000)),
    duration_ms_p50_p90_p99=[float(np.percentile(dur, q)) for q in (50, 90, 99)],
    width_mhz_p50_p90_p99=[float(np.percentile(wid, q) * dnu) for q in (50, 90, 99)],
    peak_r_p50_p90_p99=[float(np.percentile(peak, q)) for q in (50, 90, 99)],
    ms_with_over_10pct_band=int((band_frac_ms > 0.1).sum()),
    psys_counts=float(PSYS_COUNTS),
    sat_255_if_lines_visible=float(((32 + PSYS_COUNTS * (r_p + e.astype(np.float32))) > 255).mean()),
)
# coincidence of the fast RFI between inputs (external, or local to one antenna?)
co = {}
for a in inputs:
    ea = (np.asarray(D[a]) / floors[a] - 1) - ra_med[a]
    sa = 1.4826 * np.median(np.abs(ea), 0)
    ma = ea > args.threshold * sa
    others = np.zeros_like(ma, dtype=np.int8)
    for b in inputs:
        if b == a: continue
        eb = (np.asarray(D[b]) / floors[b] - 1) - ra_med[b]
        others += (eb > args.threshold * 1.4826 * np.median(np.abs(eb), 0)).astype(np.int8)
    co[a] = dict(frac_pixels=float(ma.mean()), frac_also_in_another=float((others[ma] > 0).mean()) if ma.any() else None,
                 frac_in_4_or_more=float((others[ma] >= 3).mean()) if ma.any() else None)
    print("coincidence", a, co[a], flush=True)
res["input_coincidence"] = co
json.dump(res, open(HERE / "rfi_summary.json", "w"), indent=1)
np.savez(HERE / "site_rfi_model.npz", frequencies_mhz=f, r_p=r_p.astype(np.float32), sigma_r=sig.astype(np.float32),
         e=e, inputs=np.array(inputs), dt_ms=1.0, psys_counts=PSYS_COUNTS,
         r_p_per_input=np.array([ra_med[a] for a in inputs], np.float32))
print(json.dumps({k: v for k, v in res.items() if k != "input_coincidence"}, indent=1))

# --- figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
for i, a in enumerate(inputs):
    ax[0, 0].plot(f, 10 * np.log10(1 + np.clip(ra_med[a], -0.9, None)), lw=0.5, alpha=0.6, color=plt.cm.viridis(i / len(inputs)))
ax[0, 0].plot(f, 10 * np.log10(1 + np.clip(r_p, -0.9, None)), lw=0.8, color="k", label="mean (beam)")
ax[0, 0].set(xlabel="MHz", ylabel="10 log10(1 + r_p)  (dB above the floor)", title="Persistent RFI: median over 4.1 s"); ax[0, 0].legend()
ds = r[:, :].reshape(NT, 672, 32).mean(2)
im = ax[0, 1].imshow(10 * np.log10(1 + np.clip(ds, -0.5, None)).T, aspect="auto", origin="lower", cmap="viridis",
                     extent=[0, NT / 1000, f[0], f[-1]], vmin=-0.5, vmax=6)
fig.colorbar(im, ax=ax[0, 1], label="dB above the floor"); ax[0, 1].set(xlabel="time (s)", ylabel="MHz", title="Beam (mean of 8 inputs), coarse channels")
sel = (f > 405) & (f < 432)
im = ax[1, 0].imshow((ex[:1000, sel] / sig[sel]).T, aspect="auto", origin="lower", cmap="viridis",
                     extent=[0, 1, f[sel][0], f[sel][-1]], vmin=-3, vmax=10)
fig.colorbar(im, ax=ax[1, 0], label="(r − r_p)/σ_r"); ax[1, 0].set(xlabel="time (s)", ylabel="MHz", title="Fast RFI, 405–432 MHz, first second, 21 504 channels")
z = (ex / sig)[:, ::5].ravel()
h, b = np.histogram(z, bins=np.linspace(-6, 30, 145))
ax[1, 1].semilogy(b[:-1], h / h.sum(), drawstyle="steps-post", label="measured")
from scipy.stats import norm
ax[1, 1].semilogy(b[:-1], norm.pdf(b[:-1] + 0.125) * 0.25, "k--", label="gaussian")
ax[1, 1].axvline(args.threshold, color="0.5", ls=":"); ax[1, 1].set(xlabel="(r − r_p)/σ_r", ylabel="fraction", title="Tail of the fast RFI", ylim=(1e-9, 0.2)); ax[1, 1].legend()
fig.tight_layout(); fig.savefig(HERE / "site_rfi_model.png", dpi=110)
