#!/usr/bin/env python3
"""Quick diagnostics of the site baseband (260902T161623Z): power per input and coarse channel.
Reads the first 30 000 samples (0.1 s) of chunk 0 of each group of 84 channels.
Output: input_diagnostics.npz and input_diagnostics.png."""
import os
import sys
from pathlib import Path
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

if "CHARTS_BASEBAND_DIR" not in os.environ:
    sys.exit("ERROR: export CHARTS_BASEBAND_DIR=/folder/260902T161623Z_CHARTS_hdf5 (site baseband)")
BB = Path(os.environ["CHARTS_BASEBAND_DIR"])
LUT = np.array([v - 16 if v > 7 else v for v in range(16)], np.int8)   # nibble → signed integer

def decode(b):
    """uint8 (4 real bits high + 4 imaginary low, two's complement) → complex64."""
    return (LUT[b >> 4] + 1j * LUT[b & 15]).astype(np.complex64)

P = np.zeros((32, 672)); Z = np.zeros((32, 672)); K = np.zeros((32, 672)); SAT = np.zeros((32, 672))
for g in range(8):
    with h5py.File(BB / f"bb_f{84*g:04d}_{84*g+84:04d}_chunk00000.h5", "r") as f:
        b = f["baseband"][:, :, :30000]
    x = decode(b); p = np.abs(x) ** 2
    s = slice(84 * g, 84 * g + 84)
    P[:, s] = p.mean(-1); Z[:, s] = (b == 0).mean(-1)
    K[:, s] = (p ** 2).mean(-1) / np.maximum(p.mean(-1) ** 2, 1e-9)          # 2 = complex gaussian
    SAT[:, s] = ((np.abs(LUT[b >> 4]) >= 7) | (np.abs(LUT[b & 15]) >= 7)).mean(-1)
np.savez("input_diagnostics.npz", P=P, Z=Z, K=K, SAT=SAT)
f = 300.0 + 0.3 * np.arange(672)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
im = ax[0, 0].imshow(10 * np.log10(P + 1e-3), aspect="auto", cmap="viridis", extent=[f[0], f[-1], 31.5, -0.5])
fig.colorbar(im, ax=ax[0, 0], label="power (dB, level² units)"); ax[0, 0].set(xlabel="MHz", ylabel="input", title="Mean power per input and channel")
for i in range(9):
    ax[0, 1].plot(f, 10 * np.log10(P[i] + 1e-3), lw=0.8, label=f"input {i}")
ax[0, 1].set(xlabel="MHz", ylabel="dB", title="Spectrum of inputs 0–8"); ax[0, 1].legend(fontsize=8, ncol=3)
ax[1, 0].plot(np.median(K, 1), "o"); ax[1, 0].axhline(2, color="k", ls="--"); ax[1, 0].set(xlabel="input", ylabel="⟨p²⟩/⟨p⟩² (median over channels)", title="Power kurtosis (2 = gaussian noise)")
ax[1, 1].plot(np.median(Z, 1) * 100, "o", label="% bytes = 0"); ax[1, 1].plot(SAT.mean(1) * 100, "s", label="% saturated (|x| = 7, 8)")
ax[1, 1].set(xlabel="input", ylabel="%", title="4+4 quantization"); ax[1, 1].legend()
fig.tight_layout(); fig.savefig("input_diagnostics.png", dpi=110)
for i in range(32):
    print(i, f"P med {np.median(P[i]):.2f}  rms/comp {np.sqrt(np.median(P[i])/2):.2f}  K {np.median(K[i]):.2f}  zeros {np.median(Z[i])*100:.0f}%  sat {SAT[i].mean()*100:.3f}%")
