#!/usr/bin/env python3
"""Dynamic spectrum of the site baseband at the resolution of the group-3 simulations.

Input: $CHARTS_BASEBAND_DIR = 260902T161623Z_CHARTS_hdf5 (32 inputs × 672 channels of 0.3 MHz ×
1 228 800 samples of 3.333 µs = 4.096 s; uint8 = 4 real bits high + 4 imaginary low, two's complement).
Processing per input (0–7, the only ones with signal; see 00_input_diagnostics.py):
  32-point FFT without window per coarse channel → 21 504 channels of 9.375 kHz every 106.67 µs,
  |x|², average in exact 1 ms bins (9.375 frames per bin, with fractional split at the edges).
Channel order: ascending frequency; fine channel j = 32·c + (k + 16), with k = −16..15 the index after fftshift,
centre = 300.0 + 0.3·c + 0.009375·k MHz (freq_start_MHz = 300.0 is assumed to be the centre of coarse channel 0).
Output: dynamic_spectrum.npy, float32 (8 inputs, 4096 ms, 21504 channels) = 2.8 GB, and frequencies_mhz.npy.
Usage: python 01_dynamic_spectrum.py
"""
from multiprocessing import Pool
import os
import sys
from pathlib import Path
import h5py, numpy as np

if "CHARTS_BASEBAND_DIR" not in os.environ:
    sys.exit("ERROR: export CHARTS_BASEBAND_DIR=/folder/260902T161623Z_CHARTS_hdf5 (site baseband)")
BB = Path(os.environ["CHARTS_BASEBAND_DIR"])
HERE = Path(__file__).resolve().parent
INPUTS = 8
NFFT, NCH_COARSE, NT = 32, 84, 1228800
NFR = NT // NFFT                     # 38 400 frames of 106.67 µs
NMS = 4096                           # 1 ms bins
FR_PER_MS = NFR / NMS                # 9.375
LUT = np.array([v - 16 if v > 7 else v for v in range(16)], np.float32)
OUTPUT = HERE / "dynamic_spectrum.npy"


def group(g):
    rest = None
    power = []
    for ch in range(7):
        with h5py.File(BB / f"bb_f{NCH_COARSE*g:04d}_{NCH_COARSE*g+NCH_COARSE:04d}_chunk{ch:05d}.h5", "r") as f:
            b = f["baseband"][:INPUTS]                                    # (8, 84, n)
        x = (LUT[b >> 4] + 1j * LUT[b & 15]).astype(np.complex64)
        if rest is not None:
            x = np.concatenate([rest, x], axis=-1)
        n = x.shape[-1] // NFFT * NFFT
        rest = x[..., n:]
        y = np.fft.fftshift(np.fft.fft(x[..., :n].reshape(INPUTS, NCH_COARSE, -1, NFFT), axis=-1), axes=-1)
        power.append((np.abs(y) ** 2).astype(np.float32))                  # (8, 84, nfr, 32)
    p = np.concatenate(power, axis=2)
    assert p.shape[2] == NFR, p.shape
    # average in 1 ms bins with fractional split: cumulative integral interpolated at the edges
    c = np.concatenate([np.zeros_like(p[:, :, :1]), np.cumsum(p, axis=2, dtype=np.float64)], axis=2)
    edges = np.arange(NMS + 1) * FR_PER_MS
    i0 = np.floor(edges).astype(int); fr = edges - i0
    i1 = np.minimum(i0 + 1, NFR)
    cb = c[:, :, i0] * (1 - fr)[None, None, :, None] + c[:, :, i1] * fr[None, None, :, None]
    ms = (np.diff(cb, axis=2) / FR_PER_MS).astype(np.float32)           # (8, 84, 4096, 32)
    out = np.lib.format.open_memmap(OUTPUT, mode="r+")
    out[:, :, g * NCH_COARSE * NFFT:(g + 1) * NCH_COARSE * NFFT] = ms.transpose(0, 2, 1, 3).reshape(INPUTS, NMS, -1)
    out.flush()
    return g


if __name__ == "__main__":
    np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.float32, shape=(INPUTS, NMS, 672 * NFFT))
    c = np.arange(672); k = np.arange(-16, 16)
    np.save(HERE / "frequencies_mhz.npy", (300.0 + 0.3 * c[:, None] + 0.3 / NFFT * k[None, :]).ravel())
    with Pool(8) as pool:
        for g in pool.imap_unordered(group, range(8)):
            print("group", g, "done", flush=True)
    print(OUTPUT)
