#!/usr/bin/env python3
"""Writes the site baseband channelized ×32 and integrated to 1 ms (dynamic_spectrum.npy, from 01_) as SIGPROC .fil.

For each input 0–7 and for the sum of the 8 (incoherent sum of powers):
  *_f32.fil  power |x|² averaged over 1 ms, float32, unnormalized (level² units of the 4+4 bits)
  *_8bit.fil calibrated per channel like the CHARTS 8 bits: y = 32 + 4·(P − median)/σ, with the median and
             σ = 1.4826·MAD of each channel over the 4.096 s; rounded and clipped to 0–255
Channels in descending order (fch1 = highest channel, foff < 0), like the group-3 .fil files; tsamp 1 ms.
Frequencies: centre = 300.0 + 0.3·c + 0.009375·k MHz (k = −16..15 after fftshift); 300.0 MHz is assumed to be the
centre of coarse channel 0. Start MJD taken from start_time_utc_us of the HDF5.
Output: fil/  (9 files × 2 versions; 352 MB the float32 ones, 88 MB the 8-bit ones)
"""
import struct
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "fil"; OUT.mkdir(exist_ok=True)
D = np.load(HERE / "dynamic_spectrum.npy", mmap_mode="r")
f = np.load(HERE / "frequencies_mhz.npy")
START_US = 1788365783335973                         # start_time_utc_us of the HDF5
MJD = 40587 + START_US / 1e6 / 86400


def header(nbits, name):
    def s(k): return struct.pack("<i", len(k)) + k.encode()
    def i(k, v): return s(k) + struct.pack("<i", v)
    def d(k, v): return s(k) + struct.pack("<d", v)
    return (s("HEADER_START") + s("source_name") + s(name) + i("machine_id", 0) + i("telescope_id", 0) +
            i("data_type", 1) + d("fch1", float(f[-1])) + d("foff", -float(f[1] - f[0])) + i("nchans", len(f)) +
            i("nbits", nbits) + i("nifs", 1) + d("tstart", MJD) + d("tsamp", 0.001) + s("HEADER_END"))


def write(P, name):
    P = P[:, ::-1]                                   # descending order
    with open(OUT / f"{name}_f32.fil", "wb") as fh:
        fh.write(header(32, name)); fh.write(np.ascontiguousarray(P, np.float32).tobytes())
    med = np.median(P, 0); sig = 1.4826 * np.median(np.abs(P - med), 0) + 1e-6
    y = np.clip(np.rint(32 + 4 * (P - med) / sig), 0, 255).astype(np.uint8)
    with open(OUT / f"{name}_8bit.fil", "wb") as fh:
        fh.write(header(8, name)); fh.write(y.tobytes())
    print(name, f"8 bits: {(y == 255).mean() * 100:.2f} % at 255, {(y == 0).mean() * 100:.3f} % at 0", flush=True)


total = np.zeros(D.shape[1:], np.float64)
for a in range(D.shape[0]):
    P = np.asarray(D[a]); total += P
    write(P, f"site_260902_input{a}")
write(total.astype(np.float32), "site_260902_sum8")
