#!/usr/bin/env python
"""pirate S/N calibration: dispersed boxcar pulses of known S/N in gaussian noise.

True S/N = A * sqrt(W * nchan) for a boxcar of W samples and amplitude A (in sigmas per
channel and sample), summing all channels with the exact delay of each channel centre.
It also counts candidates in pure noise (S/N >= 5).

Usage: python calibrate_snr.py [--nchan 672] [--snr 20] [--dms 200,500,1000,1500] [--widths 1,4,16] [--int4 2]
"""
import argparse

import numpy as np
import cupy as cp

from pirate_fil_search import PirateSearcher, DEFAULT_CONFIG, group_nms, quantize_int4

K_DM = 4148.808  # s MHz^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchan", type=int, default=672)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--dms", default="200,500,1000,1500")
    ap.add_argument("--widths", default="1,4,16")
    ap.add_argument("--int4", type=float, default=0.0)
    a = ap.parse_args()

    foff = -201.6 / (a.nchan - 1)
    hdr = dict(nchans=a.nchan, fch1=501.6, foff=foff, tsamp=1e-3, nbits=8)
    centres = 501.6 + foff * np.arange(a.nchan)        # .fil order (descending)
    rng = np.random.default_rng(1)

    print(f"nchan={a.nchan}  injected S/N={a.snr}")
    print(f"{'DM':>6} {'W':>3} {'pirate S/N':>10} {'ratio':>8} {'meas. DM':>8} {'meas. W':>7} {'dt (ms)':>8}")
    for dm in [float(x) for x in a.dms.split(",")]:
        for W in [int(x) for x in a.widths.split(",")]:
            b = PirateSearcher(DEFAULT_CONFIG, hdr, freq_order="asc")
            T = b.nt_in
            nchunks = 2 + int(np.ceil(K_DM * dm * (300.0 ** -2 - 501.6 ** -2) / 1e-3 / T)) + 4
            n = nchunks * T
            x = rng.standard_normal((n, a.nchan), dtype=np.float32)
            t0 = 2 * T + 100                                 # arrival at 501.6 MHz (sample)
            A = a.snr / np.sqrt(W * a.nchan)
            delay = np.rint(K_DM * dm * (centres ** -2 - 501.6 ** -2) / 1e-3).astype(int)
            for c in range(a.nchan):
                x[t0 + delay[c]: t0 + delay[c] + W, c] += A
            xg = cp.asarray(x[:, ::-1].T.copy())             # (nfreq asc, n)
            cands = []
            for i in range(nchunks):
                blk = xg[:, i * T:(i + 1) * T].astype(cp.float16)
                if a.int4:
                    blk = quantize_int4(blk, a.int4)
                cands += b.chunk(blk, 5.0)
            c = np.array(cands).reshape(-1, 4)
            c[:, 1] = b.to_t_top(c[:, 0], c[:, 1])
            t_true = t0 * 1e-3 + (W - 1) / 2 * 1e-3
            near = c[np.abs(c[:, 1] - t_true) < 0.2]
            if not len(near):
                print(f"{dm:6.0f} {W:3d}   not detected")
                continue
            m = near[np.argmax(near[:, 2])]
            print(f"{dm:6.0f} {W:3d} {m[2]:10.2f} {m[2] / a.snr:8.3f} {m[0]:8.1f} {m[3]:7.1f} {(m[1] - t_true) * 1e3:8.1f}")
            far = group_nms(c[np.abs(c[:, 1] - t_true) > 1.0])
            dur = (n - 2 * T) * 1e-3
            print(f"{'':>10} noise: {len(far)} grouped candidates S/N>=5 in {dur:.0f} s "
                  f"(max noise S/N {far[:, 2].max() if len(far) else 0:.2f})")


if __name__ == "__main__":
    main()
