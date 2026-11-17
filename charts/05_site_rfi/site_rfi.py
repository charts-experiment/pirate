"""RFI measured at the site (baseband of 2026-09-02) to add to the group-3 simulations (21 504 channels, 1 ms, 8 bits).

The model (site_rfi_model.npz, from 02_rfi_model.py) gives the RFI in units of the system power P_sys:
  r_p(f)    persistent (median over 4.096 s)
  e(t,f)    fast (excess > 5σ over r_p, with neighbours > 3σ), 4096 ms × 21 504 channels
In the .fil the system power is P_sys = σ·√(Δν τ) = 4·√9.375 = 12.25 counts (floor 32, σ = 4 counts).

For each .fil pixel, with s = x − 32 (signal + noise of σ0 = 4), g = 10^(−attenuation_db/10),
rp = g·r_p(f) and re = g·e(t,f):
  "visible" channel (line with 10 log10(1 + rp) > line_threshold_db; the 8-bit stage does not flatten it):
      y = 32 + s + P_sys·(rp + re) + σ0·√((1 + rp + re)² − 1)·N(0,1)
  "calibrated" channel (the per-channel offset/scale of the 8-bit stage normalizes the persistent power):
      y = 32 + [s + P_sys·re + σ0·√((1 + rp + re)² − 1)·N(0,1)] / (1 + rp)
The N(0,1) term adds the radiometer noise of the RFI power (σ grows with the total power).
In calibrated channels the noise stays at σ0 where there is no fast RFI, and the FRB signal is divided by (1 + rp).
The result is rounded and clipped to 0–255, like the real 8-bit stage.

The measurement lasts 4.096 s. For longer files, a bootstrap of contiguous blocks is used: each block of
`block_ms` takes a random offset inside the capture (circular). In real time all beams use the same clock
(the RFI enters through the sidelobes and reaches all beams at once); `gain` lets each beam see it with a
different sidelobe gain.

Works with numpy or cupy (xp).

Model keys: frequencies_mhz, r_p, e, psys_counts. Model files written before the translation use frecuencias_mhz and
psys_cuentas; both spellings are accepted.
"""
import os
from pathlib import Path

import numpy as np

MODEL = Path(os.environ.get("CHARTS_RFI_MODEL", Path(__file__).resolve().parent / "site_rfi_model.npz"))
FLOOR, SIGMA0 = 32.0, 4.0


class SiteRfi:
    def __init__(self, fil_freqs_mhz, line_threshold_db=10.0, attenuation_db=0.0, fast=True, persistent=True,
                 block_ms=4096, seed=0, model=MODEL, xp=np):
        m = np.load(model)
        self.xp = xp
        key = lambda en, es: en if en in m.files else es      # English keys, or those of older model files
        fm = m[key("frequencies_mhz", "frecuencias_mhz")]
        # .fil channel -> nearest measured channel in frequency (the grids differ by < 0.2 %)
        idx = np.clip(np.searchsorted(fm, fil_freqs_mhz), 1, len(fm) - 1)
        idx = np.where(np.abs(fm[idx - 1] - fil_freqs_mhz) < np.abs(fm[idx] - fil_freqs_mhz), idx - 1, idx)
        g = 10 ** (-attenuation_db / 10)
        rp = np.clip(m["r_p"][idx], 0, None) * g if persistent else np.zeros(len(idx), np.float32)
        visible = 10 * np.log10(1 + rp) > line_threshold_db
        self.psys = float(m[key("psys_counts", "psys_cuentas")])
        self.rp = xp.asarray(rp.astype(np.float32))
        self.visible = xp.asarray(visible)
        e = m["e"][:, idx] if fast else np.zeros((len(m["e"]), len(idx)), np.float16)
        self.e = xp.asarray((e.astype(np.float32) * g).astype(np.float16))   # float16: 176 MB, shared between threads
        self.nt = self.e.shape[0]
        self.block_ms = int(block_ms)
        self.seed = int(seed)
        self.n_visible = int(visible.sum())
        self.median_snr_loss = float(np.median(np.where(visible, 1.0, 1 / (1 + rp))))

    def _rows(self, t0, n):
        """Time indices in the capture for samples t0..t0+n−1 (bootstrap of contiguous blocks)."""
        t = np.arange(t0, t0 + n)
        b = t // self.block_ms
        d = {k: int(np.random.default_rng([self.seed, int(k)]).integers(self.nt)) for k in np.unique(b)}
        return (t % self.block_ms + np.array([d[k] for k in b])) % self.nt     # deterministic: thread-safe

    def initial_offset(self):
        return int(np.random.default_rng([self.seed, 0]).integers(self.nt))

    def apply(self, block_u8, t0, gain=1.0, rng=None):
        """block_u8: (n, nchan) uint8 in the .fil channel order; t0: sample (ms) of the first row.
        gain: extra factor for this beam (sidelobe). rng: generator for the RFI noise
        (numpy Generator or cupy RandomState; one per thread). Returns uint8 with the RFI added."""
        xp = self.xp
        if rng is None:
            rng = np.random.default_rng([self.seed, t0, 1]) if xp is np else xp.random.RandomState(self.seed + t0)
        b = xp.asarray(block_u8)
        out = xp.empty(b.shape, xp.uint8)
        rows = xp.asarray(self._rows(t0, b.shape[0]))
        rp = self.rp * gain
        div = xp.where(self.visible, 1.0, 1.0 / (1 + rp)).astype(xp.float32)
        step = 4096 if xp is np else 256          # on GPU, small temporaries: several threads share the memory with pirate
        for i in range(0, b.shape[0], step):
            x = b[i:i + step].astype(xp.float32) - FLOOR
            re = self.e[rows[i:i + step]].astype(xp.float32) * gain
            tot = 1 + rp[None, :] + re
            y = x + self.psys * re + SIGMA0 * xp.sqrt(xp.maximum(tot * tot - 1, 0)) * rng.standard_normal(x.shape, dtype=xp.float32)
            y = xp.where(self.visible[None, :], y + self.psys * rp[None, :], y * div[None, :])
            out[i:i + step] = xp.clip(xp.rint(y + FLOOR), 0, 255).astype(xp.uint8)
        return out
