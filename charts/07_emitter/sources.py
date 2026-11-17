"""Data sources for the emitter: where the 8-bit block of each beam comes from.

- .fil dataset (group 3 of the simulator): the files are shuffled and placed one after another in the streams of
  the beams that carry data, with noise before (prepad) and after (tail). `assignment.csv` records which file goes
  in which beam and from which sample: it is the ground truth for the evaluation.
- Noise: gaps between files and padding, gaussian noise with the dataset scale (floor 32, σ = 4 counts).
- Precomputed fill (beams without data): 8 noise blocks already converted to 4 bits that are reused in rotation;
  it only costs a memory copy. Useful to test many beams without the emitter being the bottleneck.

To plug in another source (for example live data), all that is needed is a class with the method
`block(t0, T, out) -> list of noise segments` that fills `out` [T, nchan] uint8 in the .fil order.
"""
import csv
import random
from pathlib import Path

import numpy as np

# Folder names of the group-3 dataset on disk (data format of the simulator; kept as generated)
FOLDERS = {"A": "A_snr_fijo", "B": "B_amplitud_fija", "NEG": "negativos", "P": "parcial"}


def header_length(path):
    with open(path, "rb") as f:
        raw = f.read(8192)
    return raw.index(b"HEADER_END") + len(b"HEADER_END")


def build_streams(set_dir, sets, nbeams, prepad, tail, seed, limit=None, T=2048):
    """Distributes the files of the sets over nbeams streams. Returns (segments per beam, assignment, total samples)."""
    rows = []
    for s in sets:
        with open(Path(set_dir) / FOLDERS[s] / "manifest.csv", newline="") as f:
            m = list(csv.DictReader(f))
        rows += m[:limit] if limit else m
    random.Random(seed).shuffle(rows)
    length = [prepad] * nbeams
    segments = [[("noise", prepad)] if prepad else [] for _ in range(nbeams)]
    assignment = []
    for r in rows:
        b = int(np.argmin(length))
        n = int(r["nsamp"])
        assignment.append({"beam": b, "start_sample": length[b], **r})
        segments[b].append(("fil", r["fil_path"], n))
        length[b] += n
    total = ((max(length) + tail + T - 1) // T) * T
    for b in range(nbeams):
        if total > length[b]:
            segments[b].append(("noise", total - length[b]))
    return segments, assignment, total


def noise_8bit(T, nchan, n=8, seed=99):
    """n blocks [T, nchan] uint8 of gaussian noise with the dataset scale (floor 32, σ = 4 counts)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.clip(np.floor(rng.standard_normal((T, nchan), dtype=np.float32) * 4 + 32.5), 0, 255).astype(np.uint8)
                     for _ in range(n)])


class BeamReader:
    """Block [a, a+T) of the stream of one beam as uint8 [T, nchan] in the .fil order.

    Noise gaps are filled by copying from `pool` (precomputed noise blocks, see noise_8bit) if given;
    otherwise new noise is generated in each block (about 5 times slower on the CPU)."""

    def __init__(self, segments, nchan, seed, pool=None, beam=0):
        self.nchan = nchan
        self.pool, self.beam = pool, beam
        self.rng = np.random.default_rng(seed)
        self.table, t = [], 0
        for sg in segments:
            n = sg[1] if sg[0] == "noise" else sg[2]
            self.table.append((t, t + n, sg[0], sg[1] if sg[0] == "fil" else None))
            t += n
        self.mm = {}

    def _mm(self, path):
        if path not in self.mm:
            self.mm.clear()                         # one open file at a time per beam
            d = np.memmap(path, dtype=np.uint8, mode="r", offset=header_length(path))
            self.mm[path] = d.reshape(-1, self.nchan)
        return self.mm[path]

    def block(self, a, T, out):
        """Fills out with the file segments and generates the noise of the gaps. Returns the noise segments."""
        b, noise = a + T, []
        for (i0, i1, kind, path) in self.table:
            lo, hi = max(a, i0), min(b, i1)
            if hi <= lo:
                continue
            if kind == "fil":
                out[lo - a:hi - a] = self._mm(path)[lo - i0:hi - i0]
            else:
                if self.pool is not None:
                    # channels rotated by beam: without this, beams b, b+8, ... carry the same noise at the same
                    # time and produce the same noise events (false coincidences between beams)
                    blk, s = self.pool[(a // T + self.beam) % len(self.pool)], (self.beam * 2687) % self.nchan
                    out[lo - a:hi - a, :self.nchan - s] = blk[lo - a:hi - a, s:]
                    out[lo - a:hi - a, self.nchan - s:] = blk[lo - a:hi - a, :s]
                else:
                    x = self.rng.standard_normal((hi - lo, self.nchan), dtype=np.float32)
                    out[lo - a:hi - a] = np.clip(np.floor(x * 4 + 32.5), 0, 255).astype(np.uint8)
                noise.append((lo - a, hi - a))
        return noise


def precomputed_fill(nchan, T, n=8, zeros=False, seed=4242):
    """n packed blocks [nchan, T/2] uint8 (fixed quantization) for the beams without data."""
    from quantization import fixed
    from protocol import pack_int4
    out = np.zeros((n, nchan, T // 2), np.uint8)
    if not zeros:
        rng = np.random.default_rng(seed)
        for k in range(n):
            x = rng.standard_normal((T, nchan), dtype=np.float32)
            out[k] = pack_int4(fixed(np.clip(np.floor(x * 4 + 32.5), 0, 255).astype(np.uint8)))
    return out
