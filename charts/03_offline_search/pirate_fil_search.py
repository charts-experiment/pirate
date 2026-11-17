#!/usr/bin/env python
"""FRB search with pirate (GPU tree dedisperser) on an 8-bit SIGPROC .fil.

It does not go through int4 or ASDF: each chunk of nt_in samples is normalized per channel
((x - mean) / sigma, in float16) and written directly into the input buffer of the
GpuDedisperser. The S/N weights are computed with unit variance per channel.

Output: .npy with columns (dm, t_top_s, snr, width_samples), with t_top_s = arrival at the
highest frequency of the .fil (channel centre; 501.6 MHz in CHARTS).

Usage (pirate conda environment):
    python pirate_fil_search.py FILE.fil OUTPUT.npy [--config 06_configs/charts_search.yml]
        [--snr 5] [--dm-min 20] [--dm-max 3000] [--tail-chunks 3] [--no-grouping]
        [--freq-order {asc,desc}] [--int4 LEVELS_PER_SIGMA]
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "06_configs" / "charts_search.yml"

_HDR_INT = {"nchans", "nbits", "nifs", "telescope_id", "machine_id", "data_type", "barycentric",
            "pulsarcentric", "nbeams", "ibeam"}
_HDR_DBL = {"tsamp", "fch1", "foff", "tstart", "src_raj", "src_dej", "az_start", "za_start", "refdm"}


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(8192)
    hlen = raw.index(b"HEADER_END") + len(b"HEADER_END")
    hdr, i = {}, len(b"HEADER_START") + 4
    while i < hlen - len(b"HEADER_END"):
        n = struct.unpack("<i", raw[i:i + 4])[0]
        key = raw[i + 4:i + 4 + n].decode()
        i += 4 + n
        if key in _HDR_INT:
            hdr[key] = struct.unpack("<i", raw[i:i + 4])[0]
            i += 4
        elif key in _HDR_DBL:
            hdr[key] = struct.unpack("<d", raw[i:i + 8])[0]
            i += 8
        elif key in ("source_name", "rawdatafile"):
            m = struct.unpack("<i", raw[i:i + 4])[0]
            hdr[key] = raw[i + 4:i + 4 + m].decode(errors="replace")
            i += 4 + m
    return hlen, hdr


class PirateSearcher:
    """A 1-beam GpuDedisperser, configured from the .fil header.

    It can be reused for several files with the same band/channels/tsamp
    (the tree state is reset by creating a new searcher)."""

    def __init__(self, config_path, hdr, freq_order="asc"):
        import cupy as cp
        import ksgpu
        import pirate_frb as pf
        from pirate_frb.core import BumpAllocator, CudaStreamPool

        self.cp = cp
        nchan, foff, fch1 = hdr["nchans"], hdr["foff"], hdr["fch1"]
        centres = fch1 + foff * np.arange(nchan)
        self.f_lo_edge = float(centres.min() - abs(foff) / 2)
        self.f_hi_edge = float(centres.max() + abs(foff) / 2)
        self.f_top = float(centres.max())          # t_top reference (centre of the highest channel)
        self.tsamp = hdr["tsamp"]
        # The .fil is in descending order if foff < 0. 'asc' = pirate's channel 0 is the
        # lowest frequency.
        self.reverse = (foff < 0) == (freq_order == "asc")

        cfg = pf.DedispersionConfig.from_yaml(str(config_path))
        cfg.zone_nfreq = [nchan]
        cfg.zone_freq_edges = [self.f_lo_edge, self.f_hi_edge]
        cfg.time_sample_ms = self.tsamp * 1e3
        cfg.beams_per_gpu = 1
        cfg.beams_per_batch = 1
        cfg.num_active_batches = 1
        cfg.validate()
        self.config = cfg
        self.dm_per_sample = cfg.dm_per_unit_delay()

        self.plan = pf.DedispersionPlan(cfg)
        self.nt_in = self.plan.nt_in
        self.ntrees = self.plan.ntrees
        self.trees = self.plan.trees
        self.pool = CudaStreamPool(1)
        self.dd = pf.GpuDedisperser(self.plan, self.pool, cuda_device_id=0, num_consumers=1,
                                    nbatches_out=1, nbatches_wt=1, initial_chunk=0)
        self._ga = BumpAllocator(ksgpu.af_gpu | ksgpu.af_zero, -1)
        self._ha = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, -1)
        self.dd.allocate(self._ga, self._ha)
        self.dd.fill_analytic_weights(np.ones(nchan, dtype=np.float64))
        self.Dcores = list(self.dd.Dcores)
        self.seq = 0

        # Delay (in s) from the bottom edge of the band to f_top, per unit of DM.
        k_edge = self.tsamp / self.dm_per_sample   # s per (pc cm^-3) between band edges
        frac = (self.f_lo_edge ** -2 - self.f_top ** -2) / (self.f_lo_edge ** -2 - self.f_hi_edge ** -2)
        self.s_per_dm_to_top = k_edge * frac

    def chunk(self, block_gpu_f16, threshold, group=True):
        """block_gpu_f16: cupy (nfreq, nt_in) float16, already normalized and in pirate's order.
        Returns a list of (dm, t_bottom_s, snr, width_samples) for the chunk."""
        cp = self.cp
        from cupyx.scipy.ndimage import maximum_filter
        stream = cp.cuda.get_current_stream()
        with self.dd.get_input(self.seq, stream=stream) as arr:
            arr[0] = block_gpu_f16
        rows = []
        with self.dd.get_output(self.seq, stream=stream) as out:
            for it in range(self.ntrees):
                m = cp.asarray(out.out_max[it])[0].astype(cp.float32)   # (ndm, nt)
                sel = m >= threshold
                if group:
                    sel &= m == maximum_filter(m, size=3, mode="constant", cval=0.0)
                idm, itm = cp.nonzero(sel)
                if idm.size == 0:
                    continue
                snr = m[idm, itm].get()
                tok = cp.asarray(out.out_argmax[it])[0][idm, itm].get()
                idm, itm = idm.get(), itm.get()
                for k in range(len(snr)):
                    fmin, fmax, tlo, thi, p = self.plan.decode_argmax(
                        int(tok[k]), it, self.Dcores[it], int(idm[k]), int(itm[k]))
                    _, _, dm, ts, w = self.plan.decode_argmax2(it, fmin, fmax, tlo, thi, p)
                    t_bot = (self.seq * self.nt_in + ts) * self.tsamp
                    rows.append((dm, t_bot, float(snr[k]), w))
        self.seq += 1
        return rows

    def to_t_top(self, dm, t_bot):
        return t_bot - dm * self.s_per_dm_to_top


def group_nms(c, dt_s=0.1, dm_frac=0.1):
    """Non-maximum suppression: goes through the candidates by descending S/N and drops those
    within dt_s and dm_frac of one already accepted (duplicates between trees/chunks)."""
    if not len(c):
        return c
    c = c[np.argsort(-c[:, 2])]
    ok = []
    for row in c:
        if not any(abs(row[1] - o[1]) <= dt_s and abs(row[0] - o[0]) <= dm_frac * max(o[0], 1.0) for o in ok):
            ok.append(row)
    return np.array(ok, dtype=np.float64)


def quantize_int4(block, levels_per_sigma):
    """Simulates pirate's network format: int4 in [-7, 7] with scale/offset per
    (channel, block of 256 samples); -8 is reserved for masked samples.
    levels_per_sigma = how many int4 levels one noise sigma spans."""
    import cupy as cp
    nf, T = block.shape
    x = block.astype(cp.float32).reshape(nf, T // 256, 256)
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True)
    scale = cp.where(sd > 0, sd / levels_per_sigma, 1.0)
    q = cp.clip(cp.rint((x - mu) / scale), -7, 7)
    return (q * scale + mu).reshape(nf, T).astype(cp.float16)


def search_fil(path, config_path=DEFAULT_CONFIG, snr=5.0, dm_min=20.0, dm_max=3000.0,
               tail_chunks=3, group=True, freq_order="asc", seed=0, int4_levels_per_sigma=0.0):
    import cupy as cp
    hlen, hdr = read_header(path)
    assert hdr["nbits"] == 8 and hdr.get("nifs", 1) == 1, hdr
    nchan = hdr["nchans"]
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=hlen)
    data = data[: (data.size // nchan) * nchan].reshape(-1, nchan)
    nsamp = data.shape[0]

    t0 = time.time()
    b = PirateSearcher(config_path, hdr, freq_order=freq_order)
    t_init = time.time() - t0

    # Per-channel statistics over the whole file (a burst of S/N ~10 does not move them).
    d_gpu = cp.asarray(np.ascontiguousarray(data))
    mu = d_gpu.mean(axis=0, dtype=cp.float32)
    sd = d_gpu.std(axis=0, dtype=cp.float32)
    inv = cp.where(sd > 0, 1.0 / cp.maximum(sd, 1e-6), 0.0).astype(cp.float32)
    if b.reverse:
        d_gpu, mu, inv = d_gpu[:, ::-1], mu[::-1], inv[::-1]
    rng = cp.random.default_rng(seed)

    T = b.nt_in
    nchunks = -(-nsamp // T) + tail_chunks
    cands = []
    t1 = time.time()
    for i in range(nchunks):
        a, z = i * T, min((i + 1) * T, nsamp)
        block = cp.empty((nchan, T), dtype=cp.float16)
        n = max(0, z - a)
        if n:
            block[:, :n] = ((d_gpu[a:z].astype(cp.float32) - mu) * inv).T.astype(cp.float16)
        if n < T:   # tail: normalized gaussian noise (dead channels at 0)
            block[:, n:] = (rng.standard_normal((nchan, T - n), dtype=cp.float32)
                            * (inv > 0)[:, None]).astype(cp.float16)
        if int4_levels_per_sigma:
            block = quantize_int4(block, int4_levels_per_sigma)
        cands += b.chunk(block, snr, group=group)
    cp.cuda.get_current_stream().synchronize()
    t_search = time.time() - t1

    c = np.array(cands, dtype=np.float64).reshape(-1, 4)
    if len(c):
        c[:, 1] = b.to_t_top(c[:, 0], c[:, 1])
        c = c[(c[:, 0] >= dm_min) & (c[:, 0] <= dm_max)]
    if group:
        c = group_nms(c)
    info = dict(nsamp=nsamp, nchan=nchan, nt_in=T, nchunks=nchunks, ntrees=b.ntrees,
                dm_max_per_tree=[t.dm_max for t in b.trees], t_init_s=t_init, t_search_s=t_search,
                n_cands=int(len(c)), reversed=b.reverse)
    return c, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fil")
    ap.add_argument("output", help="candidates .npy (dm, t_top_s, snr, width_samples); a .json is also written")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--dm-min", type=float, default=20.0)
    ap.add_argument("--dm-max", type=float, default=3000.0)
    ap.add_argument("--tail-chunks", type=int, default=3)
    ap.add_argument("--no-grouping", action="store_true")
    ap.add_argument("--freq-order", choices=["asc", "desc"], default="asc")
    ap.add_argument("--int4", type=float, default=0.0, metavar="LEVELS_PER_SIGMA",
                    help="simulate requantization to int4 (e.g. 2 or 3 levels per sigma); 0 = float16 directly")
    a = ap.parse_args()
    c, info = search_fil(a.fil, a.config, a.snr, a.dm_min, a.dm_max, a.tail_chunks,
                         not a.no_grouping, a.freq_order, int4_levels_per_sigma=a.int4)
    np.save(a.output, c.astype(np.float32))
    Path(a.output).with_suffix(".json").write_text(json.dumps(info))
    print(json.dumps(info))


if __name__ == "__main__":
    sys.exit(main())
