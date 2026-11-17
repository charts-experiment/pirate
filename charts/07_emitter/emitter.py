#!/usr/bin/env python3
"""Standalone emitter: sends intensity data to pirate with its network protocol, as an X-engine would.

Does not depend on pirate: Python >= 3.9, numpy and PyYAML (cupy optional, with `gpu: true`). Can run on another machine.

  Sources ─► preparation processes ─► shared memory (2 buffers) ─► 1 thread per TCP connection ─► pirate receivers
             (read, optional RFI,      [beam, channel, time/2] 4 bits   (its channel range,
              quantize, mask)           + scale/offset per minichunk     minichunk by minichunk)

Each connection mimics an X-engine node sending a contiguous channel range for all beams.
Pacing: `real-time` (one minichunk every 256 samples, like the X-engine; `speed` scales it), `ack` (each
connection waits for pirate's acknowledgement before the next minichunk: as fast as pirate accepts) or
`free` (no pause).

Usage:
  python 07_emitter/emitter.py --config 06_configs/test_64b.yml --out results/run [options that override the YAML]
  e.g. --limit 20 --rfi --mask 05_site_rfi/mask_m1.npy --pacing ack --receivers 10.0.0.2:5000,10.0.0.2:5001

Outputs in --out: assignment.csv (which file goes in which beam and from which sample), emitter.json, emitter.log.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sources import FOLDERS, BeamReader, build_streams, noise_8bit, precomputed_fill  # noqa: E402
from protocol import MINICHUNK_SAMPLES, Connection, metadata_yaml, pack_int4  # noqa: E402

DT_NS_PER_SEQ = 5000                                 # 5 µs per seq count; 1 ms = 200 counts


# ---------------------------------------------------------------------------------------------- configuration
def load_config(a):
    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    t, net, e = cfg.get("test", {}), cfg.get("network", {}), cfg.get("emitter", {})
    root = HERE.parent                               # the charts/ folder: 05_site_rfi/ and relative paths (independent of the YAML location)
    c = dict(
        beams=t.get("beams", 64), channels=t.get("channels", 21504), band=t.get("band_mhz", [299.8497764530551, 501.7502235469449]),
        sample_ms=t.get("sample_ms", 1.0), T=t.get("block_samples", 2048),
        receivers=net.get("receivers", ["127.0.0.1:5000"]), connections=net.get("connections_per_receiver", 4),
        set_dir=os.path.expandvars(str(e.get("set_dir") or "$CHARTS_SET_DIR")), sets=e.get("sets", ["A", "B", "NEG", "P"]),
        limit=e.get("limit"), prepad_s=e.get("prepad_s", 10.0), tail_s=e.get("tail_s", 60.0),
        beams_with_data=e.get("beams_with_data"), fill=e.get("fill", "noise"),
        quantization=e.get("quantization", "fixed"), mask=e.get("mask"), rfi=e.get("rfi", {}) or {},
        pacing=e.get("pacing", "real-time"), speed=e.get("speed", 1.0), processes=e.get("processes", 8),
        gpu=e.get("gpu", False), seed=e.get("seed", 1), rfi_dir=str(root / "05_site_rfi"),
        precomputed_noise=e.get("precomputed_noise", True),
    )
    for k in ("receivers", "connections", "set_dir", "limit", "beams_with_data", "fill", "quantization", "mask",
              "pacing", "speed", "processes", "seed", "beams"):
        v = getattr(a, k, None)
        if v is not None:
            c[k] = v.split(",") if k == "receivers" and isinstance(v, str) else v
    if a.rfi:
        c["rfi"] = dict(c["rfi"], enabled=True)
    if a.gpu:
        c["gpu"] = True
    if c["rfi"].get("enabled"):
        c["quantization"] = "per_channel"            # with RFI, per-channel normalization is needed (see docs)
    if c["mask"] and not os.path.isabs(c["mask"]):
        c["mask"] = str((Path.cwd() / c["mask"]).resolve()) if (Path.cwd() / c["mask"]).exists() else str(root / c["mask"])
    return c


# ------------------------------------------------------------------------------------------ preparation processes
def prepare(w, beams, c, segments, n_data, total, shm_data, shm_so, tasks, done):
    """Process w: prepares its beams, block by block, in the buffer (slot) it is told."""
    parent = os.getppid()
    try:                                           # Linux: the kernel kills this process if the main emitter dies
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)       # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass                                       # other systems: the os.getppid() check below remains
    if os.getppid() != parent:                     # it died just before we asked
        return
    nb, nch, T = c["beams"], c["channels"], c["T"]
    shm_d, shm_s = shared_memory.SharedMemory(name=shm_data), shared_memory.SharedMemory(name=shm_so)  # keep references
    nmc = T // 256
    data = np.ndarray((2, nmc, nb, nch, 128), np.uint8, buffer=shm_d.buf)     # send layout
    so = np.ndarray((2, nmc, nb, nch, 2), np.float16, buffer=shm_s.buf)
    xp = np
    if c["gpu"]:
        import cupy as xp                            # noqa: F811
    import quantization as qz
    per_channel = c["quantization"] == "per_channel"
    so_base = qz.base_scales(nch, T, per_channel)
    mask = None
    if c["mask"]:
        f_asc = c["band"][0] + (np.arange(nch) + 0.5) * (c["band"][1] - c["band"][0]) / nch
        fm = np.load(Path(c["rfi_dir"]) / "frequencies_mhz.npy"); mk = np.load(c["mask"]).astype(bool)
        idx = np.clip(np.searchsorted(fm, f_asc), 1, len(fm) - 1)
        idx = np.where(np.abs(fm[idx - 1] - f_asc) < np.abs(fm[idx] - f_asc), idx - 1, idx)
        mask = mk[idx]                               # ascending order
        qz.apply_mask(np.zeros((nch, 1), np.uint8), so_base, mask)
    rfi = None
    if c["rfi"].get("enabled"):
        sys.path.insert(0, c["rfi_dir"])
        from site_rfi import SiteRfi
        f_fil = c["band"][1] - (np.arange(nch) + 0.5) * (c["band"][1] - c["band"][0]) / nch
        r = c["rfi"]
        rfi = SiteRfi(f_fil, line_threshold_db=r.get("line_threshold_db", 10.0), attenuation_db=r.get("attenuation_db", 0.0),
                      fast=r.get("fast", True), persistent=r.get("persistent", True), seed=c["seed"], xp=xp)
    fill = precomputed_fill(nch, T, zeros=c["fill"] == "zeros") if n_data < nb else None
    if fill is not None:                             # [n, nch, T/2] -> [n, nmc, nch, 128]
        fill = np.ascontiguousarray(fill.reshape(len(fill), nch, nmc, 128).transpose(0, 2, 1, 3))
    so_base_t = np.ascontiguousarray(so_base.transpose(1, 0, 2))             # [nmc, nch, 2]
    pool = noise_8bit(T, nch, seed=99 + w) if c.get("precomputed_noise", True) else None
    readers = {b: BeamReader(segments[b], nch, seed=1000 + b, pool=pool, beam=b) for b in beams if b < n_data}
    buf = np.empty((T, nch), np.uint8)
    rng = np.random.default_rng(7000 + w) if xp is np else xp.random.RandomState(7000 + w)
    while True:
        try:
            task = tasks.get(timeout=2)
        except queue.Empty:
            if os.getppid() != parent:             # the main emitter died (kill): do not stay orphaned
                return
            continue
        if task is None:
            return
        ichunk, slot = task
        for b in beams:
            if b >= n_data:
                blk, s = fill[(ichunk + b) % len(fill)], (b * 2687) % nch          # channels rotated by beam
                data[slot, :, b, :nch - s] = blk[..., s:, :]
                data[slot, :, b, nch - s:] = blk[..., :s, :]
                so[slot, :, b] = so_base_t
                continue
            readers[b].block(ichunk * T, T, buf)
            g = buf
            if rfi is not None:
                g = rfi.apply(buf, ichunk * T, rng=rng)
            q = qz.per_channel(g, xp) if per_channel else qz.fixed(g, xp)
            if mask is not None:
                q[xp.asarray(mask), :] = qz.MISSING
            p = pack_int4(q)
            p = p.get() if xp is not np else p                                 # [nch, T/2]
            data[slot, :, b] = p.reshape(nch, nmc, 128).transpose(1, 0, 2)
            so[slot, :, b] = so_base_t
        done.put((w, ichunk))


# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="test YAML (06_configs/test_64b.yml)")
    ap.add_argument("--out", required=True, type=Path, help="output folder")
    ap.add_argument("--receivers", help="comma-separated list of host:port")
    ap.add_argument("--connections", type=int, help="connections per receiver")
    ap.add_argument("--set-dir"); ap.add_argument("--limit", type=int, help="first N files of each set")
    ap.add_argument("--beams", type=int); ap.add_argument("--beams-with-data", type=int)
    ap.add_argument("--fill", choices=["noise", "zeros"], help="content of the beams without data")
    ap.add_argument("--quantization", choices=["fixed", "per_channel"])
    ap.add_argument("--mask"); ap.add_argument("--rfi", action="store_true", help="add the site RFI")
    ap.add_argument("--pacing", choices=["real-time", "ack", "free"]); ap.add_argument("--speed", type=float)
    ap.add_argument("--processes", type=int); ap.add_argument("--gpu", action="store_true"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    c = load_config(a)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "emitter.log", "a")

    def info(msg):
        line = f"{time.strftime('%F %T')} emitter: {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    nb, nch, T = c["beams"], c["channels"], c["T"]
    n_data = nb if c["beams_with_data"] is None else max(1, min(c["beams_with_data"], nb))
    seq_per_sample = int(round(c["sample_ms"] * 1e6 / DT_NS_PER_SEQ))
    nmc = T // MINICHUNK_SAMPLES
    if not c["set_dir"] or "$" in c["set_dir"] or not (Path(c["set_dir"]) / FOLDERS[c["sets"][0]] / "manifest.csv").exists():
        sys.exit(f"ERROR: data not found in set_dir={c['set_dir']!r}. export CHARTS_SET_DIR=/path/to/group3 "
                 "(with A_snr_fijo/, B_amplitud_fija/, ... and their manifest.csv), or --set-dir, or emitter.set_dir in the YAML")
    segments, assignment, total = build_streams(c["set_dir"], c["sets"], n_data, int(c["prepad_s"] * 1000),
                                                int(c["tail_s"] * 1000), c["seed"], c["limit"], T)
    nchunks = total // T
    with open(a.out / "assignment.csv", "w", newline="") as f:
        import csv
        fields = list(dict.fromkeys(k for r in assignment for k in r))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(assignment)
    info(f"{len(assignment)} files in {n_data} of {nb} beams; {total / 1000:.0f} s per beam ({nchunks} blocks of {T}); "
         f"{nch} channels; quantization {c['quantization']}; RFI {'yes' if c['rfi'].get('enabled') else 'no'}; "
         f"mask {c['mask'] or 'no'}; pacing {c['pacing']} ×{c['speed']}")

    # shared memory: 2 buffers of [beam, channel, T/2] plus scales/offsets
    # send layout [slot, minichunk, beam, channel, ...]: the channel range of one connection in one minichunk
    # is one contiguous piece per beam
    size_d, size_s = 2 * nmc * nb * nch * 128, 2 * nmc * nb * nch * 2 * 2
    shm_d, shm_s = shared_memory.SharedMemory(create=True, size=size_d), shared_memory.SharedMemory(create=True, size=size_s)
    data = np.ndarray((2, nmc, nb, nch, 128), np.uint8, buffer=shm_d.buf)
    so = np.ndarray((2, nmc, nb, nch, 2), np.float16, buffer=shm_s.buf)
    ctx = mp.get_context("spawn")
    P = max(1, min(c["processes"], nb))
    tasks = [ctx.Queue() for _ in range(P)]; done = ctx.Queue()
    procs = [ctx.Process(target=prepare, daemon=True,
                         args=(w, list(range(w, nb, P)), c, segments, n_data, total, shm_d.name, shm_s.name, tasks[w], done))
             for w in range(P)]
    for p in procs:
        p.start()

    # connections: each one sends a contiguous channel range (channel 0 = lowest frequency)
    nconn = len(c["receivers"]) * c["connections"]
    cuts = np.linspace(0, nch, nconn + 1).astype(int)

    def request_chunk(ichunk):
        for q in tasks:
            q.put((ichunk, ichunk % 2))

    def wait_chunk(ichunk):
        n = 0
        while n < P:
            try:
                w_, ic = done.get(timeout=5)
            except queue.Empty:
                dead = [w for w, p in enumerate(procs) if not p.is_alive()]
                if dead:
                    raise SystemExit(f"ERROR: preparation processes {dead} died (see the error above)")
                continue
            assert ic == ichunk, (ic, ichunk)
            n += 1

    # the first block is prepared BEFORE fixing the time of seq 0: otherwise the data is timestamped before it is
    # sent (cold reading of the first .fil files takes several seconds) and the latency measured by the search inflates
    t_a = time.monotonic()
    request_chunk(0)
    wait_chunk(0)
    info(f"first block ready in {time.monotonic() - t_a:.1f} s")
    unix0 = time.time_ns()
    t0 = [time.monotonic()]                          # real-time pacing clock: seq 0 goes out at unix0
    conns = []
    for k in range(nconn):
        ch0, ch1 = int(cuts[k]), int(cuts[k + 1])
        meta = metadata_yaml([nch], c["band"], list(range(nb)), list(range(ch0, ch1)), unix0, DT_NS_PER_SEQ, seq_per_sample)
        conns.append((Connection(c["receivers"][k % len(c["receivers"])], meta, ack=c["pacing"] == "ack"), ch0, ch1))
    info(f"{nconn} connections open to {', '.join(c['receivers'])}")

    errors = []

    def send(k, ichunk, slot):
        con, ch0, ch1 = conns[k]
        try:
            for m in range(nmc):
                g = ichunk * nmc + m
                if c["pacing"] == "real-time":
                    wait = t0[0] + g * (MINICHUNK_SAMPLES * c["sample_ms"] / 1000) / c["speed"] - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                seq = (ichunk * T + m * MINICHUNK_SAMPLES) * seq_per_sample
                con.send_minichunk(seq, so[slot, m, :, ch0:ch1], data[slot, m, :, ch0:ch1])
        except Exception as ex:                     # noqa: BLE001
            errors.append(f"{con.address}: {ex!r}")

    t_start = time.monotonic()
    t_prep = t_send = 0.0                            # accumulated waiting for preparation and sending time
    try:
        for ichunk in range(nchunks):
            ta = time.monotonic()
            if ichunk > 0:
                wait_chunk(ichunk)
            tb = time.monotonic(); t_prep += tb - ta
            if ichunk + 1 < nchunks:
                request_chunk(ichunk + 1)           # the next one is prepared while this one is sent
            threads = [threading.Thread(target=send, args=(k, ichunk, ichunk % 2)) for k in range(nconn)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            t_send += time.monotonic() - tb
            if errors:
                raise RuntimeError("; ".join(errors[:3]))
            if (ichunk + 1) % 10 == 0 or ichunk + 1 == nchunks:
                dt = time.monotonic() - t_start
                info(f"block {ichunk + 1}/{nchunks}, {dt:.0f} s, rate {(ichunk + 1) * T / 1000 / dt:.2f}× real time; "
                     f"per block: waiting for preparation {t_prep / (ichunk + 1):.2f} s, sending {t_send / (ichunk + 1):.2f} s")
    finally:
        for q in tasks:
            q.put(None)
        for con, _, _ in conns:
            con.close()
        dt = time.monotonic() - t_start
        summary = dict(nbeams=nb, beams_with_data=n_data, nchan=nch, T=T, total_samples=total, nchunks=nchunks,
                       files=len(assignment), prepad=int(c["prepad_s"] * 1000), tail=int(c["tail_s"] * 1000),
                       dt_ns_per_seq=DT_NS_PER_SEQ, seq_per_frb_time_sample=seq_per_sample, zone_freq_edges=c["band"],
                       receivers=c["receivers"], connections=nconn, quantization=c["quantization"],
                       scale_sigma=0.5, offset_sigma=0.0 if c["quantization"] == "per_channel" else 0.125,
                       site_rfi=c["rfi"] if c["rfi"].get("enabled") else None, mask=c["mask"], pacing=c["pacing"],
                       speed=c["speed"], duration_s=dt, measured_rate=nchunks * T / 1000 / max(dt, 1e-9),
                       minichunks_dropped_by_pirate=sum(x[0].dropped for x in conns) if c["pacing"] == "ack" else None,
                       start_unix=unix0 / 1e9, errors=errors)
        json.dump(summary, open(a.out / "emitter.json", "w"), indent=1)
        info(f"end: {nchunks} blocks in {dt:.0f} s ({summary['measured_rate']:.2f}× real time)")
        shm_d.close(); shm_s.close(); shm_d.unlink(); shm_s.unlink()


if __name__ == "__main__":
    main()
