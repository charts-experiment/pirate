#!/usr/bin/env python
"""Testbench grouper, version 1: real peak search in pirate's DM-time map.

Per chunk, per batch of beams and per tree (all on the GPU except the final grouping, which is small):
  1. Regions: cells of out_max with S/N >= --cell-threshold, joined by 8-neighbourhood (beams are not joined).
     The maximum cell of each region is taken as a candidate (cupyx.scipy.ndimage.label + lexsort).
  2. The --max-cand highest-S/N candidates per beam are kept and decoded with create_events
     (exact DM, arrival at the bottom edge, width).
  3. Grouping per burst: in pirate's map a burst is a long diagonal, and all its cells share the arrival at the
     top frequency, t_top = t_event - K*DM (K: delay per unit DM between the bottom edge and 501.6 MHz). Within
     each beam, from highest to lowest S/N, a candidate is discarded if an already accepted event (from this chunk
     or the previous one, from any tree) has |t_top - t_top'| <= --tol-t and an equal or higher S/N. This way a
     diagonal split into several regions becomes a single event; if the real peak arrives in a later chunk than a
     weak piece already sent, it is sent anyway (a duplicate may remain, but with the correct DM).
  4. Up to --k events per beam with S/N >= --threshold are sent to the sifter (at most --k-total per chunk).
     With --log-all FILE, ALL grouped events with S/N >= --threshold are also written, without the --k and
     --k-total caps, in the sifter.log format (to count everything detected without saturating the sifter).

For a like-for-like comparison, the result of grouper v0 (one maximum per chunk and beam, threshold --threshold)
is also computed on the SAME map and written to events_v0.log in the same format as the toy_sifter's sifter.log.
Map windows are saved as in v0, for the v1 events.

Usage (pirate environment):
  python grouper_v1.py 127.0.0.1:7000 -s 127.0.0.1:7500 -t 5 --out DIR
"""
import argparse
import contextlib
import itertools
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouper_testbench import VDM, PATCH, writer    # noqa: E402

SEQ = 200                     # fpga counts per sample (1 ms)


def delay_edge_to_top(cfg_yaml):
    """Seconds per unit DM between the bottom edge of the band and 501.6 MHz."""
    import pirate_frb as pf
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(cfg_yaml)
    cfg = pf.DedispersionConfig.from_yaml(f.name)
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    s_per_dm_band = 1e-3 / cfg.dm_per_unit_delay()
    return s_per_dm_band * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2)


def sifter_line(ev, i):
    return (f"    beam_id={int(ev.beam_ids[i])}, dm={float(ev.dms[i]):.4g}, "
            f"fpga_timestamp={int(ev.fpga_timestamps[i])}, width={float(ev.widths_ms[i]):.4g} ms, "
            f"subband=[{float(ev.subband_freqs_lo_MHz[i]):.1f},{float(ev.subband_freqs_hi_MHz[i]):.1f}] MHz, "
            f"snr={float(ev.snrs[i]):.4g}, tree_index={int(ev.tree_index[i])}")


class Clock:
    """Time per stage within a chunk. With profile=True it synchronizes the GPU at each mark (exact measurement)."""

    STAGES = ("wait_pirate", "copies", "v0", "regions", "v0_decode", "selection", "v1_decode", "grouping",
              "send", "save", "queue_wait", "patches")

    def __init__(self, profile, path):
        import cupy as cp
        self.cp, self.profile = cp, profile
        self.f = open(path, "w") if profile else None
        if self.f:
            self.f.write("ichunk,total_ms," + ",".join(e + "_ms" for e in self.STAGES) + ",n_cand,n_v1\n")

    def new(self):
        self.acc = dict.fromkeys(self.STAGES, 0.0)
        self.t0 = self.t = time.perf_counter()

    def mark(self, stage):
        if self.profile:
            self.cp.cuda.get_current_stream().synchronize()
        now = time.perf_counter()
        self.acc[stage] += now - self.t
        self.t = now

    def end(self, ichunk, n_cand, n_v1):
        total = time.perf_counter() - self.t0
        if self.f:
            self.f.write(f"{ichunk},{1e3 * total:.2f}," + ",".join(f"{1e3 * self.acc[e]:.2f}" for e in self.STAGES)
                         + f",{n_cand},{n_v1}\n")
            self.f.flush()
        return total


def run(grouper, sifter, a):
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    from pirate_frb.rpc import FrbSifterEvents
    from pirate_frb.utils import atomic_print

    nbeams, nbatches = grouper.total_beams, grouper.nbatches
    bpb = nbeams // nbatches
    beam_set_id = grouper.xengine_metadata.beamset
    K = delay_edge_to_top(grouper.dedispersion_config_yaml_string)
    if sifter is not None:
        sifter.send_configuration(
            pirate_yaml=grouper.dedispersion_config_yaml_string,
            xengine_yaml=grouper.xengine_metadata_yaml_string,
            dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
            grouper_yaml={"grouper_testbench": "v1", "cell_threshold": a.cell_threshold, "tol_t": a.tol_t, "k": a.k},
            search_ip_addr=grouper.search_ip_addr)

    a.out.mkdir(parents=True, exist_ok=True)
    log_all = open(a.log_all, "w") if a.log_all else None
    if log_all:
        log_all.write("grouper v1 events without the --k/--k-total caps (all grouped events with S/N > threshold)\n")
    log_v0 = open(a.out.parent / "events_v0.log", "w")
    log_v0.write("grouper v0 events (maximum per chunk and beam) computed on the same map as v1\n")
    wqueue = queue.Queue(maxsize=64)
    wthread = threading.Thread(target=writer, args=(wqueue, a.out), daemon=True)
    wthread.start()
    structure = np.zeros((3, 3, 3), bool)
    structure[1] = True                           # 8-neighbourhood in (DM, time), beams not joined
    structure = cp.asarray(structure)

    ring = tokens = None
    fpga_start, pending = {}, None
    previous = {}                                 # beam -> list of (t_top, snr) accepted in the previous chunk
    clock = Clock(a.profile, a.out.parent / "grouper_timings.csv")
    meta_info = {"nbeams": nbeams, "nbatches": nbatches, "VDM": VDM, "PATCH": PATCH, "version": "v1",
                 "full_steady_ichunk": int(grouper.full_steady_ichunk), "args": {k: str(v) for k, v in vars(a).items()}}

    def save(pend, ichunk_ev):
        if pend is None or not len(pend["snr"]):
            return
        windows, valid = [], []
        for k in range(len(pend["snr"])):
            it, b, idm = int(pend["tree"][k]), int(pend["ibeam"][k]), int(pend["idm"][k])
            rg = ring[it]
            ndm, nt = rg.shape[2], rg.shape[3]
            order = [(ichunk_ev + d) % 3 for d in (-1, 0, 1)]
            avail = [ichunk_ev - 1 >= 0, True, (ichunk_ev + 1) in fpga_start]
            lo, hi = max(0, idm - VDM), min(ndm, idm + VDM)
            v = cp.zeros((2 * VDM, 3 * nt), dtype=cp.float16)
            for j, (slot, ok) in enumerate(zip(order, avail)):
                if ok:
                    v[lo - (idm - VDM): hi - (idm - VDM), j * nt:(j + 1) * nt] = rg[slot, b, lo:hi, :].astype(cp.float16)
            windows.append(v)
            valid.append((lo - (idm - VDM), hi - (idm - VDM), int(avail[0]), int(avail[2])))
        data = {k: v for k, v in pend.items() if k != "tok"}
        data["window"] = cp.asnumpy(cp.stack(windows))
        data["tokens"] = pend["tok"]
        data["valid"] = np.array(valid, dtype=np.int32)
        data["chunk_fpga_start"] = np.int64(fpga_start[ichunk_ev])
        data["meta"] = np.array(json.dumps(meta_info))
        clock.mark("save")
        wqueue.put((f"chunk_{ichunk_ev:06d}.npz", data))
        clock.mark("queue_wait")

    ichunk = -1
    try:
        for ichunk in itertools.count():
            clock.new()
            # v0 on the same map
            v0_max = cp.full((nbeams,), -cp.inf, dtype=cp.float32)
            v0_tree = cp.full((nbeams,), -1, dtype=cp.int64)
            v0_idm = cp.full((nbeams,), -1, dtype=cp.int64)
            v0_it = cp.full((nbeams,), -1, dtype=cp.int64)
            v0_tok = cp.zeros((nbeams,), dtype=cp.uint32)
            cand = []                             # per (batch, tree): (beam, tree, idm, itime, snr, token)

            for ibatch in range(nbatches):
                beam0 = ibatch * bpb
                sl = slice(beam0, beam0 + bpb)
                clock.mark("copies")                 # (whatever is left from the previous iteration)
                with grouper.get_output(ichunk, ibatch) as outputs:
                    clock.mark("wait_pirate")
                    if ring is None:
                        ring = [cp.zeros((3, nbeams) + t.shape[1:], dtype=cp.float32) for t in outputs.out_max]
                        tokens = [cp.zeros((nbeams,) + t.shape[1:], dtype=cp.uint32) for t in outputs.out_argmax]
                        meta_info["shapes"] = [list(t.shape[1:]) for t in outputs.out_max]
                    for itree, tree_out in enumerate(outputs.out_max):
                        tok = outputs.out_argmax[itree]
                        if not a.no_windows:
                            ring[itree][ichunk % 3, sl] = tree_out
                            tokens[itree][sl] = tok
                        clock.mark("copies")
                        b, ndm, nt = tree_out.shape
                        # --- v0 ---
                        flat = tree_out.reshape(b, ndm * nt)
                        bmax, barg = flat.max(axis=1), flat.argmax(axis=1)
                        btok = tok.reshape(b, ndm * nt)[cp.arange(b), barg]
                        upd = bmax > v0_max[sl]
                        v0_max[sl] = cp.where(upd, bmax, v0_max[sl])
                        v0_tree[sl] = cp.where(upd, itree, v0_tree[sl])
                        v0_idm[sl] = cp.where(upd, barg // nt, v0_idm[sl])
                        v0_it[sl] = cp.where(upd, barg % nt, v0_it[sl])
                        v0_tok[sl] = cp.where(upd, btok, v0_tok[sl])
                        clock.mark("v0")
                        # --- v1: maximum of each connected region above the threshold ---
                        mask = tree_out >= a.cell_threshold
                        lab, nreg = ndi.label(mask, structure=structure)
                        if int(nreg) == 0:
                            clock.mark("regions")
                            continue
                        idx = cp.flatnonzero(mask.ravel())
                        labs = lab.ravel()[idx]
                        vals = tree_out.ravel()[idx]
                        order = cp.lexsort(cp.stack((-vals, labs)))
                        labs_o = labs[order]
                        first = cp.ones(labs_o.shape, dtype=bool)
                        first[1:] = labs_o[1:] != labs_o[:-1]
                        g = idx[order[first]]                         # flat index of the maximum cell of each region
                        bb, rem = g // (ndm * nt), g % (ndm * nt)
                        cand.append((bb + beam0, cp.full(g.shape, itree), rem // nt, rem % nt,
                                     tree_out.ravel()[g], tok.ravel()[g]))
                        clock.mark("regions")

            # v0 events (to the comparison log)
            ib0 = cp.nonzero(v0_max > a.threshold)[0]
            ev0 = grouper.create_events(ichunk, v0_tree[ib0], ib0, v0_idm[ib0], v0_it[ib0], v0_max[ib0], v0_tok[ib0])
            fpga_start[ichunk] = int(ev0.chunk_fpga_start)
            for old in [c for c in fpga_start if c < ichunk - 3]:
                del fpga_start[old]
            lines = [f"toy_sifter beamset={beam_set_id} fpga=[{ev0.chunk_fpga_start}:{ev0.chunk_fpga_end}] "
                     f"coarse_snr_max={float(v0_max.max()):.4g}"]
            lines += [sifter_line(ev0, i) for i in range(len(ev0))]
            log_v0.write("\n".join(lines) + "\n")
            clock.mark("v0_decode")
            n_cand = 0

            # v1 candidates: the max_cand highest-S/N per beam
            if cand:
                cb, ct, cd, ci, cs, ck = (cp.concatenate([c[j] for c in cand]) for j in range(6))
                o = cp.lexsort(cp.stack((-cs, cb)))
                cb, ct, cd, ci, cs, ck = cb[o], ct[o], cd[o], ci[o], cs[o], ck[o]
                start = cp.ones(cb.shape, dtype=bool)
                start[1:] = cb[1:] != cb[:-1]
                group = cp.cumsum(start) - 1                      # index of the beam in the ordering
                rank = cp.arange(cb.size) - cp.flatnonzero(start)[group]     # position within its beam
                sel = rank < a.max_cand
                cb, ct, cd, ci, cs, ck = cb[sel], ct[sel], cd[sel], ci[sel], cs[sel], ck[sel]
                n_cand = int(cb.size)
                clock.mark("selection")
                evc = grouper.create_events(ichunk, ct, cb, cd, ci, cs, ck)
                hb, ht, hd, hi = (cp.asnumpy(x) for x in (cb, ct, cd, ci))
                snr = np.asarray(evc.snrs)
                t_top = np.asarray(evc.fpga_timestamps) / SEQ * 1e-3 - np.asarray(evc.dms) * K
                clock.mark("v1_decode")
                # grouping per burst (t_top) within each beam, against what was already accepted
                accepted = []
                everything = []
                new_previous = {}
                for b in np.unique(hb):
                    ii = np.flatnonzero(hb == b)
                    ii = ii[np.argsort(-snr[ii])]
                    seen = list(previous.get(int(b), []))
                    nb = 0
                    for i in ii:
                        # discarded only if there is already one from the same burst at least as strong; if the new
                        # one is stronger (e.g. the real peak after a weak piece of the diagonal in the previous chunk), it goes anyway
                        if any(abs(t_top[i] - tp) <= a.tol_t and sp >= snr[i] for tp, sp in seen):
                            continue
                        seen.append((t_top[i], snr[i]))
                        new_previous.setdefault(int(b), []).append((t_top[i], snr[i]))
                        if snr[i] > a.threshold:
                            everything.append(i)
                        if snr[i] > a.threshold and nb < a.k:
                            accepted.append(i)
                            nb += 1
                previous = new_previous
                accepted = np.array(sorted(accepted, key=lambda i: -snr[i])[:a.k_total], dtype=int)
                if log_all and everything:
                    log_all.write(f"toy_sifter beamset={beam_set_id} fpga=[{ev0.chunk_fpga_start}:{ev0.chunk_fpga_end}] "
                                  f"n_all={len(everything)}\n" + "\n".join(sifter_line(evc, i) for i in everything) + "\n")
                    log_all.flush()                        # nothing written is lost if the grouper is killed with SIGKILL
            else:
                evc, accepted, previous = None, np.zeros(0, int), {}

            clock.mark("grouping")
            if len(accepted):
                ev1 = FrbSifterEvents(
                    beam_ids=np.asarray(evc.beam_ids)[accepted], fpga_timestamps=np.asarray(evc.fpga_timestamps)[accepted],
                    dms=np.asarray(evc.dms)[accepted], snrs=snr[accepted], rfi_probs=np.zeros(len(accepted)),
                    widths_ms=np.asarray(evc.widths_ms)[accepted],
                    subband_freqs_lo_MHz=np.asarray(evc.subband_freqs_lo_MHz)[accepted],
                    subband_freqs_hi_MHz=np.asarray(evc.subband_freqs_hi_MHz)[accepted],
                    tree_index=np.asarray(evc.tree_index)[accepted],
                    chunk_fpga_start=ev0.chunk_fpga_start, chunk_fpga_end=ev0.chunk_fpga_end)
            else:
                ev1 = FrbSifterEvents(*(np.zeros(0) for _ in range(9)),
                                      chunk_fpga_start=ev0.chunk_fpga_start, chunk_fpga_end=ev0.chunk_fpga_end)
            if sifter is not None:
                sifter.send_events(beam_set_id=beam_set_id, events=ev1, coarsegrain_snr=v0_max)
            clock.mark("send")

            if not a.no_windows:
                save(pending, ichunk - 1)
            pending = None
            if len(accepted) and not a.no_windows:
                patches = []
                for i in accepted:
                    tk = tokens[int(ht[i])]
                    ndm, nt = tk.shape[1], tk.shape[2]
                    p = cp.zeros((2 * PATCH, 2 * PATCH), dtype=cp.uint32)
                    a0, a1 = max(0, hd[i] - PATCH), min(ndm, hd[i] + PATCH)
                    b0, b1 = max(0, hi[i] - PATCH), min(nt, hi[i] + PATCH)
                    p[a0 - (hd[i] - PATCH):a1 - (hd[i] - PATCH), b0 - (hi[i] - PATCH):b1 - (hi[i] - PATCH)] = \
                        tk[hb[i], a0:a1, b0:b1]
                    patches.append(p)
                pending = {
                    "beam": np.asarray(evc.beam_ids)[accepted], "fpga_timestamp": np.asarray(evc.fpga_timestamps)[accepted],
                    "dm": np.asarray(evc.dms)[accepted], "snr": snr[accepted], "width_ms": np.asarray(evc.widths_ms)[accepted],
                    "tree": ht[accepted].astype(np.int32), "ibeam": hb[accepted], "idm": hd[accepted], "itime": hi[accepted],
                    "ichunk": np.full(len(accepted), ichunk, dtype=np.int64),
                    "steady": np.full(len(accepted), ichunk >= grouper.full_steady_ichunk),
                    "tok": cp.asnumpy(cp.stack(patches)),
                }
            clock.mark("patches")
            total = clock.end(ichunk, n_cand, len(accepted))
            atomic_print(f"grouper_v1: ichunk={ichunk}, v0={len(ev0)}, v1={len(accepted)}, "
                         f"t={1e3 * total:.0f} ms")
    finally:
        if ring is not None and pending is not None:
            save(pending, ichunk)
        log_v0.close()
        if log_all:
            log_all.close()
        wqueue.put(None)
        wthread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grouper_addr")
    ap.add_argument("-s", "--sifter", default=None)
    ap.add_argument("-t", "--threshold", type=float, default=5.0, help="minimum S/N of an event sent to the sifter")
    ap.add_argument("--cell-threshold", type=float, default=5.0, help="minimum S/N of the cells that form regions")
    ap.add_argument("--tol-t", type=float, default=0.1, help="t_top tolerance (s) to join candidates of the same burst")
    ap.add_argument("--max-cand", type=int, default=32, help="candidates per beam and chunk that are decoded")
    ap.add_argument("--k", type=int, default=4, help="events per beam and chunk")
    ap.add_argument("--k-total", type=int, default=256, help="events per chunk")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-all", type=Path, default=None,
                    help="write all grouped events (without the --k/--k-total caps) to this file")
    ap.add_argument("--profile", action="store_true", help="time per stage in grouper_timings.csv (synchronizes the GPU)")
    ap.add_argument("--no-windows", action="store_true", help="do not save map windows (operation cost)")
    a = ap.parse_args()

    from pirate_frb.rpc import FrbGrouper, FrbSifterClient
    from pirate_frb.utils import atomic_print
    sifter_cm = FrbSifterClient(a.sifter) if a.sifter else contextlib.nullcontext()
    with sifter_cm as sifter, FrbGrouper(a.grouper_addr, restore_cuda_device=False) as grouper:
        try:
            run(grouper, sifter, a)
        except KeyboardInterrupt:
            atomic_print("grouper_v1: interrupted")
        except RuntimeError as e:
            if grouper.is_stopped:
                atomic_print(f"grouper_v1: pirate disconnected ({e}); exiting")
            else:
                raise


if __name__ == "__main__":
    main()
