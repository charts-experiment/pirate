#!/usr/bin/env python
"""Testbench grouper, version 0: pirate's toy grouper + saving windows of the DM-time map.

Behaviour towards the sifter: IDENTICAL to pirate_frb/run_toy_grouper.py (one event per chunk and per beam with
the maximum S/N over trees, DM and time, if it exceeds the threshold). This way the evaluation of the 64-beam test
(evaluate_test_64b.py) remains valid.

What is new: for each event with S/N >= --save-threshold, one chunk later, the map window of the event's tree is
saved at native resolution (no resizing):
  - out_max: DM [idm-128, idm+128) x time [previous chunk, chunk of the event, next chunk]
    -> (256, 3*nt), float16; outside the map it is filled with 0 (and the valid range is recorded);
  - out_argmax: 32 x 32 patch of tokens centred on the peak (chunk of the event), uint32.
One .npz per chunk is saved with all the events of that chunk, in a separate thread (does not slow pirate down).

Copies on the GPU (to have the neighbouring chunks): 3 chunks of out_max of all beams per tree
(tree 0, 64 beams: 3 x 64 MB) and the tokens of the current chunk (64 MB).

Usage (pirate environment), like toy_grouper:
  python grouper_testbench.py 127.0.0.1:7000 -s 127.0.0.1:7500 -t 5 --out DIR [--save-threshold 5]
"""
import argparse
import contextlib
import itertools
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

VDM, PATCH = 128, 16          # half window in DM; half patch of tokens


def writer(wqueue, out):
    """Thread that writes the .npz files (one per chunk with events)."""
    while True:
        item = wqueue.get()
        if item is None:
            return
        name, data = item
        np.savez(out / name, **data)


def run(grouper, sifter, threshold, save_threshold, out):
    import cupy as cp
    from pirate_frb.utils import atomic_print

    nbeams = grouper.total_beams
    nbatches = grouper.nbatches
    bpb = nbeams // nbatches
    beam_set_id = grouper.xengine_metadata.beamset
    if sifter is not None:
        sifter.send_configuration(
            pirate_yaml=grouper.dedispersion_config_yaml_string,
            xengine_yaml=grouper.xengine_metadata_yaml_string,
            dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
            grouper_yaml={"toy_grouper": True, "grouper_testbench": "v0"},
            search_ip_addr=grouper.search_ip_addr)

    out.mkdir(parents=True, exist_ok=True)
    wqueue = queue.Queue(maxsize=64)
    wthread = threading.Thread(target=writer, args=(wqueue, out), daemon=True)
    wthread.start()

    ring = None              # per tree: (3, nbeams, ndm, nt) float32
    tokens = None            # per tree: (nbeams, ndm, nt) uint32, current chunk
    fpga_start = {}          # ichunk -> chunk_fpga_start
    pending = None           # events of the previous chunk, waiting for the next chunk
    meta_info = {"nbeams": nbeams, "nbatches": nbatches, "VDM": VDM, "PATCH": PATCH,
                 "full_steady_ichunk": int(grouper.full_steady_ichunk)}

    def save(pend, ichunk_ev):
        """Builds the windows of the events of chunk ichunk_ev (uses chunks ichunk_ev-1, ichunk_ev, ichunk_ev+1)."""
        if pend is None or not len(pend["snr"]):
            return
        windows, valid = [], []
        for k in range(len(pend["snr"])):
            it, b, idm = int(pend["tree"][k]), int(pend["ibeam"][k]), int(pend["idm"][k])
            rg = ring[it]
            ndm, nt = rg.shape[2], rg.shape[3]
            order = [(ichunk_ev + d) % 3 for d in (-1, 0, 1)]
            available = [ichunk_ev - 1 >= 0, True, (ichunk_ev + 1) in fpga_start]
            lo, hi = max(0, idm - VDM), min(ndm, idm + VDM)
            v = cp.zeros((2 * VDM, 3 * nt), dtype=cp.float16)
            for j, (slot, ok) in enumerate(zip(order, available)):
                if ok:
                    v[lo - (idm - VDM): hi - (idm - VDM), j * nt:(j + 1) * nt] = rg[slot, b, lo:hi, :].astype(cp.float16)
            windows.append(v)
            valid.append((lo - (idm - VDM), hi - (idm - VDM), int(available[0]), int(available[2])))
        data = {k: v for k, v in pend.items() if k != "tok"}
        data["window"] = cp.asnumpy(cp.stack(windows))
        data["tokens"] = pend["tok"]
        data["valid"] = np.array(valid, dtype=np.int32)    # first row, last row, previous chunk exists, next chunk exists
        data["chunk_fpga_start"] = np.int64(fpga_start[ichunk_ev])
        data["meta"] = np.array(json.dumps(meta_info))
        wqueue.put((f"chunk_{ichunk_ev:06d}.npz", data))

    ichunk = -1
    try:
        for ichunk in itertools.count():
            t0 = time.time()
            per_max = cp.full((nbeams,), -cp.inf, dtype=cp.float32)
            per_tree = cp.full((nbeams,), -1, dtype=cp.int64)
            per_idm = cp.full((nbeams,), -1, dtype=cp.int64)
            per_it = cp.full((nbeams,), -1, dtype=cp.int64)
            per_tok = cp.zeros((nbeams,), dtype=cp.uint32)

            for ibatch in range(nbatches):
                beam0 = ibatch * bpb
                with grouper.get_output(ichunk, ibatch) as outputs:
                    if ring is None:
                        ring = [cp.zeros((3, nbeams) + t.shape[1:], dtype=cp.float32) for t in outputs.out_max]
                        tokens = [cp.zeros((nbeams,) + t.shape[1:], dtype=cp.uint32) for t in outputs.out_argmax]
                        meta_info["shapes"] = [list(t.shape[1:]) for t in outputs.out_max]
                    for itree, tree_out in enumerate(outputs.out_max):
                        sl = slice(beam0, beam0 + bpb)
                        # copies for the windows (inside the context manager: afterwards the memory is reused)
                        ring[itree][ichunk % 3, sl] = tree_out
                        tokens[itree][sl] = outputs.out_argmax[itree]
                        # the same as the toy grouper
                        b, ndm, nt = tree_out.shape
                        flat = tree_out.reshape(b, ndm * nt)
                        bmax, barg = flat.max(axis=1), flat.argmax(axis=1)
                        btok = outputs.out_argmax[itree].reshape(b, ndm * nt)[cp.arange(b), barg]
                        upd = bmax > per_max[sl]
                        per_max[sl] = cp.where(upd, bmax, per_max[sl])
                        per_tree[sl] = cp.where(upd, itree, per_tree[sl])
                        per_idm[sl] = cp.where(upd, barg // nt, per_idm[sl])
                        per_it[sl] = cp.where(upd, barg % nt, per_it[sl])
                        per_tok[sl] = cp.where(upd, btok, per_tok[sl])

            ibeam = cp.nonzero(per_max > threshold)[0]
            events = grouper.create_events(ichunk, per_tree[ibeam], ibeam, per_idm[ibeam], per_it[ibeam],
                                           per_max[ibeam], per_tok[ibeam])
            fpga_start[ichunk] = int(events.chunk_fpga_start)
            for old in [c for c in fpga_start if c < ichunk - 3]:
                del fpga_start[old]
            if sifter is not None:
                sifter.send_events(beam_set_id=beam_set_id, events=events, coarsegrain_snr=per_max)

            # events of the previous chunk: the next chunk is now in the ring
            save(pending, ichunk - 1)

            # events of this chunk, to be saved with the next chunk
            sel = cp.nonzero(per_max > save_threshold)[0]
            pending = None
            if len(sel):
                ev = grouper.create_events(ichunk, per_tree[sel], sel, per_idm[sel], per_it[sel],
                                           per_max[sel], per_tok[sel])
                tr, bm, idm, itm = (cp.asnumpy(a[sel]) for a in (per_tree, cp.arange(nbeams), per_idm, per_it))
                patches = []
                for k in range(len(sel)):
                    tk = tokens[int(tr[k])]
                    ndm, nt = tk.shape[1], tk.shape[2]
                    p = cp.zeros((2 * PATCH, 2 * PATCH), dtype=cp.uint32)
                    a0, a1 = max(0, idm[k] - PATCH), min(ndm, idm[k] + PATCH)
                    b0, b1 = max(0, itm[k] - PATCH), min(nt, itm[k] + PATCH)
                    p[a0 - (idm[k] - PATCH):a1 - (idm[k] - PATCH), b0 - (itm[k] - PATCH):b1 - (itm[k] - PATCH)] = tk[bm[k], a0:a1, b0:b1]
                    patches.append(p)
                pending = {
                    "beam": ev.beam_ids, "fpga_timestamp": ev.fpga_timestamps, "dm": ev.dms, "snr": ev.snrs,
                    "width_ms": ev.widths_ms, "tree": tr.astype(np.int32), "ibeam": bm, "idm": idm, "itime": itm,
                    "ichunk": np.full(len(sel), ichunk, dtype=np.int64),
                    "steady": np.full(len(sel), ichunk >= grouper.full_steady_ichunk),
                    "tok": cp.asnumpy(cp.stack(patches)),
                }
            atomic_print(f"grouper_testbench: ichunk={ichunk}, nevents={len(events)}, "
                         f"saved={0 if pending is None else len(pending['snr'])}, "
                         f"coarse_snr_max={float(per_max.max()):.4g}, t={1e3 * (time.time() - t0):.0f} ms")
    finally:
        # when pirate disconnects: save the events of the last chunk (no next chunk) and empty the queue
        if ring is not None and pending is not None:
            save(pending, ichunk)
        wqueue.put(None)
        wthread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grouper_addr")
    ap.add_argument("-s", "--sifter", default=None)
    ap.add_argument("-t", "--threshold", type=float, default=10.0)
    ap.add_argument("--save-threshold", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from pirate_frb.rpc import FrbGrouper, FrbSifterClient
    from pirate_frb.utils import atomic_print
    sifter_cm = FrbSifterClient(a.sifter) if a.sifter else contextlib.nullcontext()
    with sifter_cm as sifter, FrbGrouper(a.grouper_addr, restore_cuda_device=False) as grouper:
        try:
            run(grouper, sifter, a.threshold, a.save_threshold, a.out)
        except KeyboardInterrupt:
            atomic_print("grouper_testbench: interrupted")
        except RuntimeError as e:
            if grouper.is_stopped:
                atomic_print(f"grouper_testbench: pirate disconnected ({e}); exiting")
            else:
                raise


if __name__ == "__main__":
    main()
