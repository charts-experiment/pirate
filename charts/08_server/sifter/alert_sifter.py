#!/usr/bin/env python3
"""Alert sifter: last stage of the chain (replaces pirate's toy_sifter).

It is a gRPC server with pirate's FrbSifter service (`frb_sifter.proto`): once per 2.048 s block, the grouper
sends it the events of all beams. It does two things:

  1. Logs all events to standard output in the same format as the toy_sifter (the launcher saves it to
     sifter.log and the evaluation reads it as before).
  2. Decides which events are an FRB ALERT and writes them to <out>/alerts.jsonl (one JSON line per alert)
     and to standard output ("FRB ALERT ..."). Optionally runs a command per alert (--action), with the JSON
     on standard input: notify, dump data, send a VOEvent, etc.

Alert rule (all configurable):
  - S/N >= --threshold (7.6 by default: ~10 false alarms per hour for the 64-beam system, with the site RFI
    and mask m1; see docs/RESULTS_64beams.md)
  - --dm-min <= DM <= --dm-max (terrestrial RFI has DM ~ 0)
  - RFI veto: if more than --max-beams beams have events with the same arrival (± --window-s, at 501.6 MHz),
    it is discarded (a sky signal falls in one or a few neighbouring beams; RFI enters all of them)
  - events with the same arrival (± --window-s) in consecutive blocks and in coincident beams form a group; the
    group is emitted when --wait-blocks blocks pass without its S/N improving, with its best event (a bright
    pulse appears earlier, with lower S/N, at lower DMs). If it later improves by > 20 %, an "update" is emitted

Times: the UTC time comes from the X-engine's unix_ns_at_seq_0 (sent by the emitter). `utc_arrival` is the arrival
at 501.6 MHz; `latency_s` is the time from when the pulse finished arriving (bottom edge of the band) until the
alert was emitted.

Needs pirate's environment (grpc and pirate_frb). Usage:
  python 08_server/sifter/alert_sifter.py 127.0.0.1:7500 --out results/x [--threshold 7.6] [--action CMD]
"""
import argparse
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path

import grpc
import yaml
from pirate_frb.rpc.grpc import frb_sifter_pb2, frb_sifter_pb2_grpc

F_TOP_MHZ = 501.6


def delay_edge_to_top(dd_yaml):
    """Seconds per unit DM between the bottom edge of the band (where pirate timestamps the event) and 501.6 MHz."""
    import pirate_frb as pf
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(dd_yaml)
    cfg = pf.DedispersionConfig.from_yaml(f.name)
    Path(f.name).unlink()
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    return (1e-3 / cfg.dm_per_unit_delay()) * (f_lo ** -2 - F_TOP_MHZ ** -2) / (f_lo ** -2 - f_hi ** -2)


def utc(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def say(text):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def run_action(command, alert):
    """Runs the command with the alert (JSON) on standard input; an error does not stop the sifter."""
    try:
        r = subprocess.run(command, shell=True, input=json.dumps(alert), text=True, capture_output=True, timeout=60)
        if r.returncode:
            say(f"action: rc={r.returncode} {r.stderr.strip()[:200]}")
    except Exception as e:                                # noqa: BLE001
        say(f"action: failed ({e})")


class Group:
    """Events with the same arrival at 501.6 MHz (± window) in one beam or in coincident beams."""

    def __init__(self, ev, t, beams, block):
        self.best, self.t, self.beams, self.best_block = ev, t, set(beams), block
        self.emitted = None                           # alert id, "veto", or None while it is open
        self.emitted_snr = 0.0
        self.vetoed = False
        self.n = 0


class AlertSifter(frb_sifter_pb2_grpc.FrbSifterServicer):
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.k_top, self.unix0, self.dt_ns = 0.0, 0, 5000
        self.groups = []
        self.n_alerts = self.n_vetoed = 0
        a.out.mkdir(parents=True, exist_ok=True)
        self.file = open(a.out / "alerts.jsonl", "w")

    def CheckConfiguration(self, request, context):
        if request.protocol_version != frb_sifter_pb2.PROTOCOL_VERSION_CURRENT:
            msg = f"protocol version {request.protocol_version}, {frb_sifter_pb2.PROTOCOL_VERSION_CURRENT} required"
            say(f"CheckConfiguration REJECTED ({msg})")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, msg)
        with self.lock:
            if request.pirate_yaml:
                self.k_top = delay_edge_to_top(request.pirate_yaml)
            x = yaml.safe_load(request.xengine_yaml) if request.xengine_yaml else {}
            self.unix0 = int(x.get("unix_ns_at_seq_0", 0))
            self.dt_ns = int(x.get("dt_ns_per_seq", 5000))
        say(f"CheckConfiguration(search_ip_addr={request.search_ip_addr!r}) unix_ns_at_seq_0={self.unix0} "
            f"dt_ns_per_seq={self.dt_ns} delay_to_top={self.k_top:.5f} s/DM")
        return frb_sifter_pb2.ConfigReply(ok=True)

    def FrbEvents(self, request, context):
        cg = request.coarsegrain_snr
        lines = [f"toy_sifter beamset={request.beam_set_id} fpga=[{request.chunk_fpga_start}:{request.chunk_fpga_end}] "
                 f"coarse_snr_max={max(cg) if len(cg) else 0.0:.4g}"]
        for ev in request.events:
            lines.append(f"    beam_id={ev.beam_id}, dm={ev.dm:.4g}, fpga_timestamp={ev.fpga_timestamp}, "
                         f"width={ev.width_ms:.4g} ms, subband=[{ev.subband_freq_lo_MHz:.1f},{ev.subband_freq_hi_MHz:.1f}] MHz, "
                         f"snr={ev.snr:.4g}, tree_index={ev.tree_index}")
        say("\n".join(lines))
        with self.lock:
            self.decide(list(request.events), request.chunk_fpga_start, request.chunk_fpga_end)
        return frb_sifter_pb2.FrbEventsReply(ok=True, message="")

    def decide(self, events, c0, c1):
        """Adds the events of block [c0, c1) to their groups and emits the groups whose S/N stopped growing.

        A bright pulse appears first at lower DMs (they complete earlier) with growing S/N, and the maximum arrives
        at the correct DM some blocks later. That is why a group stays open until --wait-blocks blocks pass without
        its S/N improving, and the alert carries the best event."""
        a = self.a
        s_seq = self.dt_ns * 1e-9
        t_top = [ev.fpga_timestamp * s_seq - ev.dm * self.k_top for ev in events]     # s since seq 0, at 501.6 MHz
        cand = sorted((i for i, ev in enumerate(events)
                       if ev.snr >= a.threshold and ev.dm >= a.dm_min and (a.dm_max is None or ev.dm <= a.dm_max)),
                      key=lambda i: -events[i].snr)
        for i in cand:
            ev, t = events[i], t_top[i]
            beams = {events[j].beam_id for j in range(len(events)) if abs(t_top[j] - t) <= a.window_s}
            g = next((g for g in self.groups if abs(t - g.t) <= a.window_s and ev.beam_id in g.beams), None)
            if g is None:
                g = Group(ev, t, beams, c1)
                self.groups.append(g)
            g.beams |= beams
            g.n += 1
            if len(beams) > a.max_beams:
                g.vetoed = True
            if ev.snr > g.best.snr:
                g.best, g.t, g.best_block = ev, t, c1
                if g.emitted not in (None, "veto") and ev.snr > 1.2 * g.emitted_snr:
                    self.emit(g, "update")
        length = max(c1 - c0, 1)
        for g in self.groups:
            if g.emitted is None and c1 - g.best_block >= a.wait_blocks * length:
                self.close(g)
        t_now = c1 * s_seq
        self.groups = [g for g in self.groups if g.emitted is None or t_now - g.t < a.memory_s]

    def close(self, g):
        if g.vetoed or len(g.beams) > self.a.max_beams:
            g.emitted = "veto"
            self.n_vetoed += 1
            say(f"RFI VETO beam={g.best.beam_id} DM={g.best.dm:.1f} S/N={g.best.snr:.2f}: "
                f"{len(g.beams)} beams with the same arrival")
        else:
            self.emit(g, "alert")

    def close_all(self):
        with self.lock:
            for g in self.groups:
                if g.emitted is None:
                    self.close(g)

    def emit(self, g, kind):
        a, ev, now = self.a, g.best, time.time_ns()
        ns_top = self.unix0 + int(round(g.t * 1e9))
        ns_end = self.unix0 + ev.fpga_timestamp * self.dt_ns     # arrival at the bottom edge: the whole pulse has arrived
        if kind == "alert":
            self.n_alerts += 1
            g.emitted = (f"CHARTS-{datetime.fromtimestamp(ns_top / 1e9, tz=timezone.utc):%Y%m%dT%H%M%S.%f}"[:-3]
                         + f"-b{ev.beam_id}")
        g.emitted_snr = float(ev.snr)
        al = {"id": g.emitted, "type": kind, "utc_arrival": utc(ns_top), "f_ref_mhz": F_TOP_MHZ, "beam": int(ev.beam_id),
              "dm": round(float(ev.dm), 2), "snr": round(float(ev.snr), 2), "width_ms": float(ev.width_ms),
              "subband_mhz": [round(ev.subband_freq_lo_MHz, 2), round(ev.subband_freq_hi_MHz, 2)],
              "coincident_beams": sorted(int(b) for b in g.beams), "events_in_group": g.n,
              "t_stream_s": round(g.t, 4), "fpga_timestamp": int(ev.fpga_timestamp),
              "utc_emitted": utc(now), "latency_s": round((now - ns_end) / 1e9, 3), "n": self.n_alerts}
        self.file.write(json.dumps(al) + "\n")
        self.file.flush()
        say(f"{'FRB ALERT' if kind == 'alert' else 'UPDATE'} {al['id']} beam={al['beam']} DM={al['dm']} "
            f"S/N={al['snr']} width={al['width_ms']} ms arrival={al['utc_arrival']} latency={al['latency_s']} s")
        if a.action:
            th = threading.Thread(target=run_action, args=(a.action, al), daemon=True)
            th.start()
            if a.action_wait:
                th.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("address", help="ip:port to listen on (the grouper connects here)")
    ap.add_argument("--out", type=Path, required=True, help="folder where alerts.jsonl is written")
    ap.add_argument("--threshold", type=float, default=7.6, help="minimum S/N of an alert")
    ap.add_argument("--dm-min", type=float, default=20.0)
    ap.add_argument("--dm-max", type=float, default=None)
    ap.add_argument("--max-beams", type=int, default=8, help="more beams than this with the same arrival = RFI")
    ap.add_argument("--window-s", type=float, default=0.5, help="arrival tolerance (s) for grouping and veto")
    ap.add_argument("--wait-blocks", type=int, default=1, help="blocks without S/N improvement before emitting the alert")
    ap.add_argument("--memory-s", type=float, default=60.0, help="seconds an emitted group is remembered")
    ap.add_argument("--action", default=None, help="command run per alert, with the JSON on standard input")
    ap.add_argument("--action-wait", action="store_true", help="wait for the command to finish (default: no)")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *x: (_ for _ in ()).throw(KeyboardInterrupt()))
    service = AlertSifter(a)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[("grpc.so_reuseport", 0)])
    frb_sifter_pb2_grpc.add_FrbSifterServicer_to_server(service, server)
    if server.add_insecure_port(a.address) == 0:
        sys.exit(f"ERROR: could not listen on {a.address} (in use?)")
    server.start()
    say(f"alert_sifter: waiting for the grouper at {a.address}; S/N threshold {a.threshold}, DM {a.dm_min}–{a.dm_max}, "
        f"veto > {a.max_beams} beams; alerts in {a.out / 'alerts.jsonl'}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        service.close_all()
        say(f"alert_sifter: {service.n_alerts} alerts, {service.n_vetoed} vetoed as RFI")
        server.stop(grace=1.0).wait()


if __name__ == "__main__":
    main()
