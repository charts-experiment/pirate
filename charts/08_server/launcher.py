#!/usr/bin/env python3
"""Test launcher: builds pirate's configs from ONE YAML file and manages the processes.

Subcommands:
  server    On the search machine: alert sifter + grouper v1 + pirate server. Keeps listening on the receivers
            of the YAML until Ctrl-C, `--duration`, or until the file <out>/STOP exists.
  emitter   Runs 07_emitter/emitter.py with the same YAML (can be on another machine: only needs numpy and PyYAML).
  local     Both on this machine, one after the other, and evaluates the run at the end.

Usage:
  python 08_server/launcher.py local   --config 06_configs/test_64b.yml --out results/short --limit 20
  python 08_server/launcher.py server  --config 06_configs/test_64b.yml --out results/remote
  python 08_server/launcher.py emitter --config 06_configs/test_64b.yml --out results/remote --receivers 10.0.0.2:5000
Options that are not the launcher's (--limit, --rfi, --mask, --pacing, ...) are passed to the emitter.

In <out>: config_dedispersion.yml and config_server.yml (generated), server.log, grouper.log, sifter.log,
alerts.jsonl (FRB alerts, also shown on the console), all_events.log, grouper_timings.csv, gpu.csv, vmstat.log
and what the emitter writes.
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------------- pirate configs
def pirate_configs(cfg, out):
    """Writes pirate's dedispersion and server configs from the test YAML."""
    t, net, s = cfg["test"], cfg["network"], cfg.get("server", {})
    d = dict(s.get("dedispersion", {}))
    dd = {
        "zone_freq_edges": [float(x) for x in t["band_mhz"]],
        "zone_nfreq": [int(t["channels"])],
        "time_sample_ms": float(t.get("sample_ms", 1.0)),
        "beams_per_gpu": int(t["beams"]),
        "dtype": "float16",
        "time_samples_per_chunk": int(t.get("block_samples", 2048)),
        "frequency_subband_counts": d.get("frequency_subband_counts", [0, 0, 0, 0, 1]),
        "max_gpu_clag": int(d.get("max_gpu_clag", 5)),
        "beams_per_batch": int(d.get("beams_per_batch", 2)),
        "num_active_batches": int(d.get("num_active_batches", 2)),
        "toplevel_tree_rank": int(d.get("rank", 15)),
        "primary_trees": d.get("trees", [{"num_early_triggers": 0, "max_width": 32,
                                         "wt_dm_downsampling": 64, "wt_time_downsampling": 64}]),
        "future_write_max_samples": int(d.get("future_write_max_samples", 5000)),
    }
    sv = {
        "num_servers": 1,
        "server_cpus": s.get("cpus", [0]),
        "host_memory_per_server": f"{s.get('ram_gib', 150)} GiB",
        "gpu_memory_per_server": f"{s.get('gpu_gib', 17)} GiB",
        "use_hugepages": bool(s.get("hugepages", False)),
        "ringbuf_nchunks": int(s.get("ring_blocks", 64)),
        "data_ip_addrs": [list(net["receivers"])],
        "rpc_ip_addrs": [net.get("rpc", "127.0.0.1:6000")],
        "grouper_ip_addrs": [net.get("grouper", "127.0.0.1:7000")],
        "min_data_mtu": 1500, "min_rpc_mtu": 1500,
        "ssd_dirs": [s.get("ssd_dir", "/dev/shm/pirate_ssd")],
        "ssd_threads_per_server": 1,
        "nfs_dir": str(Path(s.get("dumps_dir", out / "dumps")).resolve()),
        "nfs_threads_per_server": 1,
        "check_mountpoints": [],
    }
    (out / "config_dedispersion.yml").write_text(yaml.safe_dump(dd, sort_keys=False, default_flow_style=None))
    (out / "config_server.yml").write_text(yaml.safe_dump(sv, sort_keys=False, default_flow_style=None))
    Path(sv["nfs_dir"]).mkdir(parents=True, exist_ok=True)
    return out / "config_dedispersion.yml", out / "config_server.yml"


def port_free(address):
    host, port = address.rsplit(":", 1)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, int(port)))
            return True
        except OSError:
            return False


# ------------------------------------------------------------------------------------------------ processes
class Processes:
    def __init__(self, out):
        self.out, self.items, self.monitors = out, [], []

    def start(self, name, cmd):
        log = open(self.out / f"{name}.log", "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.items.append((name, p))
        self.log(f"{name} pid {p.pid}: {' '.join(map(str, cmd))}")
        return p

    def monitor(self, cmd, filename):
        if shutil.which(cmd[0]):
            self.monitors.append(subprocess.Popen(cmd, stdout=open(self.out / filename, "w"), stderr=subprocess.DEVNULL))

    def log(self, msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(self.out / "launcher.log", "a") as f:
            f.write(line + "\n")

    def shutdown(self):
        self.log("shutting down")
        for _, p in reversed(self.items):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        end = time.time() + 60
        for _, p in self.items:
            try:
                p.wait(timeout=max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        for m in self.monitors:
            m.terminate()


def wait_for_text(path, text, timeout_s, process):
    end = time.time() + timeout_s
    while time.time() < end:
        if path.exists() and text in path.read_text(errors="replace"):
            return True
        if process.poll() is not None:
            return False
        time.sleep(1)
    return False


def pirate_environment(cfg):
    """Finds pirate_frb: $CHARTS_PIRATE_ENV, environments.pirate in the YAML, this Python's environment, or the PATH."""
    candidates = [os.environ.get("CHARTS_PIRATE_ENV"), (cfg.get("environments") or {}).get("pirate")]
    for c in candidates:
        if c:
            c = Path(os.path.expandvars(os.path.expanduser(str(c))))
            if (c / "bin" / "pirate_frb").exists():
                return str(c / "bin" / "python"), str(c / "bin" / "pirate_frb")
            raise SystemExit(f"ERROR: {c}/bin/pirate_frb does not exist (check CHARTS_PIRATE_ENV or environments.pirate)")
    here = Path(sys.executable).parent / "pirate_frb"
    if here.exists():
        return sys.executable, str(here)
    in_path = shutil.which("pirate_frb")
    if in_path:
        return str(Path(in_path).parent / "python"), in_path
    raise SystemExit("ERROR: pirate_frb not found. Activate its environment (conda activate ...) or "
                     "export CHARTS_PIRATE_ENV=/path/to/environment. Installation: INSTALL.md")


def show_alerts(path, pr):
    """Shows on the console (and in launcher.log) each alert the sifter writes."""
    while not path.exists():
        time.sleep(1)
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                x = json.loads(line)
                pr.log(f"*** FRB ALERT {x['id']}: beam {x['beam']}, DM {x['dm']}, S/N {x['snr']}, "
                       f"arrival {x['utc_arrival']}, latency {x['latency_s']} s")
            except (ValueError, KeyError):
                pass


def start_server(cfg, out, pr):
    net, s = cfg["network"], cfg.get("server", {})
    py, pirate = pirate_environment(cfg)
    pr.log(f"pirate: {pirate}")
    busy = [d for d in list(net["receivers"]) + [net.get("rpc", "127.0.0.1:6000"), net.get("grouper", "127.0.0.1:7000"),
                                                 net.get("sifter", "127.0.0.1:7500")] if not port_free(d)]
    if busy:
        raise SystemExit(f"ERROR: ports in use {busy}: is another run active?")
    dd, sv = pirate_configs(cfg, out)
    pr.monitor(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv", "-l", "5"], "gpu.csv")
    pr.monitor(["vmstat", "-t", "5"], "vmstat.log")
    al = s.get("alerts") or {}
    args_s = [py, str(ROOT / "08_server" / "sifter" / "alert_sifter.py"), net.get("sifter", "127.0.0.1:7500"),
              "--out", str(out), "--threshold", str(al.get("threshold", 7.6)), "--dm-min", str(al.get("dm_min", 20)),
              "--max-beams", str(al.get("max_beams", 8)), "--window-s", str(al.get("window_s", 0.5))]
    args_s += ["--dm-max", str(al["dm_max"])] if al.get("dm_max") is not None else []
    args_s += ["--action", str(al["action"])] if al.get("action") else []
    (out / "alerts.jsonl").unlink(missing_ok=True)
    pr.start("sifter", args_s)
    threading.Thread(target=show_alerts, args=(out / "alerts.jsonl", pr), daemon=True).start()
    time.sleep(3)
    g = s.get("grouper", {})
    args_g = [py, str(ROOT / "08_server" / "grouper" / "grouper_v1.py"), net.get("grouper", "127.0.0.1:7000"),
              "-s", net.get("sifter", "127.0.0.1:7500"), "-t", str(g.get("threshold", 6)), "--out", str(out / "windows")]
    args_g += ["--no-windows"] if not g.get("windows", False) else []
    args_g += ["--log-all", str(out / "all_events.log")] if g.get("full_log", True) else []
    args_g += ["--profile"] if g.get("profile", False) else []
    args_g += [str(x) for x in g.get("extra", [])]
    pr.start("grouper", args_g)
    time.sleep(3)
    srv = pr.start("server", [pirate, "run", "server", str(sv), str(dd)])
    if not wait_for_text(out / "server.log", "server(s) started", 900, srv):
        raise SystemExit("ERROR: the pirate server did not start (see server.log)")
    pr.log(f"server ready; receivers {', '.join(net['receivers'])}")


def emitter_cmd(cfg_path, out, extra):
    return [sys.executable, str(ROOT / "07_emitter" / "emitter.py"), "--config", str(cfg_path), "--out", str(out)] + extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["server", "emitter", "local"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output folder of the run")
    ap.add_argument("--duration", type=float, help="seconds the server keeps listening (server mode)")
    ap.add_argument("--no-eval", action="store_true", help="do not evaluate at the end (local mode)")
    a, extra = ap.parse_known_args()
    cfg = yaml.safe_load(open(a.config))
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, a.out / "test.yml")

    if a.mode == "emitter":
        sys.exit(subprocess.call(emitter_cmd(a.config, a.out, extra)))

    if a.mode == "local":
        sd = os.path.expandvars(str((cfg.get("emitter") or {}).get("set_dir") or "$CHARTS_SET_DIR"))
        if "--set-dir" not in extra and ("$" in sd or not Path(sd).is_dir()):
            raise SystemExit(f"ERROR: data folder {sd!r} not found: export CHARTS_SET_DIR=/path/to/group3")
    pr = Processes(a.out)
    signal.signal(signal.SIGTERM, lambda *x: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        start_server(cfg, a.out, pr)
        if a.mode == "server":
            end = time.time() + a.duration if a.duration else None
            while (end is None or time.time() < end) and not (a.out / "STOP").exists():
                if any(p.poll() is not None for _, p in pr.items):
                    pr.log("a process exited by itself; see the logs"); break
                time.sleep(2)
        else:
            rc = subprocess.call(emitter_cmd(a.config, a.out, extra))
            pr.log(f"emitter finished (rc={rc}); waiting 30 s for pirate to process the queue")
            time.sleep(30)
    except KeyboardInterrupt:
        pr.log("interrupted")
    finally:
        pr.shutdown()
    if a.mode == "local" and not a.no_eval:
        py, _ = pirate_environment(cfg)
        subprocess.call([py, str(ROOT / "09_evaluation" / "evaluate_test_64b.py"), str(a.out)])
        pr.log(f"evaluation in {a.out / 'results.md'}")
        subprocess.call([py, str(ROOT / "09_evaluation" / "verify_alerts.py"), str(a.out)])


if __name__ == "__main__":
    main()
