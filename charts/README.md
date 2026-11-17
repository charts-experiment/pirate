# CHARTS · real-time FRB search with pirate (64 beams)

Code, configurations and results of the real-time FRB search test for **CHARTS** with
[pirate](https://github.com/kmsmith137/pirate) (tree dedispersion on the GPU): **64 beams × 21 504 channels**
(300–502 MHz, 9.4 kHz, 1 ms), DM ≤ 1105, on one RTX 4090.

The project is split into independent pieces that communicate only over the network, using pirate's protocol:

```
  Machine producing the data (X-engine or test)         Search machine (GPU)
  ┌──────────────────────────────────┐                 ┌───────────────────────────────────────────────┐
  │ 07_emitter/  (Python + numpy)       │   TCP, 4 bits   │ 08_server/launcher.py                            │
  │  reads .fil or another source,   │ ──────────────► │  pirate (this fork): RAM ring + GPU tree      │
  │  optional RFI and mask,          │  + scale/offset │  grouper v1: events in the DM-time map        │
  │  4 bits + scale/offset           │                 │  sifter: decides and emits FRB ALERTS         │
  └──────────────────────────────────┘                 └───────────────────────────────────────────────┘
                                                        alerts.jsonl → 09_evaluation/: verified alerts,
                                                                       completeness and false alarms
```

The test data (4000 simulated FRBs, 21 504-channel `.fil` files) are generated with `01_simulator/`.
The emitter does not depend on pirate: it can run on another machine. To port it to **Kotekan**, see
[`docs/PROTOCOL_pirate_and_Kotekan_adaptation.md`](docs/PROTOCOL_pirate_and_Kotekan_adaptation.md).
To understand what each file does, see [`docs/STRUCTURE.md`](docs/STRUCTURE.md).

## Paths: the only things to provide

There are no machine-specific paths in the code. The test needs only two things:

| What | How | If missing |
|---|---|---|
| Where pirate is | `conda activate <pirate environment>` (autodetected), or `export CHARTS_PIRATE_ENV=/path/to/environment` | the launcher says it cannot find `pirate_frb` |
| Where the data are (group-3 .fil files) | `export CHARTS_SET_DIR=/path/to/group3` | the launcher and the emitter say so before starting |

The other variables (table at the end) are only for optional scripts, and each one says when its variable is missing.
Everything else is found relative to the `charts/` folder.

## Main results

Details in [`docs/RESULTS_64beams.md`](docs/RESULTS_64beams.md). Full runs of 4000 files
(297 A, 297 B and 318 partial with DM ≤ 1000), grouper v1, channel mask m1:

| | No RFI + mask | Unmitigated site RFI* | Site RFI + mask m1 |
|---|---|---|---|
| False alarms at S/N ≥ 6 (per beam·hour) | 674 | ~66 000 | 2 463 |
| S/N threshold for 10 FA per beam·hour | 6.72 | not reached | 7.04 |
| Completeness A / B / partial at 10 FA/h | 89 / 83 / 63 % | ~8 % | 77 / 82 / 59 % |
| False positives in negatives | 41 % | 100 % | 39 % |
| Detection per 2.048 s block (pirate + grouper, without emitter) | 0.49 s | — | 0.59 s |

\* Short test of 80 files; completeness corrected for chance coincidences.
Set A: expected S/N 8 · set B: fixed flux 617 Jy · partial: burst partly outside the file · negatives: dispersed
signals with index ≠ −2 (non-FRB).

Those results were obtained with an earlier chain (emitter based on pirate's `FakeXEngine`, not included here).
The chain in this folder (`07_emitter/` + `08_server/launcher.py`) was validated against it: same detections in the 8-file test.

## FRB alerts

The last stage, `08_server/sifter/alert_sifter.py`, decides which event is an FRB and emits an alert: one JSON line in
`<out>/alerts.jsonl` (id, UTC arrival time at 501.6 MHz, beam, DM, S/N, width, coincident beams, latency), a
message on the launcher console and, if configured, a command per alert with the JSON on standard input
(`server.alerts.action`: notify, dump data, VOEvent...). Default rule (`server.alerts` in the YAML):

- S/N ≥ 7.6 (~10 false alarms per hour for the 64-beam system with site RFI + mask m1) and DM ≥ 20;
- RFI veto: more than 8 beams with the same arrival;
- a group (same arrival in consecutive blocks or in coincident beams) is emitted when one block passes without its S/N
  improving, with its best event: a bright pulse appears earlier at lower DMs and with lower S/N.

`09_evaluation/verify_alerts.py` matches each alert with the injected FRBs → `alerts.md` (the `local` mode runs it).
8-file test without RFI (2026-11-15): 2 alerts, both true FRBs at the correct DM, 0 false alarms,
latency 4.6–5.8 s from the end of the pulse arrival (≤ 2 s waiting for the end of the block + 2 s for the block
without improvement + ~2.4 s of pirate and grouper).

## Installation

See **[`INSTALL.md`](INSTALL.md)**: what to install on each machine, the address and commit of each repository
(this fork, ksgpu, fitburst), and the `install_pirate.sh` script.

## Usage

All commands are run from the `charts/` folder.

```bash
conda activate /path/to/pirate/environment         # or export CHARTS_PIRATE_ENV=/path/to/pirate/environment
export CHARTS_SET_DIR=/path/to/group3              # simulator .fil files (A_snr_fijo, B_amplitud_fija, negativos, parcial)

# everything on this machine: server + emitter + evaluation (80 files)
python 08_server/launcher.py local --config 06_configs/test_64b.yml --out results/short --limit 20 --processes 16

# with the site RFI and mask m1
python 08_server/launcher.py local --config 06_configs/test_64b.yml --out results/short_rfi --limit 20 \
    --rfi --mask 05_site_rfi/mask_m1.npy

# on two machines: on the search machine (with network.receivers = IP of this machine in the YAML) ...
python 08_server/launcher.py server --config 06_configs/test_64b.yml --out results/remote
# ... and on the one producing the data (only needs numpy and PyYAML)
python 07_emitter/emitter.py --config 06_configs/test_64b.yml --out results/remote --receivers 10.0.0.2:5000
# afterwards, on the search machine
python 09_evaluation/evaluate_test_64b.py results/remote
```

Everything is configured in **one file** (`06_configs/test_64b.yml`): beams, channels, band, network (receivers,
connections), server memory, dedispersion and grouper parameters, alert rule, and emitter options. The emitter's
command-line options (`python 07_emitter/emitter.py --help`) override the YAML.

Before launching: the GPU must be free and the `network` ports unused (the launcher checks this).

## Requirements

| Piece | Needs |
|---|---|
| `07_emitter/` | Python ≥ 3.9, numpy, PyYAML. Optional: cupy (`gpu: true`) and, for `--rfi`, the model `05_site_rfi/site_rfi_model.npz` (177 MB, not in git; regenerated with `05_site_rfi/01_` and `02_`) |
| `08_server/` | this fork of pirate built (based on kmsmith137/pirate commit `255edd26`, 1.5.0), `sm_89`; its environment (Python 3.13, numpy 2.5, cupy 14.2); NVIDIA GPU (tested on RTX 4090, driver 590, CUDA 13.1); ~150 GB of RAM with the default config. Notes on pirate: [`docs/GUIDE_pirate_installation_usage_benchmark_CHARTS_2026-10-08.md`](docs/GUIDE_pirate_installation_usage_benchmark_CHARTS_2026-10-08.md). pirate's code was not modified |
| `09_evaluation/` | pirate's environment (uses its `DedispersionConfig`) |
| `05_site_rfi/` 00–07 | numpy, scipy, h5py, matplotlib; the site baseband to regenerate the model |
| `01_simulator/` | Python 3.12, numpy, scipy, fitburst (CHIME/FRB, commit `3c76da8`): generates the 4000 group-3 .fil files (~4.3 TB; not in git) |

Performance of the Python emitter (64 beams, 16 processes, 16-core CPU): ~1.07× real time without RFI.

## Environment variables

| Variable | Used by | Required |
|---|---|---|
| `CHARTS_SET_DIR` | emitter (if `emitter.set_dir` in the YAML is `null`), `02_quantization/` | yes, for the test |
| `CHARTS_PIRATE_ENV` | launcher and evaluation (if `environments.pirate` is `null`). Without it, `pirate_frb` is looked for in the active environment and in the PATH | no, if pirate's environment is activated |
| `CHARTS_RFI_MODEL` | `--rfi`: `site_rfi_model.npz`, if it is not in `05_site_rfi/` | no |
| `CHARTS_BASEBAND_DIR` | `05_site_rfi/00` and `01` (site baseband) | only to regenerate the RFI |
| `CHARTS_RFI_RESULTS` | `10_rfi_analysis/` (folder with several runs) | only for those scripts |
| `CHARTS_SUBBAND_RESULTS` | `09_evaluation/subbands/paired_comparison.py` | only for that script |
| `CHARTS_RUN`, `CHARTS_ANALYSIS_OUT` | `09_evaluation/map_statistics/` (or pass the run as an argument) | no |

## Data formats kept as generated

The names of the group-3 dataset on disk (folders `A_snr_fijo`, `B_amplitud_fija`, `negativos`, `parcial`, and the
columns of their `manifest.csv`, e.g. `t_top_s`, `snr_esperado`, `barrido_s`, `dm_equivalente_pos`) come from the
generators and are kept unchanged, so that the existing 4.3 TB dataset remains readable and the simulator still
reproduces it byte for byte. `05_site_rfi/02_rfi_model.py` now writes English keys in `site_rfi_model.npz`;
`site_rfi.py` also reads model files written with the earlier Spanish keys.

## Known limitations

- The RFI is a single 4.1 s capture (average of 8 antennas) that repeats; it does not include broadband impulses or level variations common to the whole band; all beams see it the same way.
- The mask comes from the same capture that is injected (perfect knowledge of the RFI).
- The grouper (v1) and the alert sifter are our own and simple; there is no classifier (RFI vs. FRB) nor coordination between several search nodes.
- 10 FA/h for the 64-beam system (0.16 per beam·hour) cannot be measured with these runs: the signal-free time is ~3 beam·hours.
- With RFI, the test emitter on the CPU runs at ~0.16× real time; to measure latency with RFI use `gpu: true` or `.fil` files with the RFI already injected.
- In the emitter, the noise in the gaps between files comes from 8 precomputed blocks that repeat (with channels rotated per beam); `precomputed_noise: false` generates new noise, about 5 times slower.

CHARTS project, Departamento de Astronomía, Universidad de Chile.
