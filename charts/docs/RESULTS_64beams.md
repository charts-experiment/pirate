# Results of the 64-beam real-time test

Period: 2026-10-08 to 2026-11-05. Machine: RTX 4090 (24 GB), Xeon w5-3435X (16 cores), 314 GB of RAM.

## 1. Common configuration

- Data: group 3, 4000 .fil files of 21 504 channels (9.389 kHz, 300–502 MHz), 1 ms, 8 bits (floor 32, σ = 4 counts), without RFI of their own. Sets of 1000: A (expected S/N 8), B (617 Jy), negatives (dispersion index ≠ −2) and partial.
- Distributed over 64 streams (one per beam), with 10 s of noise before and 60 s after; 3510 s of stream per beam.
- pirate: rank 15, one tree, DM ≤ 1105, boxcar ≤ 32 ms (`06_configs/charts_64b_dm1000.yml`), server `06_configs/server_64b.yml`.
- Hit: DM ±10 % and time ±max(50 ms, boxcar). False alarms (FA): events outside the signal windows, per beam·hour of signal-free time. FP in negatives: non-FRB with any event within ±100 ms. Metrics with DM ≤ 1000.

## 2. Reference without RFI with the toy grouper (2026-10-10)

At 10 FA per beam·hour (threshold 6.54): A 91.9 %, B 82.8 %, partial 64.5 %, FP in negatives 41.1 %.
Chain at 1.98× real time.

## 3. Site RFI (T3d, 2026-11-03/04)

- Model: site baseband (2026-09-02, 8 antennas, 4.1 s) channelized ×32 to 21 504 channels × 1 ms (`05_site_rfi/`). 2.1 % of the channels with lines > 10 dB; narrow fast RFI in 0.5 % of the pixels; no broadband impulses.
- Injected on the fly by the emitter (`--rfi`), with the same clock in the 64 beams and per-channel, per-block normalization.
- **Unmitigated** (short test, 80 files): ~110× more candidates (47 per beam and block), 99 % present in ≥ 32 beams; no threshold reaches 10 FA/h; real completeness ~8 % (21 % of the "hits" were chance coincidences with RFI). Removing the grouper's cap of 4 events per beam does not recover the FRBs.
- Grouper diagnosis: discarding by t_top is not the cause; the merging of regions with S/N ≥ 5 absorbs some FRBs (with cell threshold 7 the real detection rises from ~8 to ~28 %), but the underlying problem is the RFI in pirate's map.

## 4. Static channel mask (T3e, 2026-11-04/05)

- m1: lines > 3 dB or fast RFI > 1 % of the time = 17.5 % of the band (removes 98 % of the fast RFI). m2: > 1 dB or > 0.2 % = 39 %.
- Masked channels are sent as missing data (−8). pirate computes the S/N assuming unit variance in every channel, so the scale of the good channels is multiplied by √(N/N_good); without this correction the whole S/N is scaled by √(1 − f).
- Short tests: with RFI + m1 the real detection rises to 78 % (control 83 %); m2 loses more S/N without improving.

### Full runs with m1

| | No RFI + m1 | RFI + m1 |
|---|---|---|
| Candidates (S/N > 6, no caps) | 49 496 (0.45 per beam and block) | 185 502 (1.7) |
| FA at S/N ≥ 6 | 674 per beam·hour | 2 463 |
| Threshold for 10 FA per beam·hour | 6.72 | 7.04 |
| Completeness A / B / partial | 88.9 / 82.5 / 63.2 % | 76.8 / 81.8 / 58.8 % |
| FP in negatives | 40.7 % | 39.1 % |
| First threshold without FA (≤ 0.3 per beam·hour) | 7.24: 77.1 / 81.1 / 58.8 % | 7.60: 64.3 / 80.5 / 52.8 % |
| Chance hits (S/N ≥ 6) | ≤ 0.7 % | ≤ 2 % |

## 5. Timing and resources

- Detection per 2.048 s block (without the emitter): pirate 0.17 s (`time_dedisperser`, 764 beams in real time with the GPU to itself) + grouper v1 0.32 s without RFI / 0.42 s with RFI + m1 (`--profile`) = **0.49 / 0.59 s**. The full chain with the test emitter takes 0.96 / 1.44 s per block: the emitter slows it down.
- FRB alerts (2026-11-15, 8 files without RFI): latency 4.6–5.8 s from the end of the pulse arrival to the alert (≤ 2 s waiting for the end of the block + 2 s for one block without S/N improvement + ~2.4 s of pirate and grouper).
- Pending: measuring network reception and block assembly inside the server (the 0.17 s comes from the isolated benchmark).
- Resources (short test with RFI + m1, per process): pirate server 17.4 GiB of GPU, 150 GiB of RAM (the whole allocated pool; it needs ~20 GiB of tree + the ring buffer), ~0.3 cores on average; grouper 0.6 GiB of GPU, ~0.15 cores.
- Capacity (`04_pirate_capacity/`): the tree memory grows ×4 for each doubling of the maximum DM (rank +1); rank 16 is pirate's maximum; extra trees with 2 and 4 ms samples are much cheaper; with the tree in RAM, PCIe (~22 GB/s) limits the capacity.

## 6. Limitations

See the main README. In particular: RFI from a single 4.1 s capture repeated (average of 8 antennas, more optimistic than antenna 4), mask built from the same capture that is injected, no level variations common to the whole band, and identical RFI in all beams.
