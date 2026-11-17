# How to send data to pirate: network protocol and Kotekan adaptation

This document explains what the X-engine (or any emitter) has to send for pirate to search for FRBs, and how to
port this repository's reference emitter (`07_emitter/`, Python) to a **Kotekan** stage (C++).

## 1. What pirate receives

pirate receives **intensity already formed into beams**: a cube `(beam, frequency channel, time)`.

| | 64-beam test | Set by |
|---|---|---|
| Beams per search node | 64 | `test.beams` |
| Channels | 21 504 (672 coarse × 32), 299.85–501.75 MHz | `test.channels`, `test.band_mhz` |
| Time sample | 1 ms | `test.sample_ms` |
| Bits per sample | 4 (int4) + scale and offset in float16 | protocol |
| Rate per search node | 64 × 21 504 × 4 bits × 1000/s = 5.5 Gbit/s + ~3 % of scales | — |

Beamforming, the ×32 channelization and the integration to 1 ms are done by the X-engine; pirate starts at the
requantization to 4 bits.

## 2. The protocol (version 2)

Reference: pirate's `notes/network_protocol.md`. Reference implementation: `07_emitter/protocol.py`.

**One persistent TCP connection per emitter → receiver pair.** Each emitter (X-engine node) sends **a subset of
channels for all the beams** of the receiver. The receiver gathers the channels of all its emitters.

When the connection opens (everything little-endian):

| Field | Type | Value |
|---|---|---|
| magic | uint32 | `0xf4bf4b02` (the `02` is the version) |
| flags | uint32 | `0x1` = FLAG_ACK (the receiver acknowledges each minichunk; for tests only) |
| length | uint32 | bytes of the text that follows, including ≥ 1 zero padding byte |
| metadata | zero-terminated ASCII text | YAML in the format of pirate's `configs/xengine_metadata.yml` |

Required metadata (see `metadata_yaml()` in `07_emitter/protocol.py`):

| Field | What it is | Same for all emitters? |
|---|---|---|
| `version` | 2 | yes |
| `zone_nfreq`, `zone_freq_edges` | channels and band edges in MHz (one or more zones) | yes |
| `freq_channels` | **the channels this connection sends** (indices, 0 = lowest frequency) | no |
| `beamset`, `beam_ids`, `beam_positions_x/y` | identification of the receiver's beams | yes |
| `unix_ns_at_seq_0`, `dt_ns_per_seq`, `seq_per_frb_time_sample` | clock: epoch, ns per seq count, counts per 1 ms sample | yes |
| `tel_*` | telescope geometry (pirate requires it; the search does not use it) | yes |
| `noise_variance` | noise variance per zone (1.0 if the data are already normalized) | yes |

Then **minichunks** of 256 time samples, one after another, with no padding:

| Field | Type | Size |
|---|---|---|
| magic | uint32 | 4 bytes |
| seq | uint64 | 8 bytes: seq of the start of the minichunk |
| scales and offsets | float16 `[nbeams, nfreq, 2]` | nbeams × nfreq × 4 bytes |
| data | int4 `[nbeams, nfreq, 256]` | nbeams × nfreq × 128 bytes |

- `nfreq` = length of that connection's `freq_channels`. Time is the fastest axis.
- Two samples per byte: `(x[1] << 4) | x[0]`. The value −8 marks a missing sample (pirate adds it as 0).
- pirate reconstructs `value = scale × q + offset` and assumes **zero mean and unit variance** per channel: the scale
  and offset must leave the data that way (or with variance = `noise_variance`).
- Consecutive minichunks differ in seq by `256 × seq_per_frb_time_sample`. The first one must be a multiple of that
  value. Skipping seq = missing data.
- With FLAG_ACK, after each minichunk the receiver answers 1 byte: 1 = copied to the ring buffer, 0 = dropped.

## 3. Requantization to 4 bits (the most important part for sensitivity)

Measured in the tests (`docs/ANALYSIS_pirate_quantization_and_group3_21504_2026-10-09.md`, `docs/RESULTS_64beams.md`):

- **2 levels per σ** (scale = σ/2), centred rounding and clipping to ±7 keeps ~99 % of the S/N.
- **The scale and offset must be per channel.** With one fixed value for all channels, channels with strong RFI stay
  stuck at +7 and add an offset to every DM (every block gave S/N ≈ 118).
  `07_emitter/quantization.py::per_channel`: median and σ = (p84 − p16)/2 per channel and block.
- **Bad channels as −8** (known RFI mask, saturated channels) **and correct the scale of the good ones** by
  √(N/N_good). pirate computes the S/N assuming unit variance in every channel: without the correction, the whole S/N
  comes out scaled by √(1 − masked fraction).

## 4. The reference emitter (`07_emitter/`)

```
sources.py       where the data come from (.fil files, noise, fill)          ← in Kotekan: the input buffer
quantization.py  8 bits → 4 bits + scale/offset per minichunk, mask          ← in Kotekan: the stage's computation
protocol.py      header, YAML metadata, minichunks, int4 packing             ← in Kotekan: sending on the socket
emitter.py       preparation processes + 1 thread per connection, pacing     ← in Kotekan: main_thread()
```

To test another data source without touching the rest: a class with `block(t0, T, out)` that fills `out [T, nchan]`
uint8 (see `sources.py`).

## 5. Kotekan adaptation

Kotekan organizes the processing in **stages** (C++ classes that inherit from `kotekan::Stage`) connected by
**buffers** of frames with metadata, configured in YAML.

### 5.1 Existing reference stages

| Stage | What to take from it |
|---|---|
| `lib/stages/bufferSend.{hpp,cpp}` | **The main model:** persistent TCP connection, reconnection (`reconnect_time`), `send()` with `MSG_NOSIGNAL` in a loop, timeout (`SO_SNDTIMEO`), dropping frames if the socket is slow (`drop_frames`) |
| `lib/stages/frbNetworkProcess.{hpp,cpp}` | CHIME's FRB sender (UDP, several destinations, liveness pings, REST endpoints to enable/disable destinations) |
| `lib/stages/UpchannelizationSchedule.*`, `chartsBasebandReadout.*` | CHARTS' own stages: what the intensity buffer and its metadata look like in our chain |

### 5.2 Proposed stage: `pirateSender`

```
[intensity buffer: beam × channel × time, float/uint8]  ──►  pirateSender  ──TCP──►  pirate receivers
```

Suggested YAML configuration (same keys as `06_configs/test_64b.yml`):

```yaml
pirate_sender:
  kotekan_stage: pirateSender
  in_buf: frb_intensity_buffer
  receivers: ["10.0.0.2:5000"]      # one per search node (or several receivers per node)
  freq_channels: [0, 5376]          # channel range sent by THIS X-engine node
  beam_ids: [0, 63]
  zone_nfreq: [21504]
  zone_freq_edges: [299.8497764530551, 501.7502235469449]
  samples_per_minichunk: 256
  requantization: per_channel       # median and σ per channel (see section 3)
  mask_file: mask_m1.bin            # channels sent as −8 (optional)
  ack: false
  reconnect_time: 5
```

Steps of `main_thread()`:

1. **Start-up:** build the YAML metadata text (like `metadata_yaml()`), with `unix_ns_at_seq_0` = Kotekan's
   `time0_ns`, `dt_ns_per_seq` = F-engine FFT length / sampling frequency, and `seq_per_frb_time_sample` =
   channelization factor × integration factor (as pirate proposes in `configs/xengine_metadata.yml`). Open one
   connection per receiver and send the header.
2. **Per buffer frame:** `wait_for_full_frame()`; take the frame's seq from its metadata.
3. **Requantize** each beam and channel to int4 with scale/offset per minichunk (section 3). This is the compute step:
   on the CPU with SIMD or on the GPU if the buffer is already there (in the tests, ~0.2 s per beam and 2 s block in
   one numpy thread; vectorized C++ should be much less).
4. **Pack** with the protocol layout: for each minichunk, `[magic][seq][float16 nbeams×nfreq×2]`
   `[uint8 nbeams×nfreq×128]`. It helps to keep the frame already in the order `[minichunk, beam, channel, 128 bytes]`
   (like `emitter.py`), so that each send is contiguous.
5. **Send** with `send()` in a loop (like `bufferSend`); if `ack` is enabled, read 1 byte per minichunk.
6. `mark_frame_empty()`.

Recommendations:
- **Continuous seq:** if a frame is lost, skip its seq (pirate treats it as missing); do not resend.
- **Reconnection:** on reconnecting the header must be sent again; the first seq must be a multiple of `256 × seq_per_frb_time_sample`.
- **Several receivers per search node** if the rate per receiver exceeds ~11–13 Gbit/s (measured: one pirate receiver
  reached ~13.5 Gbit/s and two ~16 with 256 beams).
- **Validation:** the same cube sent with `07_emitter/emitter.py` and with the stage must give the same events in
  pirate (same DM, time and S/N). For a byte-by-byte comparison, `protocol.py` builds exactly the same bytes.

### 5.3 What remains to be defined with the X-engine team

- Format and metadata of the intensity buffer in Kotekan (dtype, axis order, where the seq goes).
- Whether the requantization to 4 bits is done in the X-engine (this stage) or in a converter on the search node.
- How channels are split between X-engine nodes and beams between search nodes (`freq_channels`, `beam_ids`).
- The real telescope geometry (`tel_*`) and the beam positions.
