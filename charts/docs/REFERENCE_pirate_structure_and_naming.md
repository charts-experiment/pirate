# pirate: work structure, naming and data outputs

Reference for CHARTS: 21 504 channels, 1 ms per sample, 300–501.6 MHz, 64 beams, 4-bit data.
Checked against pirate's code (v1.5.0, the folder above `charts/`) on 2026-10-23.

---

## 1. Naming

The names are those used by pirate's author.

| Term | What it is | Size in CHARTS | Where it is defined |
|---|---|---|---|
| sample | One time instant | 1 ms | — |
| **minichunk** | 256 samples. This is how each X-node sends the data, each with its own 4-bit scale and offset per channel | 256 ms | `notes/developer.md:43` |
| **chunk** | Range of 2048 samples of **all beams**. It is the processing unit: pirate dedisperses and delivers the map once per chunk | 2.048 s · 64 frames · ~1.45 GB | `notes/developer.md:42` |
| **batch** | Group of beams the GPU processes at once within a chunk | `beams_per_batch` | `notes/developer.md:44` |
| **frame** | A **(chunk, beam)** pair: one chunk of a single beam. It is the storage unit | ~22.7 MB (raw data) | `notes/developer.md:45`, `include/pirate/MegaRingbuf.hpp:48` |
| **frame_ringbuf** | Ring buffer of **raw data** (4 bits) in the server's RAM | ~20 chunks (40 s) × 64 beams ≈ 28 GB | `FrbServer::frame_ringbuf` |
| **MegaRingbuf** | Ring buffer of the dedispersion's **partial sums**, split into GPU and RAM zones | depends on `max_gpu_clag` | `include/pirate/MegaRingbuf.hpp:18` |
| **segment** | Smallest piece of the MegaRingbuf (one GPU cache line) | 128 bytes | `notes/developer.md:46` |
| **clag** (chunk lag) | Delay of a MegaRingbuf datum, in chunks: how many chunks back each tree row keeps | 0 to 16 in tree 0 | `MegaRingbuf.hpp:55-57` |
| **flag** (frame lag) | The same delay, counted in frames (`flag = clag × num_batches`) | — | `MegaRingbuf.hpp:57` |
| **zones** | Stretches of the MegaRingbuf with different lengths, each on the GPU or in RAM | — | `MegaRingbuf.hpp:59` |
| **max_gpu_clag** | The largest clag kept in VRAM; the rest goes to RAM | 5 in the 64-beam test | `include/pirate/DedispersionConfig.hpp:101-103` |

**Easiest to confuse:**
- **Chunk vs. frame:** a chunk is the frames of all beams in the same 2.048 s interval. With 64 beams, 1 chunk = 64 frames. pirate **processes** one chunk at a time and **stores** one frame at a time.
- **Clag:** it is not a datum but a **number each tree row has** (roughly, each DM). The stored partial sums are the history, in the MegaRingbuf.
- **Two ring buffers with different contents:**
  - frame_ringbuf: raw data, for dumps and for classifiers on raw data;
  - MegaRingbuf: half-finished sums, internal to the dedispersion.

  pirate uses the word "frame" in both, always meaning (chunk, beam).

---

## 2. Figures

### Inside a frame
![A frame: 21 504 channels × 2048 4-bit samples in 8 minichunks, plus one scale and one offset per channel and minichunk](img/fig1_frame.png)

- `data`: int4, shape (21 504, 2048), 22.0 MB. The value −8 marks a missing sample.
- `scales_offsets`: float16, shape (21 504, 8, 2), 0.69 MB.
- Metadata: `beam_id`, `time_chunk_index` and the X-engine metadata.
- `intensity = scale × 4bit_value + offset`, with the scale and offset of its channel and minichunk. On the GPU this is done by `GpuDequantizationKernel`.
- Definition: `include/pirate/AssembledFrame.hpp`.

### From frame to chunk
![A chunk is the 64 frames of the same 2.048 s interval](img/fig2_frame_chunk.png)

### The beams × chunks grid
![Grid of beams by chunks: a frame is a cell, a chunk a column, a batch a group of beams within a column, and the frame ring buffer the window of the last columns](img/fig3_grid.png)

### Where each thing lives
![Frame ring buffer in RAM, MegaRingbuf split between RAM and VRAM, dedispersion trees and DM-time map towards the grouper](img/fig4_storage.png)

### The clag
![Each tree row keeps clag chunks back; those with clag up to 5 in VRAM and the rest in RAM](img/fig5_clag.png)

At DM ≈ 1105 the low frequencies arrive ~33 s after the high ones, which is why that row keeps 16 chunks.

| `max_gpu_clag` | Where the history is kept | Cost per beam (tree 0) |
|---|---|---|
| 16 | All on the GPU | 0.56 GB of VRAM, no history traffic over PCIe |
| 5 (the 64-beam test) | 38 % on the GPU, 62 % in RAM | 0.22 GB of VRAM + 30 MB/s of PCIe |
| 1 | Almost all in RAM | 0.045 GB of VRAM + 56 MB/s of PCIe |

### The full chain
![X-engines, pirate server, grouper, sifter and trigger: what exists and what is missing](img/fig6_chain.png)

---

## 3. How data come out of pirate

pirate works in a loop: for each **chunk**, for each **batch** of beams and for each dedispersion **tree**, it delivers a result to the grouper.

### 3.1 The DM-time map (pirate → grouper, on the GPU)

For each tree, two arrays of shape **(beams_per_batch, coarse_ndm, coarse_ntime)**:
- `out_max`: maximum S/N in each cell, over the DM-time positions of the cell and over the widths;
- `out_argmax`: a token that encodes the exact position within the cell and the width of that maximum.

`FrbGrouper.create_events()` converts it into DM, time, width and subband.

With the 3-tree configuration (`charts_rt_r15t3`):

| Tree | DM range | DMs tried | Map per chunk (DM × time) | DM cell | Time cell | Widths |
|---|---|---|---|---|---|---|
| 0 (1 ms, rank 15) | 0 – 1107 | 32 768 | 2048 × 128 | 0.54 | 16 ms | 16 |
| 1 (2 ms, rank 14) | 1107 – 2213 | 16 384 | 1024 × 64 | 1.08 | 32 ms | 13 |
| 2 (4 ms, rank 14) | 2213 – 4427 | 16 384 | 1024 × 32 | 2.16 | 64 ms | 10 |

In total pirate tries 65 536 DMs: the extra trees are rank 14 and only cover the upper half of their range.

**Conditions for using the map** (`notes/grouper_interface.md`):
- It shares GPU memory with pirate (shared memory over IPC); the metadata go over gRPC on loopback.
- It is only valid **inside each batch's context manager** (`grouper.get_output(ichunk, ibatch)`). Afterwards pirate reuses that memory. To keep earlier chunks they must be copied (~67 MB per chunk for tree 0 with 64 beams).
- The grouper must finish each chunk in less than 2.048 s, or it slows pirate down.
- The GPU must not be touched outside `FrbGrouper`'s context manager.

### 3.2 The candidate list (grouper → sifter, over gRPC)

Once per chunk, an `FrbEventsMessage` (`grpc/frb_sifter.proto`):

| Field | Content |
|---|---|
| `chunk_fpga_start`, `chunk_fpga_end` | Time range of the chunk |
| `coarsegrain_snr` | Maximum S/N of each beam in that chunk (all beams) |
| `events[]` | List of `FrbEvent` |

Each `FrbEvent`:

| Field | Content |
|---|---|
| `beam_id` | Beam |
| `fpga_timestamp` | Arrival at the **bottom edge of the band**; it can fall outside the chunk |
| `dm`, `snr`, `width_ms` | Decoded from the token: exact position, not the cell's |
| `subband_freq_lo_MHz`, `subband_freq_hi_MHz` | Band where it was detected (early triggers) |
| `tree_index` | Tree of origin |
| `rfi_prob` | **Reserved for the classifier**; currently always 0 |

Before the events, the grouper sends the sifter a `ConfigMessage` with three YAMLs: pirate's configuration, the X-engine metadata and the dedispersion plan.

### 3.3 The raw data (frame_ringbuf → files, on request)

- They are not read directly: they are requested with the **`WriteFiles`** RPC (`grpc/frb_search.proto`), giving beams and an `fpga_seq` range.
- pirate writes **one file per frame** in `{nfs_dir}/{acqdir}/frame_b(BEAM)_t(CHUNK).asdf`, with the 4-bit data, the scales and offsets and the metadata.
- The range may include **future** chunks: they are written as they arrive, and `SubscribeFiles` reports when each file is ready.
- If the range has already left the ring buffer, those files simply do not appear.
- Python clients: `pirate_frb/rpc/FrbSearchClient.py` and `FileSubscriber.py`.
- **Cost:** a burst at DM ~1000 needs ~17 chunks (~35 s) per beam, about 370 MB written before it can be classified.

---

## 4. What exists and what has to be built

| Piece | Status | Language | File |
|---|---|---|---|
| Reception, frames, frame ring buffer | exists (pirate) | C++ | `src_lib/`, `include/pirate/Receiver.hpp`, `AssembledFrame.hpp` |
| Tree dedispersion + per-cell peak-finding | exists (pirate) | C++/CUDA (kernels generated from Python) | `include/pirate/Dedisperser.hpp`, `pirate_frb/cuda_generator/` |
| MegaRingbuf (GPU/RAM split) | exists (pirate) | C++/CUDA | `include/pirate/MegaRingbuf.hpp` |
| Interface to read the map and decode events | exists (pirate) | Python over C++ | `pirate_frb/rpc/FrbGrouper.py` |
| Frame dumps (`WriteFiles`, `SubscribeFiles`) | exists (pirate); nobody calls it yet | gRPC + Python | `pirate_frb/rpc/FrbSearchClient.py` |
| Grouper | pirate has only a toy one (maximum per beam and chunk); **CHARTS: grouper v1** (regions in the map, grouping per burst) | Python/cupy | `charts/08_server/grouper/grouper_v1.py` |
| Sifter | pirate has only a toy one (prints the events); **CHARTS: alert sifter** (alert rule, RFI veto, `alerts.jsonl`) | Python | `charts/08_server/sifter/alert_sifter.py` |
| Classifier | **does not exist** (only the `rfi_prob` field) | — | — |
| Coordination between search nodes, trigger to the X-nodes | **does not exist** | — | — |

---

## 5. Where a classifier fits

| Stage | Input | Where it runs | Constraint |
|---|---|---|---|
| 1 · on the map | Window of `out_max` (+ `out_argmax`) of each tree around the event | Inside the grouper, on the GPU, every chunk | < 2.048 s per chunk for all beams; without resizing the map (cut-out at native resolution) |
| 2 · on raw data | 4-bit frames requested with `WriteFiles` → frequency-time and DM-time images | After the grouper or in the sifter, only for the survivors | Reading the ring buffer costs ~370 MB per candidate at DM ~1000 |

Real-time budget:
- X-node buffer: ~40 s;
- pirate's latency at DM ~1000: ~33 s;
- **~5 s** remain for grouper + classifier + trigger.

---

## Related files

- `docs/GUIDE_pirate_installation_usage_benchmark_CHARTS_2026-10-08.md`: installation, usage and benchmark.
- `notes/grouper_interface.md` (in pirate, the folder above `charts/`): pirate ↔ grouper interface (by the author).
- `notes/developer.md` (in pirate): definitions of chunk, batch, frame and segment (by the author).

## Appendix · Grouper v1, step by step

`08_server/grouper/grouper_v1.py`, explained with figures:

![Input: out_max map](img/fig_g1_map.png)
![Step 1: connected regions above S/N 5](img/fig_g2_regions.png)
![Step 2: maximum per region](img/fig_g3_maxima.png)
![Step 3: grouping by t_top](img/fig_g4_ttop.png)
![Step 3b: case across chunks](img/fig_g6_across_chunks.png)
![Step 4: output to the sifter, v1 versus v0](img/fig_g5_output.png)
