# Project brief — starting state and hypotheses

This documents where the project started: the hardware, the baseline inherited from
prior work, and the three hypotheses the study set out to test. The dated verdicts live
in [`research/log.md`](../research/log.md).

## Goal

Run LLMs far larger than RAM+VRAM on a single consumer machine at usable interactive
speed — concretely, beat the inherited 0.9 tok/s decode on GLM-5.2 off NVMe at
comparable quality, with every claim an A/B on the same box and the command recorded.

## Method rule

Every architectural bet gets a cheap, decisive experiment before any engine code:
state the thesis and the number that would confirm or kill it, get that number the
cheapest way possible (instrumented oracle, trace analysis, simulation), and only build
around survivors. Negative results are recorded as first-class findings.

## The box

- GPU0: RTX 5080, 16 GB, PCIe x16; GPU1: RTX 5060 Ti, 16 GB on a slower chipset link
  (~3.0 GiB/s host→device).
- RAM: 31 GB — the binding constraint; the target model is ~10× this.
- Disk D: WD_BLACK SN7100 1 TB. Measured ~5.3 GiB/s with `FILE_FLAG_NO_BUFFERING` +
  overlapped reads at QD8×8 MB; ~2.6 GiB/s buffered single-stream.
- Disk E: WD SN570 (DRAM-less), ~3.2 GiB/s sequential, poor random-fault latency.
- Windows 11, CUDA 13.2, Blackwell sm_120. Known trap: CUDA 13.2 miscompiles
  IQ1/IQ2/IQ3-family dequant kernels on sm_120 — validate low-bit GPU kernels against a
  CPU reference.

## The governing equation

    tokens/sec (decode) ≈ effective_memory_bandwidth / bytes_touched_per_token

For a disk-streamed MoE, bytes/token is dominated by routed expert weights. Everything
in this study is a fight over one of the two terms.

## Reference models

- **GLM-5.2** (753.86B, 256 experts, run at top-4 via `expert_used_count` override),
  UD-IQ4_XS, 365 GB in 9 splits. ~317 GiB is experts; ~18.6 GiB non-expert fits on the
  GPUs. MLA attention; ships an MTP head. Warm-cache cold-miss traffic ≈ 3.4 GB/token.
- **Qwopus3.6-35B-A3B** (256 experts, top-8, Q6_K, ~29 GB) — the fast-iteration proxy.
- A private llama.cpp fork (branch `moe-tiering`) used strictly as a measurement oracle:
  reference logits, a direct-read expert streamer, routing/activation profilers.

## Inherited baseline (measured on this box before the study)

Decode ladder on GLM-5.2: stock 0.2 → expert prefetch 0.3 → top-4 routing 0.6 →
hot-expert RAM pin 0.9 → direct-read NVMe streamer 0.9–1.1 (pp 1.1 / tg 0.9 cold).
At 0.9, ~64% of wall-clock is disk wait; ~0.36 s/token is CPU matmul + per-layer sync.

Dead ends already paid for, with reasons:
- Dual-NVMe fault striping: −22% (SN570 random-fault latency).
- VRAM hot-expert tier: neutral-to-negative at ~6.5 GB free VRAM.
- Row-sparse FFN: quality holds only near keep-density 0.5, where the I/O saving is ~17%.
- Static neuron reorder + prefix read: refuted — keep-rate is uniform ~0.5, no block
  structure to exploit.
- Sibling-fused expert bursts: −22% (locality beats queue depth on this drive).
- MTP/lookahead speculative decode: net loss — the verify pass reads the union of
  experts across draft positions at ~0% cache hit. Treated as a clue, not a closed door
  (see Thesis A).
- Structural facts: expert usage is strongly Zipfian; consecutive tokens share few
  experts (routing churns per token).

## The three hypotheses

**Thesis A — amortize weight I/O across a breadth of candidate tokens.** MTP failed
reading the union across *depth*; candidates at the *same* position share a prefix and
may share routing. Decisive experiment: measure how the per-layer union of routed
experts grows with candidate-set width K. Union ≪ K× ⇒ alive; ~linear ⇒ dead.

**Thesis B — prompt-conditioned resident working set.** Is Zipfian usage per-prompt
small and predictable? Decisive experiment: working-set size and prompt→generation
expert-overlap curves.

**Thesis C — adaptive-precision cold experts.** Hot experts at IQ4, cold at IQ2/IQ3 —
fewer bytes/token by exploiting the Zipf. Decisive experiment: perplexity floor of
low-bit experts before building any mixed-precision store.

All three were driven to measured verdicts; see the log for what survived (spoiler:
the interesting part is why the survivors still weren't enough).
