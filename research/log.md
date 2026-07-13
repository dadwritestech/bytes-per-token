# Research log — a new inference engine for very large LLMs on consumer hardware

Target box: RTX 5080 16GB + RTX 5060 Ti 16GB, 31 GB RAM, WD_BLACK SN7100 NVMe
(~5.3 GiB/s deep-QD direct read). Reference model: GLM-5.2 UD-IQ4_XS, 754B, 365 GB,
256 experts, run at top-4. Quick-iteration model: Qwopus3.6-35B-A3B Q6_K (29 GB, top-8).

Governing equation:  `tokens/sec (decode) ≈ effective_bandwidth / bytes_touched_per_token`.

**Number to beat: 0.9 tok/s decode on GLM-5.2** (direct-read NVMe streamer, cold-start,
established by prior work in the llama.cpp `moe-tiering` oracle fork). At that point ~64%
of wall-clock/token is pure disk wait; ~0.36 s/token is CPU matmul + per-layer CPU↔GPU
sync + sampling.

Baseline reproduce command (oracle, for ground truth):
```
"D:/Local/llama build/llama.cpp/build/bin/Release/llama-cli.exe" \
  -m "D:/Local/models/GLM-5.2/UD-IQ4_XS/GLM-5.2-UD-IQ4_XS-00001-of-00009.gguf" \
  -cmoe --override-kv glm-dsa.expert_used_count=int:4 \
  -c 4096 -n 128 -p "<prompt>"   # + GGML_MOE_STREAM_GB=12 for the streamer path
```

---

## Inherited state (from oracle fork `research/2026-07-12-expert-tiering.md`) — do NOT rederive

Confirmed structural facts about GLM-5.2 / Qwopus MoE routing:
- Expert usage is strongly **Zipfian** globally (top ~1.7–10% of slots serve 25–50% of touches).
- **Consecutive decoded tokens do NOT share experts much** — warm cache stays ~3.4 GB/token;
  routing churns per token. (⇒ *depth/chain* reuse is weak.)
- **Neuron keep-rate under a d=0.5 activation mask is near-uniform ~0.5 with no block
  structure** — identical curve on GLM (754B IQ4) and Qwopus (35B Q6). This is a
  *fundamental* property of MoE FFN activations, not model-specific.

Dead ends already paid for (don't repeat):
- Dual-NVMe fault striping: −22% (SN570 random-fault latency).
- VRAM hot-expert tier at this box's ~6.5 GB free VRAM: neutral-to-negative (coverage
  overlaps the RAM pin; per-layer CPU↔GPU merge sync eats the win).
- Row-sparse FFN (drop low-|silu(gate)| up-rows): quality only holds near keep-density 0.5
  (+4.6% PPL), where realizable I/O saving is ~17% (up-only). d≤0.15 craters PPL (+24–38%).
- Static neuron reorder + prefix read: refuted (uniform 0.5 keep-rate, no contiguity to win).
- Sibling-fused expert bursts (gate/up/down in one read): −22% (locality beats queue depth
  on this NVMe).
- **MTP / lookahead speculative decode: net loss at 31 GB RAM** — the verify pass reads the
  *union* of experts across draft positions (~10× I/O) at ~0% cache hit. This failure is the
  motivation for Thesis A below, not a closed door.

Best known config for this box: `GGML_MOE_STREAM_GB=12` + top-4 = **pp 1.1 / tg 0.9**, cold.

---

## Thesis A — amortize weight I/O across a *breadth* of candidate tokens

**Claim:** we are I/O-bound because ~3.4 GB is read to produce ONE token. MTP failed by reading
the *union* of experts across a *deep* draft chain (where routing churns → no reuse). Invert it:
evaluate a *breadth* of candidate continuations that share the same context, so their per-layer
routed-expert sets **overlap**, and schedule the shared expert working set to be read **once**
and reused across all K candidates in one weighted pass.

**Why breadth might reuse where depth does not:** K candidate tokens at the *same* position
attend to the *same* prefix KV; only the last-token embedding differs. Early-layer hidden states
are then nearly identical across candidates → routing nearly identical; divergence grows with
depth. If the per-layer union over K candidates ≪ K × single-token set, one expensive read
serves K token-evaluations → up to K× effective throughput at the same bandwidth.

**Decisive first experiment (no engine):** over real generation contexts, for the top-K most
likely next tokens at a fixed context, dump each candidate's per-layer routed expert set and
measure `union(K)` vs `K × mean single-token set`. Qwopus first (fast, validated proxy), then
GLM. Verdict rule:
- `union(K) ≈ K × single` (overlap ≈ 0) ⇒ Thesis A is **dead** (record and move to B/C).
- `union(K) ≪ K × single` (high overlap) ⇒ Thesis A is **alive**; quantify the amortization
  factor `A(K) = (K × single) / union(K)` per layer and overall, then build the batched engine.

Status: **measuring** (see below).
