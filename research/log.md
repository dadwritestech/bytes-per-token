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

Status: **first gate PASSED** (2026-07-13) — see result below.

### Thesis A — result: breadth expert-reuse is strong and model-general (2026-07-13)

Tooling: `llama-route-trace` (oracle example `examples/route-trace`, opt-in, zero model surgery).
It registers a `ggml_backend_sched_eval_callback` that captures the `ffn_moe_topk-<il>` tensor
(final global expert IDs, group-mask already applied). At each real generation step it seeds K
scratch KV sequences from the shared prefix (`seq_cp`) and decodes the top-K next-token candidates
as ONE batch (a real "verify batch") at the same position — exactly the Thesis-A workload.
`ffn_moe_topk` is a *strided top-k view* of a full `[n_expert, n_cand]` argsort — must index by
`nb[1]`, not `n_used` (this bug first showed as impossible perfect-disjoint sets; a same-token
control `RT_SAMETOK` that must give A(K)=K caught it). Analysis: `analysis/union_growth.py`.

Metric: `A(K) = (K·n_used) / mean_layer_union(K)` = expert-slab reads amortized across K candidate
evaluations. A(K)=1 ⇒ no reuse (dead); A(K)≫1 ⇒ strong reuse (alive).

**Qwopus3.6-35B (top-8, 256 exp), 3 prompts × 32 steps × WARM 8, K=16, 3840 (step,layer) groups:**

| K | union(K) | K·n_used | A(K) | overlap% |
|---|----------|----------|------|----------|
| 2 | 12.9 | 16 | 1.24 | 39% |
| 4 | 18.8 | 32 | 1.71 | 55% |
| 8 | 26.5 | 64 | 2.41 | 67% |
| 16 | 35.8 | 128 | **3.58** | 77% |

A(K) is **still climbing at K=16** (not saturated); mid-layers reuse most. Per-prompt A(16):
prose 3.95, science 3.59, code 3.01 — consistent.

**GLM-5.2 (top-4, 256 exp, group-routed, MLA), streamer 12GB, K=16, 10 steps × WARM 4,
750 groups:** A(4)=1.64, A(8)=2.13, **A(16)=2.69** (union 23.75/layer vs 64), overlap 67%,
still climbing; mid-layers reuse most (A_mid 2.86). Lower than Qwopus because top-4 has less
overlap headroom than top-8, but the same climbing law. Two very different models giving the same
shape ⇒ breadth reuse is a *structural* property of MoE routing, like the neuron keep-rate finding.

Per-level I/O model for the tree engine: a 16-wide sibling level on GLM costs ≈ union(16)/n_used =
23.75/4 ≈ **5.9× one token's expert bytes, not 16×**. That factor is what Thesis A2 plugs in.

**Verdict: Thesis A's first gate is PASSED — union(K) grows far sub-linearly (≈K^0.5).** A breadth
of candidate tokens sharing a context is I/O-cheap: one expert read serves ~3.6–4 candidate
evaluations at K=16, and more as K grows. This is the exact inverse of the MTP/depth failure
(routing churns *across positions* but is shared *across siblings at a position*).

### The honest caveat — breadth being cheap is necessary, not yet sufficient

A(K) counts reuse across candidate *evaluations*. Throughput is *accepted tokens per byte read*.
K alternatives at ONE position yield ≤1 accepted token, so same-position breadth alone is not a
win (reading union(16)=32 slabs to emit 1 token is 8× the greedy I/O). The win comes from using
cheap breadth to make **wide speculative trees** affordable: I/O per verify ≈ Σ over tree *levels*
of union(width) (siblings reuse within a level at A≈4; levels are ~independent, per the depth
result). Cheap width ⇒ explore wider ⇒ longer accepted path per verify ⇒ more tokens per unit I/O.

### The sharpened win condition (derived 2026-07-13) — and why it reframes Thesis A

Working the wall-clock math changes the target. Let `single` = one token's routed expert bytes,
`byte_time` = cold read time per expert byte, `t_fix` ≈ 0.36 s/token = the measured non-I/O cost
(per-layer CPU↔GPU sync + sampling). Greedy reads EXACTLY the experts it uses → **greedy is
I/O-optimal for expert bytes**; any speculation reads extra (rejected branches). So Thesis A
cannot win by cutting expert I/O.

Compare one tree-verify producing `m` accepted tokens against `m` greedy tokens:
```
m greedy   : m·(single·byte_time + t_fix)
one verify : union_total·byte_time + t_fix        (fixed cost paid ONCE for the batch)
WIN  ⟺  (union_total − m·single)·byte_time  <  (m−1)·t_fix
        \_____ extra expert bytes read _____/       \__ fixed cost saved __/
```
**The lever is amortizing `t_fix`, not reducing I/O.** Sibling reuse (the A(K) result) is what
keeps `union_total` from exploding with width, so the extra-bytes term stays smaller than the
`(m−1)·t_fix` saving. With `t_fix`≈0.36 s and `single·byte_time`≈0.71 s (64% of the 1.11 s/token),
the condition is roughly `(union_total/single − m) < 0.56·(m−1)` — i.e. the tree must read only a
little more than the accepted path. This favors **narrow, deep** trees with high acceptance, NOT
wide ones (wide inflates `union_total`). It is genuinely knife-edge → must be measured.

### Cross-position (depth) reuse is partial, not zero (2026-07-13, from existing traces)

`analysis/depth_union.py` unions the routed experts of D *consecutive committed* tokens
(the rank-0/greedy row at each step). This is the depth analogue of breadth A(K):

| D | GLM union(D) | GLM A_depth | reuse% | Qwopus A_depth | reuse% |
|---|------|------|-----|------|-----|
| 2 | 6.66 | 1.20 | 33% | 1.21 | 35% |
| 5 | 13.02 | 1.54 | 44% | 1.58 | 46% |
| 10| 22.53 | 1.78 | 48% | 1.97 | 55% |
| 32| — | — | — | 3.26 | 72% |

So a token's routed set is ~half-covered by the preceding ~9 tokens (GLM). This does NOT
contradict the inherited "cache ~3.4 GB/token" fact — steady-state cold-miss stays high, but
short-window overlap is real. **Consequence:** a tree's `union_total` is *below* the
levels-independent `Σ_levels union(width)` bound; cross-level cache hits (~48% on GLM) shrink it.

### Corrected win condition — count COLD-MISS bytes (both greedy and tree cache experts)

Greedy over m tokens reads `union_depth(m)` cold (each expert once, then cached), not `m·single`.
A tree-verify producing m accepted reads `union_tree(w,D)` cold. Extra I/O is only the *new*
experts the rejected siblings pull in (mitigated by breadth reuse). Corrected:
```
WIN  ⟺  (union_tree(w,D) − union_depth(m))·byte_time  <  (m−1)·t_fix
```
Because cross-level reuse pulls `union_tree` well under the pessimistic `D·union(w)`, the byte
budget is looser than the earlier estimate — Thesis A is back to **genuinely uncertain**, so the
measurement is worth doing. Both `union_tree(w,D)` and `E[m]` must be measured (below).

### Next decisive experiment — Thesis A2 (tree acceptance vs union-I/O)

Two measured quantities feed the condition above, as a function of tree shape (width w, depth D):
- `union_total(tree)` — I/O cost. Measurable NOW by extending route-trace to expand a real
  w-ary, depth-D tree (each node → its own KV seq from the parent) and unioning all nodes'
  captured expert sets. (Cheap on Qwopus.)
- `E[m]` — accepted tokens per verify. **Requires a real draft** (target's own top-w is circular:
  the true token is the argmax, so α≡1). Use the shipped **MTP head** (`blk.*.nextn`) as the draft;
  measure acceptance of an MTP-drafted tree against the target's greedy continuation. This is the
  substantial build; do it only after `union_total` confirms the byte budget is plausible.

Build the batched tree-verify engine only if measured `(union_total, m)` clears the win condition
AND the projected tok/s beats 0.9. Note: prior MTP failure was a width-1 chain (`m ≤ D`, and byte
term `= D·single` with zero fixed-cost amortization modelled) — the tree + fixed-cost framing is
the new, untested angle.

### Thesis A2 RESULT — tree-verify is byte-dead on this box, even with a perfect draft (2026-07-13)

Measured the I/O half directly: `route-trace` TREE mode expands a real w-ary depth-D tree (each
node its own KV seq via `seq_cp` from its parent) and dumps every node's routed experts;
`analysis/tree_io.py` computes `tree_slabs = Σ_layers |∪ all nodes|` vs the accepted root→leaf
`path_slabs`, and projects tok/s with the measured `io_tok=0.71s`, `t_fix=0.36s` (greedy≈0.93).

| tree (w×D) | model | nodes | R=tree/path | tok/s @ **perfect** accept (m=D) | verdict |
|---|---|---|---|---|---|
| 2×4 | GLM-5.2 | 30 | 3.30 | 0.54 (need m>6.9, max 4) | LOSE |
| 4×2 | Qwopus | 20 | 2.88 | 0.53 (need m>3.5, max 2) | LOSE |
| 3×3 | Qwopus | 39 | 2.84 | 0.61 (need m>4.6, max 3) | LOSE |
| 2×6 | Qwopus | 126 | 3.01 | 0.77 (need m>7.3, max 6) | LOSE |

**Every shape loses even at m=D (perfect acceptance).** The required accepted length always
*exceeds the tree's own depth* — i.e. reading the tree once already costs more wall-clock than
greedy takes to emit D tokens, so no draft quality can rescue it. Acceptance never enters the
verdict; the byte budget is blown first. (⇒ the expensive MTP-draft build was correctly skipped.)

**Root cause (fundamental, not a tuning miss):** greedy is I/O-optimal (reads only experts it
uses). The only lever is amortizing `t_fix`, but the tolerance is `R* = 1 + t_fix/io_tok ≈ 1.51`.
MoE breadth reuse is only ~2–3× (not the ~w× that would make width free), so any branching that
buys acceptance depth costs R≈2.6–3.3 in I/O — 2× over budget. Sibling reuse is real but too weak.

**VERDICT: Thesis A (batched speculative tree to amortize expert I/O) is DEAD on this box** for the
I/O-bound regime. It would only turn positive if `io_tok/t_fix` fell by ~2× — i.e. expert reads got
~4× cheaper (model largely resident in RAM / much faster storage), which contradicts the premise
(model ≫ RAM). Redeems the MTP negative with a general reason, and closes the whole speculative-
amortization family (breadth *and* depth). Tools (`route-trace`, `union_growth`, `depth_union`,
`tree_io`) kept.

### Where next (post-Thesis-A)

The measurements establish two hard facts: (1) greedy already sits near the I/O ceiling (≈64% disk,
streamer optimal); (2) the residual 36% is `t_fix` = per-layer CPU↔GPU sync + sampling. Remaining
honest levers, in KICKOFF terms:
- **Thesis B — prompt-conditioned resident working set** (highest unexplored ceiling): is the expert
  set *per-conversation* small and predictable from the prompt pass? If so, pin it to RAM+VRAM and
  run generation mostly from fast memory. Decisive measurement: per-prompt working-set size +
  prompt→generation expert-overlap. (Note: this reduces `io_tok` itself — the only way, per the A2
  analysis, that anything here improves.)
- **Cut `t_fix`** — profile the 0.36 s/token per-layer sync/sampling; it's the only software lever
  once I/O is optimal, and it speeds up *greedy directly* (halving it ≈ +19% tok/s).
- **Thesis C — adaptive-precision cold experts** (IQ2/IQ3 for rare experts) also reduces `io_tok`.

### Thesis B RESULT — predictive prompt-pin loses to the reactive streamer (2026-07-13)

Tooling: `route-trace` WS mode (`RT_WS_GEN`) captures routing for prompt tokens (phase 0) + a
greedy generation (phase 1); `analysis/working_set.py` computes working-set size, saturation, and
touch-coverage at a fixed pin budget H experts/layer for LRU (what the streamer does) vs predictive
(pin prompt top-H) vs hybrid vs oracle (static top-H by gen freq) vs random. LRU sim validated
against hand cases.

Two decisive facts (Qwopus 128-gen prose/code + GLM 96-gen prose):
1. **The per-prompt working set is large and does NOT saturate.** GLM: generation touches 38% of
   experts/layer (98/256) in 96 tokens, still climbing (4→19→51→98). Qwopus prose 45%, code 67%.
   At ~120 GB of experts that is far beyond the ~12 GB RAM a pin could use — it cannot be resident.
2. **Predictive < reactive at every feasible budget.** Coverage of gen touches (GLM):

   | H/layer | LRU (streamer) | predictive (prompt) | oracle | random |
   |---|---|---|---|---|
   | 8  | **42.1%** | 27.0% | 41.8% | 2.9% |
   | 16 | **52.6%** | 32.7% | 56.0% | 5.6% |
   | 32 | **62.1%** | 43.8% | 73.0% | 12.3% |

   The prompt carries real signal (27% ≫ 3% random) but **less** than the LRU's reactive
   convergence to the generation's own hot set. Ranking is oracle ≥ LRU ≥ predictive ≫ random.
   Hybrid (pin prompt-half + LRU rest) ≈ LRU (no gain at feasible H≈8). At the RAM-feasible budget
   even the oracle covers only ~42% → ~58% streams from disk no matter the strategy.

**VERDICT: Thesis B (prompt-conditioned resident working set) is DEAD.** The deployed reactive LRU
streamer is already at/above what any prompt-predicted static pin achieves, and per-prompt working
sets are too large to pin in 31 GB anyway. Matches the deep A2 reason: the streamer already sits at
the I/O frontier; predicting the hot set earlier doesn't add bytes it isn't already caching. This
also subsumes the inherited "pin ≈ streamer (both 0.9)" result with a coverage-level explanation.

### Now the only unrefuted lever left: cut bytes-per-expert (Thesis C)

A/B and A2 leave exactly one software lever that reduces `io_tok` without needing a resident set:
**store/stream the Zipfian-cold experts at lower precision** (hot IQ4, cold IQ2/IQ3). The cold
tail dominates streamed bytes (it's the cache misses), so halving its precision ≈ halves the
dominant I/O term — a direct `bytes_touched_per_token` cut, the one term greedy can't optimize away.
Decisive first experiment (quality, no engine): perplexity floor of low-bit experts — whole model,
then hot/cold split — before building the mixed-precision store. MIND the CUDA 13.2 IQ2/IQ3 sm_120
kernel miscompile: validate low-bit expert matmuls on the CPU reference. This is the next measurement.

### Thesis C — I/O ceiling confirmed large (2026-07-13, `llama-quantize --dry-run`)

Overriding only the expert tensors (`ffn_{gate,up,down}_exps`) to low-bit, Qwopus total size:

| experts at | total size | BPW | vs Q6_K experts (bytes) |
|---|---|---|---|
| Q6_K (baseline) | 27.8 GB | 6.58 | 1.0× |
| IQ3_XXS | 14.1 GB | 3.32 | ~2× fewer expert bytes |
| IQ2_XXS | 10.1 GB | 2.39 | ~3× fewer expert bytes |

Streamed bytes ARE the cold experts, so this is a near-direct `bytes_touched_per_token` cut ⇒ up to
~2× (IQ3) / ~3× (IQ2) tok/s **iff quality holds**. On GLM the target, IQ3 experts ⇒ ~2× fewer
streamed bytes ⇒ projected ~1.8 tok/s (2× the 0.9 baseline). Unlike A/B this does not fight the
streamer — it shrinks each slab. The whole bet now reduces to ONE quality number.

**Quality pipeline (next):** build `llama-imatrix`; compute an importance matrix on a calibration
corpus (IQ2_XXS needs it — a no-imatrix requant would understate C); produce Qwopus expert-IQ3 and
expert-IQ2 variants; run `llama-perplexity` (experts CPU via `-cmoe`, dodging the sm_120 low-bit
kernel trap) for Q6_K vs IQ3 vs IQ2 on a fixed corpus (wikitext-2). Then the hot/cold split
(hot experts IQ4, only the Zipf-cold tail IQ2) — expected to recover most of any IQ2 quality loss
while keeping most of the byte saving, since cold experts are rarely routed. Verdict rule: if
expert-IQ3 holds PPL within a few % it is an immediate ~2× lever; if only the hot/cold split holds,
quantify the byte/quality trade and size the split from the existing Zipf profile.

### Thesis C RESULT — low-bit experts hold quality; C SURVIVES (2026-07-13)

k-quants (Q2_K/Q3_K, not the sm_120-trapped i-quants) let me test without an imatrix. Requant
Qwopus experts only (`--tensor-type ffn_*_exps=...`), perplexity on PTB (24×512, experts on GPU —
k-quants compile fine on Blackwell):

| Qwopus experts | size | BPW | PPL | ΔPPL |
|---|---|---|---|---|
| Q6_K (baseline) | 29.2 GB | 6.58 | 10.93 ± 0.36 | — |
| **Q3_K** | 16.3 GB | 3.67 | **11.12 ± 0.37** | **+1.7% (within error bars)** |
| Q2_K | 12.3 GB | 2.92 | 11.87 ± 0.40 | +8.6% |

**Q3_K experts are near-lossless at ~2× fewer expert bytes.** This is the FIRST surviving thesis:
a direct `bytes_touched_per_token` cut that the streamer can't already do. Q2_K trades +8.6% PPL for
~2.5× — the hot/cold split (hot Q4, only Zipf-cold tail Q2) should recover most of that.

**Caveat — magnitude won't transfer 1:1 to GLM.** Qwopus started at Q6_K (6.6 bpw = big headroom);
GLM is ALREADY UD-IQ4_XS (~4.25 bpw). GLM headroom: IQ4→~Q3 (~1.4× fewer bytes ⇒ ~1.2 tok/s) or
IQ4→~Q2 (~1.8× ⇒ ~1.6 tok/s) — still beats 0.9, but GLM needs its own PPL measurement, and being
"Unsloth Dynamic" (already hot/cold mixed) some win may be baked in. (PTB is small/noisy, ±0.36;
tighten with more chunks / wikitext before final GLM numbers.)

**Status: Thesis C ALIVE — the one lever that works.** Next: (1) GLM expert-Q3/Q2 PPL on the target
(needs a GLM requant, or imatrix i-quants); (2) hot/cold split sized from the Zipf profile to push
past IQ4 at held quality; (3) then the mixed-precision streamer + measured tok/s A/B vs 0.9.
