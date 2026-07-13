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
