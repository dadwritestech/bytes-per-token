# bytes-per-token

**An empirical study of running MoE LLMs far larger than RAM+VRAM on one consumer box —
what works, what provably can't, and the equation that decides.**

```
tokens/sec  ≈  effective_bandwidth / bytes_touched_per_token
```

We took GLM-5.2 (754B MoE, 365 GB) from 0.2 → 0.9 tok/s on a 2-GPU desktop by streaming
experts from NVMe — then drove every remaining software idea to a measured verdict and
proved 0.9 is the hardware floor for that (model, box) pair. The same box runs GLM-4.5-Air
at 19.5 tok/s the moment weights are resident. **The negative results are the product.**

![trajectory](writeup/figs/fig1_trajectory.png)

## Findings (each measured, each reproducible from this repo)

1. **Greedy decode is I/O-optimal.** Any speculative scheme (MTP, Medusa-style trees,
   lookahead) must beat a byte-overhead tolerance `R* = 1 + t_fix/t_io ≈ 1.5`; measured
   trees cost 2.8–3.3× their accepted path — dead at *perfect* acceptance. Closes the
   family for the disk-bound regime.
2. **Breadth reuse is real but too weak to pay.** K sibling candidates share experts with
   union growth ≈ √K (A(16) = 2.7–3.6 on two unrelated models) — a structural property of
   MoE routing, and still 2× short of what branching costs.
3. **A reactive LRU is at the caching frontier.** Prompt-predicted expert pinning loses to
   plain LRU at every feasible budget; per-prompt working sets (38–67% of all experts)
   can't fit consumer RAM anyway.
4. **Model file size ≠ streamed bytes.** A quality-preserving dynamic quant 0.70× the size
   streamed the *same* 3.4 GB/token → same 0.9 tok/s. Dynamic quants protect hot experts —
   exactly the bytes streamed every token. Uniform Q3_K experts (+1.7% PPL) is the real
   lever, if your model ships fat.
5. **The two walls.** 10 tok/s for the 754B needs 34 GB/s of storage (6× above PCIe 4.0 x4)
   *and* a ≤100 ms/token compute budget vs a measured 360 ms CPU-expert floor. Independent
   walls; software dodges neither. Residency — not engine cleverness — sets tok/s.

## Layout

- [`research/log.md`](research/log.md) — the dated research log: every thesis, measurement,
  and verdict, including the dead ends and what they cost.
- [`writeup/ARTICLE.md`](writeup/ARTICLE.md) — the narrative write-up with figures
  ([Reddit version](writeup/REDDIT.md)).
- [`analysis/`](analysis) — the analysis scripts (`union_growth`, `depth_union`, `tree_io`,
  `working_set`) that turn routing traces into the verdicts above.
- [`traces/`](traces) — sample routing traces and measurement logs.
- [`tools/`](tools) — benchmark scripts; the `route-trace` oracle instrument (a
  `ggml_backend_sched_eval_callback` that dumps per-layer routed-expert sets) lives in the
  companion llama.cpp fork, branch `moe-tiering`.
- [`docs/BRIEF.md`](docs/BRIEF.md) — the starting state: hardware, inherited baseline, and the three hypotheses as originally posed.

Hardware: RTX 5080 16 GB + RTX 5060 Ti 16 GB, 31 GB RAM, WD SN7100 NVMe (~5.3 GiB/s
direct reads). Models: GLM-5.2 754B (UD-IQ4_XS / UD-Q2_K_XL), Qwopus3.6-35B-A3B (proxy),
GLM-4.5-Air 106B.

**Method note:** every architectural bet got a cheap decisive experiment before any engine
code — which is why the `engine/` directory is empty and the log is full. A companion tool, `vramwise` — a
tok/s predictor calibrated from this study's measured anchors — is being prepared for
release separately.

MIT for code; the write-up text and figures are CC BY 4.0.
