# new-inference-engine

A clean-sheet inference engine for running LLMs far larger than RAM+VRAM (e.g. GLM-5.2,
754B, 365 GB) at usable interactive speed on a single consumer box, by streaming MoE expert
weights from NVMe and amortizing that I/O across useful work.

- `research/log.md` — dated research log; measurements, verdicts, negative results (first-class).
- `analysis/` — Python analysis of routing/expert traces.
- `prompts/` — fixed calibration prompts.
- `tools/` — measurement probes (may link the llama.cpp oracle as a library).
- `engine/` — the engine itself (built only around theses that survive measurement).

The llama.cpp fork at `D:/Local/llama build/llama.cpp` (branch `moe-tiering`) is used as a
**measurement oracle only** (reference logits, routing traces, baseline speed) — not extended
into the product. See `KICKOFF.md` for the full brief.

**Prime directive:** measure before you build. Every architectural bet gets a cheap decisive
experiment first; only survivors get engine code.
