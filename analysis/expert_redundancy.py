"""Thesis D decisive experiment: do routed experts share compressible structure?

For sampled MoE layers of Qwopus3.6-35B (256 experts, Q6_K), dequantize the expert
tensors and measure, per (layer, proj):
  - mean/max |cosine| between expert weight vectors (flattened)
  - centered-gram eigenspectrum: energy captured by a shared rank-r subspace
    (base + rank-r delta is the proposed "cross-expert quant"; this bounds its win)
  - delta-entropy proxy: std(W_i - W_mean) / std(W_i)  ->  bits/weight saving
    potential = log2(1/ratio) under a Gaussian model
Baseline: 256 iid Gaussian experts of the same dimension (analytic: cos ~ 0,
rank-r energy ~ r/255, std ratio ~ 1).

Verdict rule (fixed before running): ALIVE if rank-64 shared energy >= 50%
or entropy saving >= 0.5 bits/weight. DEAD if it tracks the random baseline.
"""
import sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = "D:/Local/models/Qwopus3.6-35B-A3B-Coder-MTP-Q6_K.gguf"
LAYERS = [2, 20, 38]
PROJS = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
RANKS = [1, 8, 32, 64]

reader = GGUFReader(MODEL)
by_name = {t.name: t for t in reader.tensors}

print(f"{'tensor':28s} {'mean|cos|':>9s} {'max|cos|':>8s} "
      + " ".join(f"E@r{r:<3d}" for r in RANKS) + "  std_ratio  bits_saved")

results = []
for il in LAYERS:
    for proj in PROJS:
        name = f"blk.{il}.{proj}.weight"
        if name not in by_name:
            print(f"{name}: MISSING, skipping"); continue
        t = by_name[name]
        w = dequantize(t.data, t.tensor_type)          # numpy fp32, shape reversed ne
        w = w.reshape(256, -1).astype(np.float32)      # expert-major (slowest dim)
        n = w.shape[0]

        # cosine stats on normalized experts
        norms = np.linalg.norm(w, axis=1, keepdims=True)
        u = w / norms
        g = u @ u.T
        off = g[~np.eye(n, dtype=bool)]
        mean_cos, max_cos = np.abs(off).mean(), np.abs(off).max()

        # shared-subspace energy: eigenspectrum of centered gram
        c = w - w.mean(axis=0, keepdims=True)
        gram = c @ c.T
        ev = np.linalg.eigvalsh(gram)[::-1]
        ev = np.clip(ev, 0, None)
        tot = ev.sum()
        energy = {r: ev[:r].sum() / tot for r in RANKS}

        # delta entropy proxy
        std_orig = w.std(axis=1).mean()
        std_delta = c.std(axis=1).mean()
        ratio = std_delta / std_orig
        bits = np.log2(1.0 / ratio) if ratio > 0 else float("inf")

        print(f"blk.{il:02d}.{proj:20s} {mean_cos:9.4f} {max_cos:8.4f} "
              + " ".join(f"{energy[r]:5.1%}" for r in RANKS)
              + f"  {ratio:9.4f}  {bits:9.3f}")
        results.append((il, proj, mean_cos, max_cos, energy, ratio, bits))

# analytic random baseline for reference
d = 1  # printed generically; rank-r energy for iid is ~ r/(n-1)
print("\nrandom-orthogonal baseline: mean|cos|~0.006, E@r ~ r/255 "
      f"(r64 -> {64/255:.1%}), std_ratio ~ 1.0, bits ~ 0")

alive = any(e[4][64] >= 0.50 or e[6] >= 0.5 for e in results)
print(f"\nVERDICT: Thesis D {'ALIVE' if alive else 'DEAD'} "
      f"(rule: E@r64 >= 50% or bits_saved >= 0.5)")
