#!/usr/bin/env python3
# Thesis A decisive analysis: how does the per-layer UNION of routed experts grow
# with candidate-tree width K (breadth of top-K next-token candidates at a shared context)?
#
# For each (step, layer) we have K candidate expert-sets (each of size n_used).
# union(K) = |set_0 ∪ ... ∪ set_{K-1}|, averaged over all (step, layer).
#
# Amortization factor  A(K) = (K * n_used) / union(K)
#   - Without sharing, K candidates would read K*n_used expert slabs.
#   - With sharing, one pass reads union(K) slabs and serves all K.
#   - A(K) ≈ 1  → no reuse → Thesis A DEAD.
#   - A(K) ≫ 1  → strong reuse → Thesis A ALIVE (≈ throughput multiplier ceiling).
#
# usage: python union_growth.py <trace.csv> [more.csv ...]

import sys, csv
from collections import defaultdict

def load(paths):
    # rows keyed by (file, step, layer) -> {rank: set(eids)}; n_used, n_layers
    data = defaultdict(dict)
    n_used = None
    layers = set()
    for pi, p in enumerate(paths):
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            for row in rd:
                step = int(row['step']); rank = int(row['rank']); il = int(row['layer'])
                nu = int(row['n_used'])
                eids = tuple(int(x) for x in row['eids'].split())
                n_used = nu if n_used is None else n_used
                layers.add(il)
                data[(pi, step, il)][rank] = set(eids)
    return data, n_used, sorted(layers)

def main():
    if len(sys.argv) < 2:
        print("usage: union_growth.py <trace.csv> [more.csv ...]"); return
    data, n_used, layers = load(sys.argv[1:])
    L = len(layers)
    # infer max K available (min ranks present across groups)
    maxK = min(len(rk) for rk in data.values())
    n_groups = len(data)
    print(f"# files={len(sys.argv)-1} layers={L} n_used={n_used} "
          f"maxK={maxK} groups(step*layer)={n_groups}")

    # depth buckets for a per-region view (early/mid/late thirds by layer index)
    def bucket(il):
        t = il / max(1, (L - 1))
        return 0 if t < 1/3 else (1 if t < 2/3 else 2)
    bnames = ["early", "mid  ", "late "]

    print("\nK   union(K)  K*nused   A(K)=amort   overlap%   |  A_early A_mid A_late")
    for K in range(1, maxK + 1):
        tot_union = 0.0
        b_union = [0.0, 0.0, 0.0]; b_cnt = [0, 0, 0]
        for (pi, step, il), rk in data.items():
            u = set()
            for r in range(K):
                u |= rk[r]
            tot_union += len(u)
            bb = bucket(il); b_union[bb] += len(u); b_cnt[bb] += 1
        mean_union = tot_union / n_groups
        ideal = K * n_used
        A = ideal / mean_union
        # overlap%: fraction of the non-shareable read avoided vs no-reuse
        # (K*nused - union) / (K*nused - nused) = redundancy captured beyond a single set
        overlap = 0.0 if K == 1 else (ideal - mean_union) / (ideal - n_used) * 100.0
        Ab = []
        for bi in range(3):
            mu = b_union[bi] / max(1, b_cnt[bi])
            Ab.append(ideal / mu if mu else 0.0)
        print(f"{K:<3d} {mean_union:8.2f}  {ideal:7d}   {A:8.3f}     {overlap:6.1f}    |  "
              f"{Ab[0]:6.2f} {Ab[1]:6.2f} {Ab[2]:6.2f}")

    print(f"\n({bnames[0].strip()}/{bnames[1].strip()}/{bnames[2].strip()} = layer thirds; "
          f"A>1 means one expert read serves >1 candidate)")

if __name__ == '__main__':
    main()
