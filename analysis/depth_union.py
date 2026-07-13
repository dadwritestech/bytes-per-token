#!/usr/bin/env python3
# Cross-position (DEPTH) reuse: how does the per-layer union of routed experts grow across
# D CONSECUTIVE committed tokens? The committed greedy token at each step == the rank-0
# candidate, so consecutive steps are consecutive generated positions.
#
# This is the depth analogue of union_growth.py's breadth A(K). It quantifies how expensive
# a DEPTH-D accepted path is in expert I/O: A_depth(D) = (D*n_used) / union_over_D_positions.
# A_depth≈1  => positions barely share experts (depth churn) => a deep chain reads ~D*single
#              (expensive; matches the MTP failure and the "cache ~3.4GB/token" prior finding).
# A_depth>1  => consecutive positions reuse => cross-level cache hits shrink a tree's union_total,
#              improving the Thesis-A byte budget.
#
# usage: python depth_union.py <trace.csv> [more.csv ...]

import sys, csv
from collections import defaultdict

def load(paths):
    # per file: (layer) -> {step: set(eids)} using rank-0 (committed) rows only
    files = []
    n_used = None
    for p in paths:
        by_layer = defaultdict(dict)
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                if int(row['rank']) != 0:
                    continue
                il = int(row['layer']); step = int(row['step'])
                eids = set(int(x) for x in row['eids'].split())
                by_layer[il][step] = eids
                nu = int(row['n_used']); n_used = nu if n_used is None else n_used
        files.append(by_layer)
    return files, n_used

def main():
    if len(sys.argv) < 2:
        print("usage: depth_union.py <trace.csv> [more.csv ...]"); return
    files, n_used = load(sys.argv[1:])
    # max depth = min consecutive steps available
    maxD = min(len(next(iter(bl.values()))) for bl in files if bl)
    print(f"# files={len(files)} n_used={n_used} maxD(consecutive positions)={maxD}")
    print("\nD   union(D)  D*nused   A_depth(D)   reuse%")
    for D in range(1, maxD + 1):
        tot_union = 0.0; groups = 0
        for bl in files:
            for il, steps in bl.items():
                ks = sorted(steps)
                # slide a window of D consecutive positions
                for i in range(0, len(ks) - D + 1):
                    u = set()
                    for j in range(D):
                        u |= steps[ks[i + j]]
                    tot_union += len(u); groups += 1
        mu = tot_union / groups
        ideal = D * n_used
        A = ideal / mu
        reuse = 0.0 if D == 1 else (ideal - mu) / (ideal - n_used) * 100.0
        print(f"{D:<3d} {mu:8.2f}  {ideal:7d}   {A:8.3f}    {reuse:6.1f}")

if __name__ == '__main__':
    main()
