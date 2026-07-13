#!/usr/bin/env python3
# Thesis A2 (I/O half): given a real w-ary depth-D tree (route-trace TREE mode), compute the
# total distinct expert-slabs a single batched verify must read (union over ALL nodes, per layer)
# vs the slabs of one accepted root-to-leaf PATH. Their ratio R = tree_slabs / path_slabs is the
# I/O inflation of exploring the tree over just walking the accepted path.
#
# Win condition (derived in research/log.md), perfect acceptance m=D:
#   (R-1)*path_io < (D-1)*t_fix,  with path_io ≈ D*io_tok and io_tok/t_fix ≈ 0.71/0.36 ≈ 1.97
#   ⇒ large-D threshold:  R < 1 + t_fix/io_tok ≈ 1 + 0.36/0.71 ≈ 1.51
# So a narrow tree can only win if it reads < ~1.5x its accepted path's slabs, AND acceptance is
# near-perfect. R above that ⇒ byte-dead regardless of the draft.
#
# usage: python tree_io.py <tree.csv> [io_tok_over_t_fix]

import sys, csv
from collections import defaultdict

def main():
    path = sys.argv[1]
    ratio_io_fix = float(sys.argv[2]) if len(sys.argv) > 2 else (0.71/0.36)
    parent = {}; depth = {}
    sets = defaultdict(dict)   # node -> {layer: set}
    layers = set(); n_used = None
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            nd = int(row['node']); parent[nd] = int(row['parent']); depth[nd] = int(row['depth'])
            il = int(row['layer']); eids = set(int(x) for x in row['eids'].split())
            sets[nd][il] = eids; layers.add(il)
            nu = int(row['n_used']); n_used = nu if n_used is None else n_used
    L = len(layers); nodes = sorted(sets)
    N = len(nodes)
    D = max(depth.values()) + 1  # depths are 0-based

    # union over ALL tree nodes, per layer, summed
    tree_slabs = 0
    for il in layers:
        u = set()
        for nd in nodes:
            u |= sets[nd].get(il, set())
        tree_slabs += len(u)

    # leaves = nodes that are nobody's parent
    parents_set = set(parent.values())
    leaves = [nd for nd in nodes if nd not in parents_set]
    # path slabs for each leaf: union over nodes on root->leaf path, per layer, summed
    def path_of(leaf):
        p = []; x = leaf
        while x in parent and x != 0:
            p.append(x); x = parent[x]
        return p
    path_slabs_list = []
    for lf in leaves:
        p = path_of(lf)
        s = 0
        for il in layers:
            u = set()
            for nd in p:
                u |= sets[nd].get(il, set())
            s += len(u)
        path_slabs_list.append((len(p), s))
    mean_path_slabs = sum(s for _, s in path_slabs_list) / len(path_slabs_list)
    mean_path_len   = sum(m for m, _ in path_slabs_list) / len(path_slabs_list)

    single_tok_slabs = n_used * L
    R = tree_slabs / mean_path_slabs
    thresh = 1.0 + 1.0 / ratio_io_fix

    print(f"# {path}")
    print(f"nodes={N} depth(D)={D} layers={L} n_used={n_used} leaves={len(leaves)}")
    print(f"single-token slabs (n_used*L)      = {single_tok_slabs}")
    print(f"accepted-path slabs (mean, len={mean_path_len:.1f}) = {mean_path_slabs:.0f}")
    print(f"whole-tree slabs (union all nodes) = {tree_slabs}")
    print(f"R = tree/path                      = {R:.3f}")
    print(f"win threshold R* (perfect accept)  = {thresh:.3f}   ({'PASS' if R < thresh else 'FAIL'})")
    print(f"  path/single-token = {mean_path_slabs/single_tok_slabs:.2f}x  "
          f"(intra-path depth reuse: {(1-mean_path_slabs/(mean_path_len*single_tok_slabs))*100:.0f}% saved vs D independent)")

    # absolute tok/s projection on this box (measured greedy: io_tok=0.71s, t_fix=0.36s => ~0.93 tok/s)
    io_tok = 0.71; t_fix = 0.36
    t_io_slab = io_tok / single_tok_slabs
    greedy_tps = 1.0 / (io_tok + t_fix)
    verify_time = tree_slabs * t_io_slab + t_fix     # read whole tree once, pay fixed once
    m = D                                            # PERFECT acceptance (upper bound)
    tree_tps = m / verify_time
    print(f"projected tok/s (io_tok={io_tok},t_fix={t_fix}): greedy={greedy_tps:.2f}  "
          f"tree@perfect-accept(m={m})={tree_tps:.2f}   "
          f"{'WIN' if tree_tps>greedy_tps else 'LOSE'}  (need m>{verify_time*greedy_tps:.1f} tokens; max={D})")

if __name__ == '__main__':
    main()
