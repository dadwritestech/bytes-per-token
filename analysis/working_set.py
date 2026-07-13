#!/usr/bin/env python3
# Thesis B: is a conversation's expert working set small, and PREDICTABLE from the prompt pass?
#
# From a WS trace (route-trace RT_WS_GEN mode: phase 0 = prompt tokens, phase 1 = generation),
# per layer we get prompt touch-counts and generation touch-counts per expert.
#
# Questions:
#  1. Working-set size: how many distinct experts does generation touch per layer (vs n_expert)?
#  2. Saturation: does cumulative distinct-expert count plateau over the generation?
#  3. PREDICTABILITY: at a fixed pin budget of H experts/layer, what fraction of generation's
#     expert TOUCHES is covered by
#       (a) predictive  = top-H experts by PROMPT count       (select during prefill), vs
#       (b) oracle      = top-H experts by GENERATION count    (best reactive, upper bound), vs
#       (c) random      = H random experts                     (no structure baseline).
#     If predictive ≈ oracle and ≫ random, prompt-conditioned pinning is viable.
#
# usage: python working_set.py <ws_trace.csv>

import sys, csv
from collections import defaultdict, Counter

def main():
    path = sys.argv[1]
    prompt = defaultdict(Counter)     # layer -> Counter(expert -> prompt touches)
    gen    = defaultdict(Counter)     # layer -> Counter(expert -> gen touches)
    gen_tokens = defaultdict(list)    # layer -> list of expert-sets, in generation order
    n_expert_seen = set(); n_used = None
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            ph = int(row['phase']); il = int(row['layer'])
            eids = [int(x) for x in row['eids'].split()]
            n_used = len(eids) if n_used is None else n_used
            n_expert_seen.update(eids)
            if ph == 0:
                prompt[il].update(eids)
            else:
                gen[il].update(eids)
                gen_tokens[il].append(set(eids))
    layers = sorted(gen)
    L = len(layers)
    maxexp = max(n_expert_seen) + 1

    # 1. working-set size
    ws_sizes = [len(gen[il]) for il in layers]
    mean_ws = sum(ws_sizes)/L
    # generation length (tokens) = touches/n_used for a layer
    glen = sum(gen[layers[0]].values())//n_used
    print(f"# {path}")
    print(f"layers={L} n_used={n_used} experts~={maxexp} gen_tokens={glen}")
    print(f"gen working-set size/layer: mean={mean_ws:.1f}/{maxexp} "
          f"({100*mean_ws/maxexp:.0f}%), min={min(ws_sizes)}, max={max(ws_sizes)}")

    # 2. saturation: cumulative distinct experts after k gen tokens (mean over layers)
    print("\nsaturation (mean distinct experts touched in first k gen tokens):")
    for k in [1,2,4,8,16,32,64,glen]:
        if k > glen: continue
        tot = 0
        for il in layers:
            u = set()
            for s in gen_tokens[il][:k]: u |= s
            tot += len(u)
        print(f"  k={k:<4d} distinct/layer={tot/L:6.1f}")

    # 3. predictability at fixed budget H experts/layer.
    #    LRU = what the streamer already does (reactive). predictive = Thesis B (pin prompt top-H).
    #    hybrid = pin prompt top-H, LRU the remaining budget. Compares against oracle upper bound.
    total_gen_touches = sum(sum(gen[il].values()) for il in layers)
    def lru_hits(seq, H, pinned=frozenset()):
        # returns hits over the token-set sequence with an LRU cache of size H (pinned held fixed)
        from collections import OrderedDict
        cache = OrderedDict(); hits = 0; tot = 0
        cap = max(0, H - len(pinned))
        for s in seq:
            for e in s:
                tot += 1
                if e in pinned or e in cache:
                    hits += 1
                    if e in cache: cache.move_to_end(e)
                else:
                    cache[e] = 1
                    if len(cache) > cap and cap > 0: cache.popitem(last=False)
                    elif cap == 0: cache.clear()
        return hits, tot
    print("\ncoverage of generation touches at pin budget H experts/layer:")
    print("H     LRU(reactive=streamer)  predictive(prompt-pin)  hybrid(pin+LRU)  oracle   random")
    import random as _r; _r.seed(0)
    for H in [4,8,16,32,64]:
        cov_lru = cov_pred = cov_hyb = cov_orc = cov_rnd = 0
        for il in layers:
            pred = set([e for e,_ in prompt[il].most_common(H)])
            orc  = set([e for e,_ in gen[il].most_common(H)])
            allexp = list(range(maxexp)); _r.shuffle(allexp); rnd = set(allexp[:H])
            for e,c in gen[il].items():
                if e in pred: cov_pred += c
                if e in orc:  cov_orc  += c
                if e in rnd:  cov_rnd  += c
            h,_ = lru_hits(gen_tokens[il], H);                     cov_lru += h
            # hybrid: pin half the budget to prompt top-(H/2), LRU the rest
            phalf = set([e for e,_ in prompt[il].most_common(H//2)]) if H>=2 else set()
            h2,_ = lru_hits(gen_tokens[il], H, pinned=phalf);      cov_hyb += h2
        f = 100.0/total_gen_touches
        print(f"{H:<5d} {cov_lru*f:6.1f}%                {cov_pred*f:6.1f}%                "
              f"{cov_hyb*f:6.1f}%          {cov_orc*f:6.1f}%   {cov_rnd*f:5.1f}%")

if __name__ == '__main__':
    main()
