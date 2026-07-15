"""Does the Zipfian hot set hold across domains, or shift between prose/code/sci?

For each domain trace, count touches per (layer, expert) slot, take the top-p%
of slots by touch count (the "hot set" at several budgets), and measure the
pairwise overlap (Jaccard and directed coverage: what fraction of domain B's
touches does domain A's hot set serve?). If the hot set is domain-stable, a
pin computed on one domain transfers to another.
"""
import csv
from collections import Counter
from itertools import combinations

DOMAINS = {
    "prose": "D:/Local/new-inference-engine/traces/qwopus_prose.csv",
    "code":  "D:/Local/new-inference-engine/traces/qwopus_code.csv",
    "sci":   "D:/Local/new-inference-engine/traces/qwopus_sci.csv",
}
BUDGETS = [0.017, 0.05, 0.10]  # fraction of all (layer,expert) slots pinned

counts = {}
for name, path in DOMAINS.items():
    c = Counter()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            for eid in row["eids"].split():
                c[(layer, int(eid))] += 1
    counts[name] = c

n_slots = len(set().union(*[set(c) for c in counts.values()]))
print(f"slots touched across all domains: {n_slots}")

for budget in BUDGETS:
    k = max(1, int(round(budget * n_slots)))
    hot = {name: set(s for s, _ in c.most_common(k)) for name, c in counts.items()}
    # self-coverage: fraction of a domain's touches served by its own hot set
    print(f"\n--- budget {budget:.1%} of slots (k={k}) ---")
    for name, c in counts.items():
        tot = sum(c.values())
        self_cov = sum(c[s] for s in hot[name]) / tot
        print(f"{name}: self-coverage {self_cov:.1%}")
    for a, b in combinations(DOMAINS, 2):
        jac = len(hot[a] & hot[b]) / len(hot[a] | hot[b])
        # cross-coverage: B's touches served by A's hot set, and vice versa
        cov_ab = sum(counts[b][s] for s in hot[a]) / sum(counts[b].values())
        cov_ba = sum(counts[a][s] for s in hot[b]) / sum(counts[a].values())
        print(f"{a}<->{b}: jaccard {jac:.2f} | {a}-pin serves {cov_ab:.1%} of {b} "
              f"| {b}-pin serves {cov_ba:.1%} of {a}")
