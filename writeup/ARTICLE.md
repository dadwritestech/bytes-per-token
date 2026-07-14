# Three days against the wall: running a 754B MoE from NVMe until the physics said stop

I wanted GLM-5.2 — 754 billion parameters, 365 GB on disk at IQ4_XS — running at usable
speed on my desktop. The desktop has a 5080, a 5060 Ti, 31 GB of RAM and one fast NVMe.
That's 63 GB of fast memory for a 365 GB model. Every decoded token has to find its routed
expert weights somewhere, and for a model this size, "somewhere" is mostly the SSD.

This is the write-up of three very dense days: a real 4.5x that took the model from 0.2 to 0.9 tok/s,
then a series of measurements that killed every remaining idea I had, one by one, until
what was left was a proof that 0.9 is the floor — and a 21.6x that had been sitting in
plain sight the whole time. The dead ends are the useful part. I've kept them all.

Everything here is measured on the box above. The repo has the dated research log,
the analysis scripts, and the traces.

## One division governs everything

```
tokens/sec  ≈  effective_bandwidth / bytes_touched_per_token
```

At top-4 routing, one GLM token touches about 3.4 GB of expert weights. That figure is
*after* a 12 GB RAM cache does its best — consecutive tokens churn through different
experts so fast that the cache can't get traction. My drive sustains 5.7 GB/s at deep
queue depth. Divide: 1.55 tok/s, ceiling, before a single matmul runs.

I did not fully believe that division on day one. The rest of this article
is the process of coming to believe it.

## Part 1 — the 4.5x that was actually there

![decode-speed trajectory](figs/fig1_trajectory.png)

Four changes survived measurement, in a llama.cpp fork with experts on the CPU path and
attention on the GPUs:

**Prefetch** (0.2 → 0.3). The router picks experts a few microseconds before the matmul
needs them. Issuing reads at pick time instead of faulting lazily is worth 50% on its own.

**Top-4 routing** (0.3 → 0.6). The model ships top-8. Overriding to top-4 halves
bytes-per-token, and bytes are everything, so this is close to a clean 2x. Quality holds
up well enough for the workloads I care about.

**Zipf-guided pinning** (0.6 → 0.9). I profiled per-(layer, expert) routing counts and the
distribution turned out to be brutally Zipfian: on GLM, the top 1.7% of expert slots take
25% of all routings. Pinning the hot set into 8 GB of locked RAM is +50%.

**A real streamer** (robust 0.9). Replacing mmap page faults with NO_BUFFERING,
deep-queue-depth direct reads into a reactive LRU. Same tok/s as pinning, far more stable,
and it sustains 3.9 GiB/s during decode — the drive is genuinely near its limit.

Things I tried and reverted, with their prices: striping across a second NVMe (−22%; the
slower drive's random-fault latency poisons the whole pipeline), fusing gate/up/down reads
into one burst (−22%; locality beats queue depth), and MTP speculative decoding (a net
loss that foreshadows everything in Part 2).

## Part 2 — killing my own ideas

At 0.9 tok/s, 64% of each token is disk wait. The remaining 0.36 s — call it `t_fix` — is
CPU expert matmul plus per-layer CPU↔GPU sync. I had three ideas left and a rule inherited
from the start of the project: every idea gets a cheap decisive experiment before it gets
engine code. The rule turned out to be the most valuable thing in the repo.

### Speculative decoding, the whole family, is dead here

The seductive version: a token costs 3.4 GB of reads, so evaluate many candidate tokens
per read. I instrumented the fork to dump per-layer routed-expert sets for K candidate
continuations of the same context and measured how the union grows:

![breadth reuse A(K)](figs/fig2_reuse.png)

Sibling candidates genuinely share experts. The amortization factor A(K) grows like √K,
and — this was the finding that made the whole exercise feel worthwhile — the *same law* shows up on
two unrelated models (a 35B top-8 and the 754B top-4). Reuse across siblings is structure,
not accident.

Then I wrote down what a speculative tree actually has to beat, and the finding stopped
feeling so good. Greedy decode reads exactly the experts it uses. Greedy is I/O-optimal.
A tree verify that accepts `m` tokens wins only if its extra bytes cost less than the
fixed cost it amortizes:

```
(union_tree − union_greedy(m)) · t_byte  <  (m − 1) · t_fix

tolerance:  R* = 1 + t_fix/t_io = 1 + 0.36/0.71 ≈ 1.51
```

I measured real trees — 2×4, 4×2, 3×3, 2×6. Every shape reads 2.8–3.3x its accepted path.
Over budget by a factor of two, *at perfect acceptance*. The required accepted depth
exceeds the tree's own depth, which no draft model can deliver because it's impossible.
Acceptance never even enters the verdict; the byte budget is spent first. That closes MTP,
Medusa-style trees, and lookahead as a family for the disk-bound regime — and tells you
exactly when they return: when `t_io/t_fix` falls about 2x, i.e. when the weights mostly
live in fast memory. Which contradicts the premise.

### Predicting the working set from the prompt loses to a dumb cache

I was fairly confident in this one. Each conversation surely uses some stable expert
subset — detect it during prompt processing, pin it, generate from fast memory.

![coverage comparison](figs/fig3_coverage.png)

Wrong twice. First, the per-prompt working set is enormous and doesn't saturate: one
96-token generation touches 38% of all experts and is still climbing when it ends (45–67%
on the proxy model). No 31 GB pin can hold that. Second, the prompt's signal is real but
weak: a prompt-predicted pin covers 27–44% of generation-time expert touches, while the
reactive LRU the streamer already has covers 42–62% and sits nearly on top of the oracle
at every feasible budget. The cache you get for free is already at the frontier.
Prediction adds bytes it was going to cache anyway.

### The null result that actually annoyed me

One lever remained that greedy can't already do: shrink bytes-per-token itself with
lower-bit experts.

![quant quality vs bytes](figs/fig4_quant.png)

On a fat Q6 proxy model this works beautifully — Q3_K experts cost +1.7% perplexity,
inside the error bars, at ~2x fewer expert bytes. So I cleared 254 GB of disk and
downloaded Unsloth's UD-Q2_K_XL of the 754B: 0.70x the file size of my IQ4_XS baseline.

Identical 0.9 tok/s. The streamer's own byte accounting showed the same ~3.4 GB streamed
per token as before.

The explanation took a while to accept. Dynamic quants earn their quality by protecting
the *hot* experts — and the hot experts are precisely the bytes streamed on every token,
because being streamed often is what hot means. The 130 GB of savings sat in cold experts
I rarely read. Worse, the folk wisdom "keep hot experts high-precision, crush the cold
ones" is exactly backwards for I/O: effective streamed bits-per-weight is coverage-
weighted, so it's the hot experts you'd need to shrink, and that's precisely the quality
trade the good quants exist to refuse. File size is not the variable. Streamed
bytes-per-token is. My own projection had used the total-size ratio and it was wrong.

## Part 3 — the two walls

Could a sufficiently clever engine hit 10 tok/s — a 100 ms/token budget — with this model
on this box? Write down what it requires:

```
Wall 1 (bus):      3.4 GB/token × 10 tok/s = 34 GB/s sustained from storage.
                   Six times the drive. More than the PCIe 4.0 x4 slot carries at all.
Wall 2 (compute):  t_fix = 0.36 s/token of CPU expert matmul, thread-saturated at
                   4 cores. With the model magically ALL in RAM: 2.8 tok/s. Still.
```

![the two walls](figs/fig5_walls.png)

The byte-cutting levers are measured and spent — quant headroom 1.15x on a model that
ships at 3.88 bpw, activation sparsity refuted (uniform 0.5 keep-rate, no structure),
top-k already cut, speculation dead, residency dead. They don't stack; they compete for
the same 3.4 GB. And a perfect I/O engine still parks at Wall 2, because 754B-scale expert
matmuls on consumer cores cost what they cost. Breaking Wall 2 means expert matmuls in
VRAM, which means ~240 GB of VRAM, which means a different machine. Two independent
inequalities. Beat one and the other catches you.

## Part 4 — the 21.6x that was always available

The equation permits exactly one remaining move: change which tier the bytes live in.
On this box that means a model whose weights fit 32 GB VRAM + 31 GB RAM.

GLM-4.5-Air — 106B total, ~12B active, 47 GB at Q2_K_XL — fully resident, about 60% of
its experts packed into VRAM by llama.cpp's auto-fit, the rest committed to RAM, zero disk
in the decode loop:

| | GLM-5.2 754B, streamed | GLM-4.5-Air 106B, resident |
|---|---|---|
| decode | 0.90 tok/s | **19.5 tok/s** |
| prompt eval | 1.1 tok/s | 27–29 tok/s |
| cold load | >60 s | ~20 s |

Same GLM family. Same machine. Same day. The equation had been saying this since the
first division — 3.4 GB over a 5.7 GB/s pipe versus 5 GB of active weights over VRAM and
RAM bandwidth — and it took three days of measurements to stop arguing with it.

## What transfers

If you're doing MoE offload on consumer hardware, the portable results: measure streamed
bytes-per-token, never file size — a quality-preserving quant can shrink the file 30% and
change your speed not at all. A reactive LRU over expert slabs is effectively the caching
frontier; don't build predictors. Expert routing is Zipfian, breadth-correlated and
depth-uncorrelated — siblings share, consecutive tokens don't — and any amortization
scheme lives or dies on that distinction. And before building anything speculative,
compute `R* = 1 + t_fix/t_io` for your own setup; if your measured tree overhead can't
get under it, the byte budget is already spent.

The meta-lesson cost the least and was worth the most: every one of these verdicts came
from a cheap instrumented measurement, hours each. The engines they would have justified
were weeks each. The empty `engine/` directory in the repo is not the failure. It's the
receipt.

---

*RTX 5080 16 GB + RTX 5060 Ti 16 GB, Ryzen 9700X, 31 GB RAM, WD SN7100. Measurements via
a llama.cpp fork used as an oracle: routing traces from a scheduler eval callback,
direct-read expert streamer, profile-guided pinning. Research log with every dated verdict
and dead end is in the repo alongside this article.*
