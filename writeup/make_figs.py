# Generates the figures for writeup/ARTICLE.md from the measured numbers in research/log.md
# and the oracle fork's research notes. Palette: dataviz reference (light mode).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
BLUE_250, BLUE_350 = "#86b6ef", "#5598e7"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

def style(ax, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(length=0)

# ---- Fig 1: decode-speed trajectory ------------------------------------------------
stages = ["stock\nllama.cpp", "+ expert\nprefetch", "+ top-4\nrouting",
          "+ Zipf pin +\nNVMe streamer", "+ requant\n(UD-Q2_K_XL)", "GLM-4.5-Air\nresident"]
tps = [0.2, 0.3, 0.6, 0.9, 0.9, 19.5]
colors = [BLUE]*5 + [AQUA]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
bars = ax.bar(stages, tps, width=0.62, color=colors)
for b, v in zip(bars, tps):
    ax.text(b.get_x()+b.get_width()/2, v+0.35, f"{v:g}", ha="center",
            color=INK, fontsize=11, fontweight="bold")
ax.set_ylabel("decode tokens / s")
ax.set_ylim(0, 21.5)
ax.set_title("Same box. The first five bars are GLM-5.2 754B streamed from NVMe;\nthe last is GLM-4.5-Air 106B fully resident in VRAM + RAM.",
             loc="left", fontsize=10.5, color=INK2)
style(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_trajectory.png"); plt.close(fig)

# ---- Fig 2: breadth expert-reuse A(K) is sublinear ---------------------------------
K = np.array([2, 4, 8, 16])
qwopus = [1.24, 1.71, 2.41, 3.58]
glm_k, glm = np.array([4, 8, 16]), [1.64, 2.13, 2.69]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(K, K, "--", color=MUTED, linewidth=1.4)
ax.text(11.1, 12.6, "perfect reuse  A(K) = K", color=MUTED, fontsize=9, rotation=38)
ax.axhline(1, linestyle=":", color=MUTED, linewidth=1.2)
ax.text(2.05, 1.35, "no reuse  A(K) = 1", color=MUTED, fontsize=9)
ax.plot(K, qwopus, "-o", color=BLUE, linewidth=2, markersize=7)
ax.plot(glm_k, glm, "-o", color=AQUA, linewidth=2, markersize=7)
ax.text(16.2, 3.58, "Qwopus-35B (top-8)", color=BLUE, fontsize=10, va="center")
ax.text(16.2, 2.55, "GLM-5.2 754B (top-4)", color="#128a60", fontsize=10, va="center")
ax.set_xscale("log", base=2); ax.set_xticks(K); ax.set_xticklabels(K)
ax.set_yscale("log", base=2); ax.set_yticks([1,2,4,8,16]); ax.set_yticklabels([1,2,4,8,16])
ax.set_xlabel("K  (candidate tokens sharing one context)")
ax.set_ylabel("A(K) = expert reads amortized")
ax.set_xlim(1.85, 30)
ax.set_title("Sibling candidates DO share experts (A grows ≈ K$^{0.5}$) —\nbut 2–3.6× reuse is too weak to pay for speculative branching.",
             loc="left", fontsize=10.5, color=INK2)
style(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_reuse.png"); plt.close(fig)

# ---- Fig 3: reactive LRU beats prompt-predictive pinning ---------------------------
H = ["8", "16", "32"]
data = {"reactive LRU (deployed streamer)": ([42.1, 52.6, 62.1], BLUE),
        "predictive (pin prompt experts)":  ([27.0, 32.7, 43.8], AQUA),
        "oracle (true gen-frequency top-H)":([41.8, 56.0, 73.0], YELLOW),
        "random pin":                       ([ 2.9,  5.6, 12.3], GREEN)}
x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for i, (name, (vals, c)) in enumerate(data.items()):
    bars = ax.bar(x + (i-1.5)*w, vals, width=w-0.02, color=c, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1.2, f"{v:g}", ha="center",
                color=INK2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"H = {h} experts/layer pinned" for h in H])
ax.set_ylabel("% of generation expert-touches covered")
ax.set_ylim(0, 82)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
ax.set_title("Thesis B: the streamer's reactive LRU already matches the oracle at feasible\nbudgets — predicting the working set from the prompt adds nothing. (GLM-5.2)",
             loc="left", fontsize=10.5, color=INK2)
style(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_coverage.png"); plt.close(fig)

# ---- Fig 4: low-bit experts, quality vs bytes --------------------------------------
sizes = [29.2, 16.3, 12.3]; ppl = [10.93, 11.12, 11.87]; err = [0.36, 0.37, 0.40]
names = ["Q6_K experts\n(baseline)", "Q3_K experts", "Q2_K experts"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.errorbar(sizes, ppl, yerr=err, fmt="o", color=BLUE, markersize=9,
            capsize=4, linewidth=1.6, ecolor=BLUE_350)
for s, p, n, dy in zip(sizes, ppl, names, [-0.28, -0.3, 0.24]):
    ax.annotate(n, (s, p), textcoords="offset points",
                xytext=(0, dy*72), ha="center", fontsize=9.5, color=INK2)
ax.axhspan(10.93-0.36, 10.93+0.36, color=BLUE_250, alpha=0.22, linewidth=0)
ax.text(12.4, 10.68, "baseline PPL ± error", color=MUTED, fontsize=8.5)
ax.set_xlabel("model size, GB  (Qwopus-35B, experts-only requant)")
ax.set_ylabel("perplexity (PTB, lower is better)")
ax.set_xlim(10.5, 31.5); ax.set_ylim(10.3, 12.5)
ax.set_title("Thesis C: Q3_K experts are near-lossless at ~2× fewer bytes.\nReal — but the 754B target already ships at 3.88 bpw, so the headroom is spent.",
             loc="left", fontsize=10.5, color=INK2)
style(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_quant.png"); plt.close(fig)

# ---- Fig 5: the two walls ----------------------------------------------------------
labels = ["754B measured\n(streamed, real)",
          "754B ceiling if\ncompute were free",
          "754B ceiling if the\nmodel fit in RAM",
          "GLM-4.5-Air measured\n(resident, real)"]
vals = [0.93, 1.41, 2.78, 19.5]
cols = [BLUE, BLUE_350, BLUE_250, AQUA]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
bars = ax.barh(labels[::-1], vals[::-1], height=0.58, color=cols[::-1])
for b, v in zip(bars, vals[::-1]):
    ax.text(v+0.25, b.get_y()+b.get_height()/2, f"{v:g}", va="center",
            color=INK, fontsize=11, fontweight="bold")
ax.axvline(10, linestyle="--", color=MUTED, linewidth=1.4)
ax.text(10.15, 3.28, "10 tok/s target", color=MUTED, fontsize=9.5)
ax.set_xlabel("decode tokens / s")
ax.set_xlim(0, 21.5)
ax.set_title("Why no engine gets the 754B to 10 tok/s: even the hypothetical ceilings\n(free compute; zero disk) sit far below the target. Residency is the lever.",
             loc="left", fontsize=10.5, color=INK2)
ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_walls.png"); plt.close(fig)

print("wrote", sorted(os.listdir(OUT)))
