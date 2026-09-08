"""Homepage focus-area figures for eleuther.ai, drawn with matplotlib.

Paste into a notebook or run as a script. Each function returns (fig, ax) so
you can tweak placement, then call save(fig, "name") to write the SVG that
data/home.yaml points at. Colors are the site's CSS tokens.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

# --- site palette ----------------------------------------------------------
BG = "#10151d"        # --panel
TEXT = "#ffffff"      # maximum contrast on the panel
MUTED = "#c4c9d1"     # --body, for secondary labels
LINE = "#3a4656"      # axis lines, a step lighter than --line for legibility
GREEN = "#62d77a"
BLUE = "#78b8ee"
GOLD = "#d8b44a"
VIOLET = "#ad8df2"
CORAL = "#ee7b79"

FIGSIZE = (10, 7)     # 10:7 keeps the four homepage panels the same shape
FONT = {"family": ["Inter", "DejaVu Sans", "sans-serif"]}
plt.rcParams.update({
    "font.family": FONT["family"],
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": LINE,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.size": 18,
    "svg.fonttype": "none",   # keep text as text in the SVG so it stays crisp and editable
})


def blank_axes(ax, spines=("left", "bottom")):
    """No grid, no ticks; keep only the named spines."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in spines)
        ax.spines[side].set_linewidth(2)


def save(fig, name, outdir="static/images/research/homepage"):
    path = f"{outdir}/{name}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    # matplotlib leaves trailing spaces in path data, which fails `git diff --check`
    with open(path) as handle:
        lines = [line.rstrip() + "\n" for line in handle]
    with open(path, "w") as handle:
        handle.writelines(lines)


def neural_network(ax, x0, y0, width, height, layers=(4, 6, 6, 3), color=VIOLET,
                   node_ms=13, edge_alpha=0.35, edge_lw=1.2, zorder=6):
    """Draw a fully connected feed-forward network inside the box (x0, y0, width, height).

    Nodes are plot markers (sized in points), so the drawing looks the same
    whatever the axes' aspect ratio."""
    xs = np.linspace(x0, x0 + width, len(layers))
    coords = []
    for x, n in zip(xs, layers):
        ys = np.linspace(y0 + height, y0, n + 2)[1:-1] if n > 1 else [y0 + height / 2]
        coords.append([(x, y) for y in ys])
    for a, b in zip(coords[:-1], coords[1:]):
        for (xa, ya) in a:
            for (xb, yb) in b:
                ax.plot([xa, xb], [ya, yb], color=color, alpha=edge_alpha, lw=edge_lw, zorder=zorder)
    for layer in coords:
        xs_, ys_ = zip(*layer)
        ax.plot(xs_, ys_, "o", ms=node_ms, mfc=color, mec=BG, mew=1.5, zorder=zorder + 1)
    return coords


def document(ax, x, y, w=0.9, h=1.1, color=VIOLET, lines=4):
    """A training document: rounded page with text lines inside."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=BG, edgecolor=color, lw=2, zorder=2))
    for i in range(lines):
        ly = y + h - 0.22 - i * (h - 0.35) / (lines - 1)
        lw_frac = 0.7 if i < lines - 1 else 0.45
        ax.plot([x + 0.15, x + 0.15 + (w - 0.3) * lw_frac], [ly, ly], color=color, lw=2.5,
                solid_capstyle="round", zorder=3)


# ---------------------------------------------------------------------------
# 1. Evaluation: benchmark saturation
# ---------------------------------------------------------------------------
def evaluation_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blank_axes(ax)
    x = np.linspace(0, 1, 400)
    logistic = lambda center, k, top=0.97: top / (1 + np.exp(-k * (x - center)))

    # Older benchmarks: saturate early, pinned against the ceiling on the right.
    for i, (center, k, alpha) in enumerate([(0.14, 22, 1.0), (0.24, 18, 0.75), (0.34, 15, 0.5)]):
        ax.plot(x, logistic(center, k), color=GOLD, lw=3.5, alpha=alpha, solid_capstyle="round",
                label="Older Benchmarks" if i == 0 else None)
        ax.plot(1, logistic(center, k)[-1], "o", ms=10, mfc=BG, mec=GOLD, mew=2.5)
    # Newer benchmarks: still climbing.
    for i, (center, k, alpha) in enumerate([(0.85, 8, 1.0), (1.05, 8, 0.65)]):
        ax.plot(x, logistic(center, k), color=BLUE, lw=3.5, alpha=alpha, solid_capstyle="round",
                label="Newer Benchmarks" if i == 0 else None)
        ax.plot(1, logistic(center, k)[-1], "o", ms=10, mfc=BG, mec=BLUE, mew=2.5)

    ax.axhline(0.97, color=GOLD, lw=2, ls=(0, (6, 6)))
    ax.text(0.995, 0.995, "Ceiling", color=GOLD, fontsize=18, ha="right", va="bottom")

    # Annotations sit in the empty regions; move freely.
    ax.text(0.50, 0.74, "Saturated: No Longer\nSeparates Models", color=GOLD, fontsize=19,
            ha="left", va="top", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BG, edgecolor=GOLD, lw=1.8))
    ax.text(0.36, 0.22, "Still Informative", color=BLUE, fontsize=19, ha="left", va="bottom")

    # Legend above the axes, out of the way of every curve.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=18,
              handlelength=1.6, columnspacing=2.0, labelcolor=TEXT)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Time Since Benchmark Release  →", fontsize=19, loc="right", labelpad=12)
    ax.set_ylabel("Score", fontsize=19, loc="top", rotation=0, labelpad=16)
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Open-weight safety: filtering pretraining data
# ---------------------------------------------------------------------------
def open_weight_safety_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blank_axes(ax, spines=())
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 9.8)
    ax.set_aspect("equal")

    # Pretraining documents flowing right; coral ones are hazardous and stop at the filter.
    docs_y = np.linspace(7.4, 1.4, 6)
    hazardous = {1, 4}
    ax.text(1.2, 9.0, "Pretraining Data", color=TEXT, fontsize=19, fontweight="bold", ha="center")
    for i, y in enumerate(docs_y):
        color = CORAL if i in hazardous else VIOLET
        document(ax, 0.75, y - 0.55, color=color)
        end = 3.0 if i in hazardous else 6.1
        ax.plot([1.85, end], [y, y], color=color, lw=2.5, alpha=0.9, zorder=1)

    # Filter bar.
    ax.add_patch(FancyBboxPatch((3.0, 0.7), 0.35, 7.6, boxstyle="round,pad=0.02,rounding_size=0.17",
                                facecolor=BG, edgecolor=VIOLET, lw=2.5, zorder=4))
    for y in np.linspace(1.2, 7.8, 12):
        ax.plot([3.08, 3.27], [y, y], color=VIOLET, lw=2, zorder=5)
    ax.text(3.17, 0.15, "Filter", color=TEXT, fontsize=19, fontweight="bold", ha="center")

    # Model: a real network.
    ax.add_patch(FancyBboxPatch((6.1, 0.9), 3.9, 7.2, boxstyle="round,pad=0.02,rounding_size=0.25",
                                facecolor="#0b1018", edgecolor=VIOLET, lw=2.5, zorder=2))
    neural_network(ax, 6.6, 1.3, 2.9, 5.4, layers=(5, 7, 7, 4), color=VIOLET, node_ms=12)
    ax.text(8.05, 8.55, "Open-Weight Model", color=TEXT, fontsize=19, fontweight="bold", ha="center", zorder=8)
    return fig, ax


# Deep Ignorance, Figure 1, redrawn in the site palette.
# Colors follow the schematic above: coral = unfiltered (hazard retained), violet = filtered.
#
# Sources:
#   Bars: exact end-of-training averages (MMLU, PIQA, Lambada, HellaSwag) from the results
#   table in github.com/EleutherAI/deep-ignorance/HF_README.md, for deep-ignorance-unfiltered,
#   deep-ignorance-e2e-weak-filter and deep-ignorance-e2e-strong-filter.
#   Attack curves: digitized from Figure 1 of the paper (arXiv:2508.06601) at 25M-token
#   intervals, with the shaded band width read off the same figure. The per-checkpoint
#   values live in wandb, not the repo; replace DI_BIOTHREAT / DI_BAND with the real series
#   (and finer spacing) when they are exported.
DI_GENERAL = {"Baseline": 0.5605, "Weak Filter": 0.5737, "Strong Filter": 0.5553}
DI_TOKENS = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300])     # millions
DI_BIOTHREAT = {
    "Baseline":      np.array([0.365, 0.376, 0.390, 0.399, 0.409, 0.415, 0.425, 0.440, 0.441, 0.440, 0.439, 0.441, 0.445]),
    "Weak Filter":   np.array([0.278, 0.330, 0.340, 0.356, 0.365, 0.369, 0.378, 0.394, 0.392, 0.393, 0.395, 0.398, 0.402]),
    "Strong Filter": np.array([0.250, 0.310, 0.325, 0.340, 0.350, 0.355, 0.364, 0.385, 0.380, 0.375, 0.378, 0.380, 0.382]),
}
DI_BAND = 0.008          # half-width of the shaded uncertainty band, as drawn in the paper
DI_RANDOM = 0.25
DI_COLORS = {"Baseline": CORAL, "Weak Filter": GOLD, "Strong Filter": VIOLET}


def deep_ignorance_figure(annotate_recovery=True):
    fig, (left, right) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.32})
    for ax in (left, right):
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(2)
        ax.tick_params(labelsize=15, length=5, width=1.5, colors=TEXT)

    names = list(DI_GENERAL)
    left.bar(range(3), [DI_GENERAL[n] for n in names], color=[DI_COLORS[n] for n in names], width=0.7)
    left.set_xticks([])
    left.set_ylim(0, 0.65)
    left.set_yticks([0, 0.2, 0.4, 0.6])
    left.set_yticklabels(["0%", "20%", "40%", "60%"])
    left.set_title("General Capability  ↑\n(Avg. on 4 Benchmarks)", fontsize=16, color=TEXT, pad=14)

    for n in names:
        y = DI_BIOTHREAT[n]
        right.fill_between(DI_TOKENS, y - DI_BAND, y + DI_BAND, color=DI_COLORS[n], alpha=0.22, lw=0)
        right.plot(DI_TOKENS, y, color=DI_COLORS[n], lw=2.8, label=n, solid_capstyle="round", solid_joinstyle="round")
    right.axhline(DI_RANDOM, color=MUTED, lw=1.8, ls=(0, (6, 6)))
    right.text(DI_TOKENS[-1], DI_RANDOM - 0.006, "Random", color=MUTED, fontsize=14, ha="right", va="top")

    if annotate_recovery:
        # Stella's annotation: how much fine-tuning it takes for a filtered model to reach the
        # unfiltered model's starting point.
        baseline_start = DI_BIOTHREAT["Baseline"][0]
        strong = DI_BIOTHREAT["Strong Filter"]
        cross = np.interp(baseline_start, strong, DI_TOKENS)          # first token count where Strong >= baseline start
        right.axhline(baseline_start, color=TEXT, lw=1.6, ls=(0, (1.5, 3)), alpha=0.9)
        right.plot([cross, cross], [DI_RANDOM, baseline_start], color=TEXT, lw=1.6, ls=(0, (1.5, 3)), alpha=0.9)
        right.text(cross + 8, 0.335, "> 150M tokens of fine-tuning\nrestore capability to match\nthe unfiltered model",
                   color=TEXT, fontsize=13.5, ha="left", va="top", linespacing=1.3)

    right.set_xlim(0, 305)
    right.set_ylim(0.22, 0.47)
    right.set_xticks([0, 50, 100, 150, 200, 250, 300])
    right.set_xticklabels(["0", "50M", "100M", "150M", "200M", "250M", "300M"])
    right.set_yticks([0.25, 0.3, 0.35, 0.4, 0.45])
    right.set_yticklabels(["25%", "30%", "35%", "40%", "45%"])
    right.set_xlabel("Adversarial Fine-Tuning Tokens", fontsize=16, labelpad=10)
    right.set_title("Biothreat Proxy Capability  ↓\n(WMDP-Bio, Cloze Eval)", fontsize=16, color=TEXT, pad=14)

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=17,
               handlelength=1.8, columnspacing=2.2, labelcolor=TEXT)
    return fig, (left, right)


# ---------------------------------------------------------------------------
# 3. Interpretability over time: checkpoints along the loss curve
# ---------------------------------------------------------------------------
def interpretability_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blank_axes(ax)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.14, 1.12)
    ax.spines["bottom"].set_position(("data", 0))

    x = np.linspace(0, 1, 400)
    loss = 0.12 + 0.88 * np.exp(-4.2 * x)
    ax.plot(x, loss, color=GREEN, lw=3.5, solid_capstyle="round", zorder=2)

    # Saved checkpoints, each with a vertical line to the axis.
    ckpt_x = np.array([0.0, 0.08, 0.17, 0.28, 0.42, 0.56, 0.7, 0.85, 1.0])
    ckpt_y = 0.12 + 0.88 * np.exp(-4.2 * ckpt_x)
    for cx, cy in zip(ckpt_x, ckpt_y):
        ax.plot([cx, cx], [0, cy], color=GREEN, lw=1.5, ls=(0, (3, 4)), alpha=0.8, zorder=1)
    ax.plot(ckpt_x, ckpt_y, "o", ms=11, mfc=BG, mec=GREEN, mew=2.5, zorder=3)

    ax.set_xlabel("Training Steps  →", fontsize=19, loc="right", labelpad=12)
    ax.set_ylabel("Loss", fontsize=19, loc="top", rotation=0, labelpad=16)
    # "Saved Checkpoints" label with a leader line to the foot of one checkpoint's dashed line.
    ax.text(0.15, -0.10, "Saved Checkpoints", color=TEXT, fontsize=18, ha="left", va="top")
    ax.annotate("", xy=(ckpt_x[3], -0.005), xytext=(0.22, -0.09),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.8, shrinkA=0, shrinkB=0))

    # Lens over one checkpoint: the network inside is what we study.
    # The axes are not square, so an Ellipse in data units is what renders as a circle.
    aspect = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) * FIGSIZE[0] / FIGSIZE[1]
    lens_ckpt = 5
    lx, ly, r = 0.76, 0.52, 0.16
    ax.plot([ckpt_x[lens_ckpt], lx], [ckpt_y[lens_ckpt] + 0.02, ly - r * aspect], color=GREEN, lw=1.8,
            ls=(0, (3, 4)), zorder=2)
    ax.add_patch(Ellipse((lx, ly), 2 * r, 2 * r * aspect, facecolor="#0b1018", edgecolor=GREEN, lw=2.5, zorder=4))
    neural_network(ax, lx - 0.12, ly - 0.10 * aspect, 0.24, 0.20 * aspect, layers=(3, 5, 5, 2), color=GREEN,
                   node_ms=9, edge_lw=1.0)

    # What we do at every checkpoint.
    todo = ["Probe Representations", "Trace Behavior to Training Data", "Compare Across Checkpoints"]
    for i, label in enumerate(todo):
        ax.text(0.30, 1.06 - i * 0.075, "•  " + label, color=TEXT, fontsize=18, ha="left", va="center")
    return fig, ax


if __name__ == "__main__":
    fig, _ = evaluation_figure(); save(fig, "evaluation")
    fig, _ = open_weight_safety_figure(); save(fig, "open-weight-safety")
    fig, _ = deep_ignorance_figure(); save(fig, "deep-ignorance-figure-1")
    fig, _ = interpretability_figure(); save(fig, "interpretability-over-time")
    print("wrote 4 SVGs")
