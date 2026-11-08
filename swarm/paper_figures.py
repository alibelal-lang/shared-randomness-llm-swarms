"""Figures for the sprint write-up (report/figures/). Run: .venv/bin/python -m swarm.paper_figures"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .explore_report import load as load_grid  # noqa: E402

OUT = "report/figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
INDEP, COUPLED = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 (validated palette)
STAG, COORD = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)


def fig1():
    rows = json.load(open("report/data/ext_ncurve.json"))
    panels = [("glm-5.2_stag_f4", "GLM 5.2", "+22 points\n(95% CI +12 to +30)"),
              ("qwen3.8-27b_stag_f4", "Qwen 3.8 27B", "+9.5 points\n(95% CI +5.5 to +13.5)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    for ax, (key, title, note) in zip(axes, panels):
        rs = sorted([r for r in rows if r["set"] == key], key=lambda r: r["N"])
        N = np.array([r["N"] for r in rs])
        ind = 100 * np.array([r["x_independent"] for r in rs])
        cou = 100 * np.array([r["x_coupled"] for r in rs])
        ax.plot(N, cou, color=COUPLED, lw=2, marker="o", ms=4.5, mec="white", mew=1, zorder=3)
        ax.plot(N, ind, color=INDEP, lw=2, marker="o", ms=4.5, mec="white", mew=1, zorder=3)
        ax.text(N[-1] * 1.12, cou[-1], "Coupled", color=INK, va="center", fontsize=8.5)
        ax.text(N[-1] * 1.12, ind[-1], "Independent", color=INK, va="center", fontsize=8.5)
        i32 = list(N).index(32)
        ax.annotate("", xy=(32, cou[i32] - 0.8), xytext=(32, ind[i32] + 0.8),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        if cou[i32] - ind[i32] > 15:  # room between the lines: label right of the arrow
            ax.text(35, (cou[i32] + ind[i32]) / 2, note, ha="left", va="center", fontsize=7.5, color=INK2)
        else:  # narrow gap: label above the coupled line
            ax.text(32, cou[i32] + 4, note, ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
        ax.set_xlim(3.4, 64 * 2.6)
        ax.set_ylim(55, 100)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlabel("Population size (agents)")
        ax.spines["bottom"].set_bounds(4, 64)
    axes[0].set_ylabel("Agents on better convention (%)")
    _save(fig, "fig1_size_and_coupling")


def fig2(split=True):
    if split:  # size from seeds 0-199 (original grid), coupling from the independent batch 200000-200199
        from .checks import seedsplit_table
        t = seedsplit_table()
        s = np.array([r["size_A"] for r in t])
        c = np.array([r["coup_B"] for r in t])
        stag = np.array([r["game"] == "stag" for r in t])
    else:
        b = [r for r in load_grid() if r["H"] == 2 and r["temperature"] == 1.0]
        s = np.array([r["size32_pp"] for r in b])
        c = np.array([r["coupling32_pp"] for r in b])
        stag = np.array([r["game"] == "stag" for r in b])
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    lim = (-72, 58)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax.plot(lim, [-lim[0], -lim[1]], ls=(0, (4, 3)), color=INK2, lw=0.9, zorder=1)
    ax.text(-54, 61, "full recovery (y = −x)", fontsize=7.5, color=INK2, va="center")
    for mask, col, lab in ((~stag, COORD, "Coordination game"), (stag, STAG, "Stag hunt")):
        ax.scatter(s[mask], c[mask], s=14, color=col, alpha=0.65, lw=0.4, edgecolors="white", label=lab, zorder=2)
    stuck = s <= -2
    up = s >= 2
    ax.text(-70, -28, f"Large groups stuck: {stuck.sum()}\ncoupling helps in {int((c[stuck] > 0).sum())}",
            fontsize=7.5, color=INK, va="bottom")
    ax.text(56, 34, f"Large groups better: {up.sum()}\ncoupling hurts in {int((c[up] < 0).sum())}",
            fontsize=7.5, color=INK, ha="right", va="top")
    ax.set_xlim(*lim)
    ax.set_ylim(-36, 70)
    ax.set_xlabel("Size effect: 32 minus 4 agents, independent (points)")
    ax.set_ylabel("Coupling effect at 32 agents (points)" + ("\n(independent seed batch)" if split else ""))
    ax.legend(frameon=False, loc="upper right", fontsize=8, handletextpad=0.2, markerscale=1.4)
    r = np.corrcoef(s, c)[0, 1]
    ax.text(0.985, 0.80, f"r = \u2212{abs(r):.2f}, n = {len(s)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK2)
    _save(fig, "fig2_mechanism" if split else "fig2_mechanism_sameseeds")


if __name__ == "__main__":
    fig1()
    fig2(split=False)
    fig2(split=True)
    print("wrote", sorted(os.listdir(OUT)))
