"""Publication figures v3 — ACL/NeurIPS-style: English labels, sans-serif, despined axes, Okabe-Ito
colour-blind palette, panel letters, non-overlapping point labels, 3-seed error bars.

fig1.pdf : (a) fertility vs separation strength, (b) stem consistency M2a vs strength with top-k coverage
           and gold ceiling, (c) fertility-M2a scatter over 24 conditions with Pareto front.
fig3_pred_vs_obs.pdf : (a) fertility predictor (c) vs observed (+ resampled seeds as hollow markers),
           (b) corrected M2a predictor vs observed; identity line and +-5% band.
"""
import sys, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS, FIGURES, PARTICLE_FREQ, GOLD_CEILING_M2A

FONT = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
             if any(ft.name == f for ft in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3,
})
S2 = RESULTS / "stage2"
# Okabe-Ito
BLUE, VERM, GREEN, ORANGE, SKY, PURPLE, YEL, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#7F7F7F"
GCOL, RCOL = BLUE, VERM
VCOL = {32000: SKY, 64000: ORANGE, 128000: "#222222"}
XLABELS = ["P0", "P2-10", "P2-30", "P2-100", "P1"]


def rows(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def cname(r):
    return r["cond"] if r["cond"] in ("P0", "P1") else f"P2{r['variant']}-{r['k']}"


def load_all():
    met = {(cname(r), int(r["vocab"])): {k: float(r[k]) for k in ("fertility", "m2a", "m2b")} for r in rows(S2 / "metrics_all_vocab.csv")}
    seed = {(r["cond"], int(r["vocab"])): r for r in rows(S2 / "s9_seed_summary.csv")}
    cov = {int(r["rank"]): float(r["cum_share"]) for r in rows(PARTICLE_FREQ)}
    return met, seed, cov


def rng(seed, c, v, m):
    r = seed.get((c, v))
    if not r or not r.get(f"{m}_min"):
        return 0.0, 0.0
    return float(r[f"{m}_mean"]) - float(r[f"{m}_min"]), float(r[f"{m}_max"]) - float(r[f"{m}_mean"])


def mean_of(seed, met, c, v, m):
    """3-seed mean for (a)(b) so the point sits at the centre of its min-max error bar."""
    r = seed.get((c, v))
    return float(r[f"{m}_mean"]) if r and r.get(f"{m}_mean") else met[(c, v)][m]


def panel_label(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left")


def label_points(ax, items):
    """items: list of (x, y, text, dx, dy) in data coords + point offsets."""
    for x, y, t, dx, dy in items:
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6.8, color="#222222",
                    ha="left" if dx >= 0 else "right", va="center")


def panel_d(ax):
    """(d) relative delta-fertility vs log2(V), 16K-256K, 3-seed mean with min-max error bars."""
    import math
    data = {}
    for r in rows(S2 / "s10_vocab_ext.csv"):
        if r["cond"] == "P0" or not r.get("rel_delta"):
            continue
        data.setdefault(r["cond"], {}).setdefault(int(r["vocab"]), []).append(float(r["rel_delta"]) * 100)
    vocabs = [16000, 32000, 64000, 128000, 256000]
    x = [math.log2(v) for v in vocabs]
    LS = {"10": ":", "30": "--", "100": "-"}
    for c in ["P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100", "P1"]:
        ys = [data[c][v] for v in vocabs]
        mean = [sum(y) / len(y) for y in ys]; lo = [m - min(y) for m, y in zip(mean, ys)]; hi = [max(y) - m for m, y in zip(mean, ys)]
        if c == "P1":
            color, marker, ls, lab = "#222222", "D", "-", "P1"
        else:
            fam, k = c[2], c.split("-")[1]
            color = GCOL if fam == "G" else RCOL; marker = "o" if fam == "G" else "s"; ls = LS[k]; lab = f"{c}"
        ax.errorbar(x, mean, yerr=[lo, hi], color=color, marker=marker, ls=ls, ms=3.6, lw=1.1, capsize=1.5, elinewidth=0.7, label=lab, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{v//1000}K" for v in vocabs], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Vocabulary size (log scale)"); ax.set_ylabel("$\\Delta$fertility vs P0 (%)")
    ax.set_ylim(0, 66)
    ax.legend(loc="upper left", fontsize=5.8, ncol=2, columnspacing=0.5, handletextpad=0.3, handlelength=1.8, labelspacing=0.2, borderaxespad=0.2)


def fig1(vocab=128000):
    met, seed, cov = load_all()
    xs = [0, 1, 2, 3, 4]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    for var, color, marker, lab in (("G", GCOL, "o", "P2G (analyzer-guided)"), ("R", RCOL, "s", "P2R (rule-based)")):
        conds = ["P0"] + [f"P2{var}-{k}" for k in (10, 30, 100)] + ["P1"]
        for ax, m, scale in ((axes[0], "fertility", 1), (axes[1], "m2a", 100)):
            y = [mean_of(seed, met, c, vocab, m) * scale for c in conds]
            lo, hi = zip(*[rng(seed, c, vocab, m) for c in conds])
            ax.errorbar(xs, y, yerr=[[e * scale for e in lo], [e * scale for e in hi]], color=color, marker=marker,
                        ms=4.2, lw=1.4, capsize=1.8, elinewidth=0.8, label=lab, zorder=3)
    axes[1].plot([1, 2, 3], [cov[10] * 100, cov[30] * 100, cov[100] * 100], color=GREY, ls=":", marker="^", ms=4.2, lw=1.2,
                 label="Top-$k$ coverage", zorder=2)
    axes[1].axhline(GOLD_CEILING_M2A * 100, color="#222222", ls=(0, (2, 3)), lw=0.8, label="Gold ceiling (99.65%)", zorder=1)
    for ax, yl in ((axes[0], "Fertility (tokens / word)"), (axes[1], "Stem consistency M2a (%)")):
        ax.set_xticks(xs); ax.set_xticklabels(XLABELS, rotation=25, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Separation strength")
        ax.set_ylabel(yl)
        ax.set_xlim(-0.3, 4.3)
    axes[0].set_ylim(1.5, 2.2); axes[1].set_ylim(0, 105)
    axes[0].legend(loc="upper left", fontsize=6.3, handlelength=1.4, labelspacing=0.3)
    hb, lb = axes[1].get_legend_handles_labels()
    lb = [l.replace(" (analyzer-guided)", "").replace(" (rule-based)", "").replace("Gold ceiling (99.65%)", "Gold ceiling") for l in lb]
    axes[1].legend(hb, lb, loc="lower right", fontsize=6.3, handlelength=1.4, labelspacing=0.3)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)"); panel_label(axes[2], "(c)"); panel_label(axes[3], "(d)")
    # (c) 24-condition scatter
    ax = axes[2]
    pts = [(mean_of(seed, met, c, v, "fertility"), mean_of(seed, met, c, v, "m2a") * 100, c, v) for (c, v) in met]
    for x, y, c, v in pts:
        mk = "o" if "G" in c else ("s" if "R" in c else "D")
        ax.scatter(x, y, color=VCOL[v], marker=mk, s=22, alpha=0.95, edgecolor="white", linewidth=0.4, zorder=3)
    front = []
    for p in sorted(pts):
        if not front or p[1] > front[-1][1]:
            front.append(p)
    ax.plot([p[0] for p in front], [p[1] for p in front], color=GREY, ls="--", lw=0.9, zorder=2)
    ax.axhline(GOLD_CEILING_M2A * 100, color="#222222", ls=(0, (2, 3)), lw=0.8, zorder=1)
    P = {c: (x, y) for x, y, c, v in pts if v == 128000}
    # labels for the 128K points placed in empty regions with thin leader lines
    def lead(c, txt, tx, ty):
        ax.annotate(txt, P[c], xytext=(tx, ty), textcoords="data", fontsize=6.8, color="#222222", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=0, shrinkB=2), zorder=4)
    ax.annotate("P0", P["P0"], textcoords="offset points", xytext=(5, 0), fontsize=6.8, va="center")
    lead("P2G-10", "G10", 1.66, 78); lead("P2R-10", "R10", 1.66, 60)
    lead("P2G-30", "G30", 1.66, 96); lead("P2R-30", "R30", 1.80, 84)
    lead("P2G-100", "G100", 1.90, 106); lead("P2R-100", "R100", 2.05, 86)
    lead("P1", "P1", 2.21, 90)
    ax.set_xlabel("Fertility (tokens / word)"); ax.set_ylabel("Stem consistency M2a (%)")
    ax.set_xlim(1.5, 2.3); ax.set_ylim(0, 110)
    h = [Line2D([], [], color=VCOL[v], marker="o", ls="", ms=4.2, label=f"{v//1000}K") for v in (32000, 64000, 128000)]
    h += [Line2D([], [], color=GREY, marker=m, ls="", ms=4.2, label=l) for m, l in (("o", "P2G"), ("s", "P2R"), ("D", "P0/P1"))]
    ax.legend(handles=h, loc="lower right", ncol=1, fontsize=6.0, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.2)
    panel_d(axes[3])
    fig.tight_layout(w_pad=1.2)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig1.pdf"); fig.savefig(FIGURES / "fig1.png", dpi=220); plt.close(fig)
    print("fig1 saved (font:", FONT + ")")


def fig3():
    base = rows(S2 / "s3_prediction.csv")
    seeds = []
    for sd in (20260908, 20260909):
        p = S2 / "seeds" / str(sd) / "s3_prediction_seed.csv"
        if p.exists():
            seeds += rows(p)
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 5.6))
    short = lambda c: c.replace("P2G-", "G").replace("P2R-", "R")
    specs = ((axes[0], "fert_pred_c", "fert_obs", "Predicted fertility (estimator c)", "Observed fertility"),
             (axes[1], "m2a_pred_first_p0", "m2a_obs", "Predicted M2a (corrected)", "Observed M2a"))
    for ax, px, py, xl, yl in specs:
        X = [float(r[px]) for r in base]; Y = [float(r[py]) for r in base]
        lo = min(X + Y); hi = max(X + Y); pad = (hi - lo) * 0.08; xs = [lo - pad, hi + pad]
        ax.fill_between(xs, [x * 0.95 for x in xs], [x * 1.05 for x in xs], color="#000000", alpha=0.07, lw=0, zorder=0)
        ax.plot(xs, xs, color=GREY, ls="--", lw=0.9, zorder=1)
        if ax is axes[0]:
            for r in seeds:
                ax.scatter(float(r["fert_pred_c"]), float(r["fert_obs"]), facecolor="none", edgecolor=VCOL[int(r["vocab"])],
                           marker="o" if r["cond"].startswith("P2G") else "s", s=18, lw=0.7, alpha=0.9, zorder=2)
        for r in base:
            v = int(r["vocab"]); mk = "o" if r["variant"] == "G" else "s"
            ax.scatter(float(r[px]), float(r[py]), color=VCOL[v], marker=mk, s=26, edgecolor="white", linewidth=0.4, zorder=3)
        P = {short(r["cond"]): (float(r[px]), float(r[py])) for r in base if int(r["vocab"]) == 128000}
        def lead(nm, tx, ty):
            ax.annotate(nm, P[nm], xytext=(tx, ty), textcoords="data", fontsize=6.8, color="#222222", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=0, shrinkB=2), zorder=4)
        if ax is axes[0]:
            # labels in the empty upper-left region (above y = x)
            lead("R10", 1.80, 1.86); lead("G10", 1.80, 1.925)
            lead("R30", 1.86, 2.02); lead("G30", 1.90, 1.965)
            lead("R100", 1.95, 2.13); lead("G100", 1.93, 2.075)
        else:
            lead("R10", 0.735, 0.665); lead("G10", 0.76, 0.705)
            lead("G30", 0.865, 0.845); lead("R30", 0.935, 0.80)
            lead("G100", 0.88, 0.99); lead("R100", 0.985, 0.93)
        ax.set_xlim(xs); ax.set_ylim(xs); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    h = [Line2D([], [], color=VCOL[v], marker="o", ls="", ms=4.2, label=f"{v//1000}K") for v in (32000, 64000, 128000)]
    h += [Line2D([], [], color=GREY, marker="o", ls="", ms=4.2, label="P2G"), Line2D([], [], color=GREY, marker="s", ls="", ms=4.2, label="P2R"),
          Line2D([], [], color=GREY, marker="o", ls="", ms=4.2, markerfacecolor="none", label="Resampled seeds"),
          Line2D([], [], color=GREY, ls="--", lw=0.9, label="y = x (±5% band)")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.8, handletextpad=0.3, columnspacing=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=1.4, rect=[0, 0.075, 1, 1])
    fig.savefig(FIGURES / "fig3_pred_vs_obs.pdf"); fig.savefig(FIGURES / "fig3_pred_vs_obs.png", dpi=220); plt.close(fig)
    print("fig3 saved")


if __name__ == "__main__":
    fig1(); fig3()
