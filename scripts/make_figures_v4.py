# -*- coding: utf-8 -*-
"""The paper's figures, with two LLM coders (figures show both coders and the pooled reading).

    python make_figures_v4.py            -> figs-v4/fig_contrasts.pdf, fig_goals_main.pdf, fig_goals.pdf,
                                            fig_itemtypes.pdf, fig_valence.pdf, fig_goals_valence.pdf, fig_models.pdf

Contrast figure: other-benefit contrasts (both generating models) per arm, with each coder's 95% interval and the
pooled interval whose bootstrap also resamples the coder (analyze_coders.two_coder, same seed). Every descriptive share
is the mean of the two coders' shares; each coder's own numbers are in results-v4-<coder>.md.
make_figures.py keeps drawing the submitted (v3, one coder) figures.
"""
import os, sys, pathlib, collections, statistics
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as A
import analyze_coders as AC
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType, not Type 3, for the ACL format check
CODERS = {"sol61": ("GPT-6.1 Sol", HERE.parent / "llm/sol61-all/codes_sol61_all.csv"),
          "opus55": ("Claude Opus 5.5", HERE.parent / "llm/opus55-all/codes_opus55_all.csv")}
OUT = pathlib.Path(os.environ.get("FIGOUT", str(HERE / "figs-v4")))
CELL_LABEL = {"us_en": "US setting\nEnglish", "us_th": "US setting\nThai", "th_th": "Thai setting\nThai", "th_en": "Thai setting\nEnglish"}
MODEL_LABEL = {"gpt": "GPT-5.6 Luna", "ds": "DeepSeek V4.1 Flash"}
KIND_LABEL = {"language": "Language", "setting": "Setting", "interaction": "Interaction"}
ARM_LABEL = {"imp": "Country not named", "exp": "Country named"}
CELL_COL = {"us_en": "#0072B2", "us_th": "#56B4E9", "th_th": "#E69F00", "th_en": "#D55E00"}
MAIN_GOALS = ("growth", "relational", "wellbeing", "resources", "achievement", "community")
GLOSS = {"growth": "got better,\ndid something new", "relational": "connection,\nbelonging", "wellbeing": "fun, relief,\nrest",
         "resources": "time, money,\norder", "achievement": "a result,\nrecognition", "community": "the school,\nothers beyond",
         "transcendent": "karma, fate,\nmeaning", "unspecified": "no goal\nnamed"}
VAL = ("pos", "neu", "neg")
VLAB = {"pos": "positive events", "neu": "neutral events", "neg": "negative events"}


def share(sel, field, denom, keys):
    tot = sum(m[denom] for m in sel); cnt = collections.Counter()
    for m in sel: cnt.update(m[field])
    return {k: (100 * cnt[k] / tot if tot else 0) for k in keys}


def mean_over(Ms, fn):
    """fn(M) -> dict or number; the mean over coders."""
    vals = [fn(M) for M in Ms]
    if isinstance(vals[0], dict):
        return {k: statistics.mean(v[k] for v in vals) for k in vals[0]}
    return statistics.mean(vals)


def other_benefit(sel):
    return 100 * statistics.mean([m["owner_share"] for m in sel if m["owner_share"] is not None])


def style(ax):
    ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)


def main():
    Mc = {n: A.load(str(p)) for n, (_, p) in CODERS.items()}
    G = tuple(g for g in A.GOALS if g != "unspecified")  # no coder used it
    names, common, est, pooled, own, mix, keys = AC.two_coder(Mc)
    OUT.mkdir(parents=True, exist_ok=True)
    print("check (imp, both, pooled):", {k: (round(pooled[("imp", "both", k)], 1), tuple(round(x, 1) for x in AC.ci(mix[("imp", "both", k)]))) for k in A.KINDS})

    # ---- contrasts: each coder and pooled, both generating models
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    y, ticks, labels = 0, [], []
    styles = [("sol61", "o", "#0072B2", -0.22), ("opus55", "s", "#E69F00", 0.0), ("pooled", "D", "black", 0.22)]
    for arm in ("imp", "exp"):
        for kind in A.KINDS:
            k = (arm, "both", kind)
            for n, mk, col, off in styles:
                e, (lo, hi) = (pooled[k], AC.ci(mix[k])) if n == "pooled" else (est[n][k], AC.ci(own[n][k]))
                ax.errorbar(e, y + off, xerr=[[e - lo], [hi - e]], fmt=mk, color=col, ms=3.2, lw=0.9, capsize=1.5,
                            label=(CODERS[n][0] if n in CODERS else "Pooled") if (arm, kind) == ("imp", "language") else None)
            ticks.append(y); labels.append(KIND_LABEL[kind]); y += 1
        y += 0.6
    ax.axvspan(-10, 10, color="0.92", zorder=0)
    ax.axvline(0, color="0.4", lw=0.6, zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(-14, 14); ax.set_xlabel("Contrast in other-benefit share (points)")
    for arm, ylab in (("imp", -0.75), ("exp", 2.85)):
        ax.text(-13.6, ylab, ARM_LABEL[arm], ha="left", va="center", fontsize=8.5, style="italic")
    ax.set_ylim(5.95, -1.2)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(); fig.savefig(OUT / "fig_contrasts.pdf"); plt.close(fig)

    Ms = list(Mc.values())
    blocks = sorted(common)
    imp = [b for b in blocks if Ms[0][b]["arm"] == "imp"]

    # ---- goal categories per cell (main text), mean of the coders
    sh = {c: mean_over(Ms, lambda M: share([M[b] for b in imp if M[b]["cell"] == c], "goal", "n_lin", G)) for c in A.CELLS}
    order = sorted(G, key=lambda k: -sum(sh[c][k] for c in A.CELLS))
    print("check (goal shares, coder mean):", {k: round(sum(sh[c][k] for c in A.CELLS) / 4, 1) for k in order})
    top = max(sh[c][k] for c in A.CELLS for k in order)
    fig, ax = plt.subplots(figsize=(6.3, 1.6)); W = 0.2
    for j, c in enumerate(A.CELLS):
        ax.bar([i + (j - 1.5) * W for i in range(len(order))], [sh[c][k] for k in order], width=W,
               color=CELL_COL[c], label=CELL_LABEL[c].replace("\n", ", "))
    ax.set_xticks(range(len(order))); ax.set_xticklabels([k + "\n" + GLOSS[k] for k in order], fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("Percent of linings"); ax.set_ylim(0, 5 * round((top + 7) / 5)); ax.set_xlim(-0.55, len(order) - 0.45)
    style(ax)
    ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper right", handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig_goals_main.pdf"); plt.close(fig)

    # ---- appendix: goal categories and item types per cell and model, mean of the coders
    for name, keys_, field, denom, ylabel in (("fig_goals", G, "goal", "n_lin", "Percent of linings"),
                                             ("fig_itemtypes", A.TYPES, "type", "n", "Percent of items")):
        cols, data = [], {k: [] for k in keys_}
        for c in A.CELLS:
            for mdl in A.MODELS:
                cols.append(f"{CELL_LABEL[c]}\n{'GPT' if mdl == 'gpt' else 'DeepSeek'}")
                s = mean_over(Ms, lambda M: share([M[b] for b in imp if M[b]["cell"] == c and M[b]["model"] == mdl], field, denom, keys_))
                for k in keys_: data[k].append(s[k])
        fig, ax = plt.subplots(figsize=(6.3, 2.6)); bottom = [0] * len(cols)
        palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000", "#999999"]
        for i, k in enumerate(keys_):
            ax.bar(range(len(cols)), data[k], bottom=bottom, label={"none": "not a lining"}.get(k, k), width=0.7, color=palette[i % len(palette)])
            bottom = [b + d for b, d in zip(bottom, data[k])]
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_ylim(0, 100)
        ax.legend(fontsize=8, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)

    # ---- valence (main text): other-benefit share by valence and cell; goals by valence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 1.7), gridspec_kw={"width_ratios": [1, 1.35]}); W = 0.2
    for j, c in enumerate(A.CELLS):
        vals = [mean_over(Ms, lambda M: other_benefit([M[b] for b in imp if M[b]["valence"] == v and M[b]["cell"] == c])) for v in VAL]
        ax1.bar([i + (j - 1.5) * W for i in range(3)], vals, width=W, color=CELL_COL[c], label=CELL_LABEL[c].replace("\n", ", "))
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["positive", "neutral", "negative"], fontsize=8)
    ax1.set_ylabel("Other-benefit share (%)", fontsize=8); ax1.set_ylim(0, 36); ax1.tick_params(labelsize=8)
    VCOL = {"pos": "#009E73", "neu": "#999999", "neg": "#CC79A7"}; W2 = 0.26
    for j, v in enumerate(VAL):
        s = mean_over(Ms, lambda M: share([M[b] for b in imp if M[b]["valence"] == v], "goal", "n_lin", MAIN_GOALS))
        ax2.bar([i + (j - 1) * W2 for i in range(len(MAIN_GOALS))], [s[g] for g in MAIN_GOALS], width=W2, color=VCOL[v], label=VLAB[v])
        print(f"check ({v}): growth {s['growth']:.1f}")
    ax2.set_xticks(range(len(MAIN_GOALS))); ax2.set_xticklabels(MAIN_GOALS, fontsize=6.5, rotation=15, ha="right")
    ax2.set_ylabel("Percent of linings", fontsize=8); ax2.set_ylim(0, 78); ax2.tick_params(labelsize=8)
    for ax in (ax1, ax2): style(ax)
    ax1.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", handlelength=0.9, handletextpad=0.3, columnspacing=0.6)
    ax2.legend(fontsize=7, frameon=False, ncol=3, loc="upper right", handlelength=1.0, handletextpad=0.3, columnspacing=0.7)
    fig.tight_layout(); fig.savefig(OUT / "fig_valence.pdf"); plt.close(fig)

    # ---- appendix: goals by valence and cell
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
    for ax, v in zip(axes, VAL):
        for j, c in enumerate(A.CELLS):
            s = mean_over(Ms, lambda M: share([M[b] for b in imp if M[b]["valence"] == v and M[b]["cell"] == c], "goal", "n_lin", MAIN_GOALS))
            ax.bar([i + (j - 1.5) * 0.2 for i in range(len(MAIN_GOALS))], [s[g] for g in MAIN_GOALS], width=0.2,
                   color=CELL_COL[c], label=CELL_LABEL[c].replace("\n", ", ") if v == "neg" else None)
        ax.set_title(VLAB[v], fontsize=8, loc="left")
        ax.set_xticks(range(len(MAIN_GOALS))); ax.set_xticklabels(MAIN_GOALS, fontsize=6, rotation=30, ha="right")
        ax.tick_params(labelsize=7); style(ax)
    axes[0].set_ylabel("Percent of linings", fontsize=8); axes[0].set_ylim(0, 75)
    h, lab = axes[2].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(OUT / "fig_goals_valence.pdf"); plt.close(fig)

    # ---- appendix: the two generating models side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    MCOL = {"gpt": "#0072B2", "ds": "#E69F00"}
    for j, mdl in enumerate(A.MODELS):
        s = mean_over(Ms, lambda M: share([M[b] for b in imp if M[b]["model"] == mdl], "goal", "n_lin", MAIN_GOALS))
        ax1.bar([i + (j - 0.5) * 0.34 for i in range(len(MAIN_GOALS))], [s[g] for g in MAIN_GOALS], width=0.34, color=MCOL[mdl], label=MODEL_LABEL[mdl])
    ax1.set_xticks(range(len(MAIN_GOALS))); ax1.set_xticklabels(MAIN_GOALS, fontsize=7)
    ax1.set_ylabel("Percent of linings", fontsize=8); ax1.set_ylim(0, 55); ax1.tick_params(labelsize=7); ax1.legend(fontsize=7, frameon=False)
    LANG = (("en", ("us_en", "th_en")), ("th", ("us_th", "th_th")))
    bars = (("not a lining", lambda sel: 100 * sum(m["type"]["none"] for m in sel) / sum(m["n"] for m in sel)),
            ("other-benefit", other_benefit))
    x, ticks, labels = 0, [], []
    for name, fn in bars:
        for lang, cells in LANG:
            for j, mdl in enumerate(A.MODELS):
                val = mean_over(Ms, lambda M: fn([M[b] for b in imp if M[b]["model"] == mdl and M[b]["cell"] in cells]))
                ax2.bar(x + j * 0.38, val, width=0.36, color=MCOL[mdl], alpha=1.0 if lang == "en" else 0.55)
            ticks.append(x + 0.19); labels.append("English" if lang == "en" else "Thai"); x += 1.1
        x += 0.5
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel("Percent", fontsize=8); ax2.set_ylim(0, 32); ax2.tick_params(labelsize=7)
    for i, (name, _) in enumerate(bars):
        ax2.text(i * 2.7 + 0.75, 29, name, ha="center", fontsize=7.5)
    for ax in (ax1, ax2): style(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_models.pdf"); plt.close(fig)
    print(f"wrote 7 figures to {OUT}")


if __name__ == "__main__":
    main()
