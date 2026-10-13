# -*- coding: utf-8 -*-
"""Pre-specified analysis on an LLM-coded export (README § "Primary outcome and analysis"; draft §4).

    python analyze.py [llm/all/codes_sol_all.csv] [--out results-llm.md] [--boot 2000] [--seed 20260919] [--figdir figures]
    (also reads a codebook v4 export from llmcode4.py / llmcoders.py: action counts where advice did, beneficiary as owner)

Unit = block; block metrics averaged within setup (schema × cell × model × arm); language / setting / interaction
contrasts within schema and model; estimate = equal-weight mean over schemas; CI = percentile bootstrap over blocks
within setups (draft §4.2); a block with no lining has no owner share and drops out of the contrasts it cannot
enter. Reading rule: CI excludes 0 and |estimate| >= 10 points -> shift; CI inside ±10 ->
no shift; else inconclusive. Writes a Markdown results file and figures/*.png. Tables only, no interpretation.
"""
import csv, json, sys, random, statistics, pathlib, collections
sys.stdout.reconfigure(encoding="utf-8")
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
CELLS = ("us_en", "us_th", "th_th", "th_en"); MODELS = ("gpt", "ds"); ARMS = ("imp", "exp"); PAIRS = ("P1", "P2", "P3")
TYPES = ("lining", "consolation", "advice", "none")
GOALS = ("growth", "achievement", "relational", "community", "transcendent", "wellbeing", "practical", "status", "unspecified")
OWNERS = ("self", "other", "group", "transcendent", "unspecified")
CTYPES = ("acceptance", "impermanence", "could_be_worse", "reassurance", "fate_meaning", "other")
KINDS = ("language", "setting", "interaction")
# Codebook v4 (camera-ready): action replaces advice, practical is renamed resources, status folds into achievement.
# A v4 export is recognized by its beneficiary column; load() then switches TYPES and GOALS for the tables and figures.
TYPES_V4 = ("lining", "consolation", "action", "none")
GOALS_V4 = ("growth", "relational", "wellbeing", "resources", "achievement", "community", "transcendent", "unspecified")
FIGDIR = None
SESOI = 10.0
BAD_GEN = {"n_items", "lang_mismatch", "other_script"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


# ---------------------------------------------------------------- load
def load(codes_csv):
    stim = {r["schema_id"]: r for r in csv.DictReader(open(HERE / "stimuli.csv", encoding="utf-8-sig"))}
    gen = {b["block_id"]: b for b in json.load(open(HERE / "gen/run/blocks.json", encoding="utf-8"))["blocks"]}
    rows = collections.defaultdict(list)
    global TYPES, GOALS
    for r in csv.DictReader(open(codes_csv, encoding="utf-8-sig")):
        if "beneficiary" in r and "owner" not in r:          # v4 export: same measures, new column names
            TYPES, GOALS = TYPES_V4, GOALS_V4
            r["owner"] = r["beneficiary"]
            for k in ("grounding", "redundant_with", "block_flags", "block_flag_note"):
                r.setdefault(k, "")
        rows[r["block_id"]].append(r)
    M = {}
    for bid, rs in rows.items():
        r0 = rs[0]; g = gen[bid]; n = len(rs); lin = [r for r in rs if r["item_type"] == "lining"]
        m = dict(schema=r0["schema_id"], cell=r0["cell"], model=r0["model"], arm=r0["arm"], pair=r0["pair"], core=g["core"],
                 domain=stim[r0["schema_id"]]["domain"], valence=stim[r0["schema_id"]]["valence"],
                 family=(int(r0["schema_id"][1:]) - 1) // 3, n=n, n_lin=len(lin),
                 type=collections.Counter(r["item_type"] for r in rs),
                 goal=collections.Counter(r["goal"] for r in lin), owner=collections.Counter(r["owner"] for r in lin),
                 ctype=collections.Counter(r["consolation_type"] for r in rs if r["item_type"] == "consolation"),
                 owner_share=(sum(r["owner"] in ("other", "group") for r in lin) / len(lin)) if lin else None,
                 relcomm=(sum(r["goal"] in ("relational", "community") for r in lin) / len(lin)) if lin else None,
                 consol=sum(r["item_type"] == "consolation" for r in rs) / n,
                 accept=sum(r["item_type"] == "consolation" and r["consolation_type"] in ("acceptance", "fate_meaning") for r in rs) / n,
                 advice=sum(r["item_type"] in ("advice", "action") for r in rs) / n, none=sum(r["item_type"] == "none" for r in rs) / n,
                 unsup=(sum(r["grounding"] == "unsupported" for r in lin) / len(lin)) if lin else None,
                 redund=(sum(r["redundant_with"] not in ("", "none") for r in lin) / len(lin)) if lin else None,
                 length=statistics.mean(len(r["item_text"].replace(" ", "")) for r in rs),
                 parse_error=any("parse_error" in r["note"] for r in rs),
                 coder_flags=r0["block_flags"], gen_flags=set(g["flags"]))
        m["flagged"] = bool(m["coder_flags"]) or bool(m["gen_flags"] & BAD_GEN) or m["parse_error"]
        M[bid] = m
    return M


# ---------------------------------------------------------------- contrasts
def setups(M, bids):
    S = collections.defaultdict(list)
    for b in bids:
        m = M[b]; S[(m["schema"], m["cell"], m["model"], m["arm"])].append(b)
    return S


def contrast_rows(M, S, metric, sample=None):
    """rows[(schema, model, arm)] = {kind: value or None}; sample maps setup -> block list (bootstrap)."""
    val = {}
    for key, bl in S.items():
        val[key] = mean(M[b][metric] for b in (sample[key] if sample else bl))
    rows = {}
    for (schema, cell, model, arm) in S:
        k = (schema, model, arm)
        if k in rows: continue
        d = {c: val.get((schema, c, model, arm)) for c in CELLS}
        def diff(a, b): return None if d[a] is None or d[b] is None else d[a] - d[b]
        lang = mean([diff("us_th", "us_en"), diff("th_th", "th_en")])
        sett = mean([diff("th_en", "us_en"), diff("th_th", "us_th")])
        inter = None if any(v is None for v in d.values()) else (d["th_th"] - d["th_en"]) - (d["us_th"] - d["us_en"])
        rows[k] = dict(language=lang, setting=sett, interaction=inter)
    return rows


def estimates(rows):
    """est[(arm, model or 'both', kind)] = mean over defined rows (points); also rq2 diffs exp - imp."""
    est = {}
    for arm in ARMS:
        for model in MODELS + ("both",):
            for kind in KINDS:
                vals = [v[kind] for (s, m, a), v in rows.items() if a == arm and (model == "both" or m == model) and v[kind] is not None]
                est[(arm, model, kind)] = 100 * statistics.mean(vals) if vals else None
    for model in MODELS + ("both",):
        for kind in KINDS:
            vals = []
            for (s, m, a), v in rows.items():
                if a != "imp" or (model != "both" and m != model): continue
                e = rows.get((s, m, "exp"))
                if e and v[kind] is not None and e[kind] is not None: vals.append(e[kind] - v[kind])
            est[("rq2", model, kind)] = 100 * statistics.mean(vals) if vals else None
    return est


def bootstrap(M, S, metric, B, seed, over="blocks"):
    rnd = random.Random(seed); draws = collections.defaultdict(list)
    schemas = sorted(set(k[0] for k in S))
    for _ in range(B):
        if over == "blocks":
            sample = {key: [rnd.choice(bl) for _ in bl] for key, bl in S.items()}
            rows = contrast_rows(M, S, metric, sample)
        else:  # over schemas (for the one-block-per-setup subset)
            pick = [rnd.choice(schemas) for _ in schemas]
            base = contrast_rows(M, S, metric)
            rows = {}
            for i, sch in enumerate(pick):
                for (s, m, a), v in base.items():
                    if s == sch: rows[(f"{s}#{i}", m, a)] = v
        for k, v in estimates(rows).items():
            if v is not None: draws[k].append(v)
    ci = {}
    for k, xs in draws.items():
        xs.sort(); ci[k] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])
    return ci


def reading(est, ci):
    if est is None or ci is None: return "–"
    lo, hi = ci
    if (lo > 0 or hi < 0) and abs(est) >= SESOI: return "shift"
    if lo > -SESOI and hi < SESOI: return "no shift"
    return "inconclusive"


def sign(rows, arm, model, kind):
    vals = [v[kind] for (s, m, a), v in rows.items() if a == arm and (model == "both" or m == model) and v[kind] is not None]
    tol = 1e-9  # exact-zero contrasts carry float residuals (~1e-17); count them as ties, not signs (fixed 19 Sep after review)
    pos = sum(v > tol for v in vals); neg = sum(v < -tol for v in vals)
    p = binomtest(pos, pos + neg).pvalue if pos + neg else float("nan")
    return pos, neg, len(vals), p


# ---------------------------------------------------------------- tables
def pct(x): return "–" if x is None else f"{100 * x:.0f}"
def pts(x): return "–" if x is None else f"{x:+.1f}"
def ci_s(c): return "–" if c is None else f"[{c[0]:+.1f}, {c[1]:+.1f}]"


def descriptives(M, bids, title):
    out = [f"### {title}\n", "| cell | model | blocks | no lining | lining % | consol % | advice % | none % | accept-type % | " +
           " | ".join(g[:6] for g in GOALS) + " | self % | other % | group % | owner share | unsupp % | redund % | len | preamble % | postamble % | ends ? % | flagged |",
           "|" + " --- |" * (9 + len(GOALS) + 11)]
    for cell in CELLS:
        for model in MODELS:
            bl = [M[b] for b in bids if M[b]["cell"] == cell and M[b]["model"] == model]
            if not bl: continue
            items = sum(m["n"] for m in bl); lins = sum(m["n_lin"] for m in bl)
            t = collections.Counter(); g = collections.Counter(); o = collections.Counter(); ct = collections.Counter()
            for m in bl: t.update(m["type"]); g.update(m["goal"]); o.update(m["owner"]); ct.update(m["ctype"])
            row = [cell, model, str(len(bl)), str(sum(m["n_lin"] == 0 for m in bl))]
            row += [f"{100 * t[x] / items:.0f}" for x in TYPES]
            row += [f"{100 * (ct['acceptance'] + ct['fate_meaning']) / items:.0f}"]
            row += [f"{100 * g[x] / lins:.0f}" if lins else "–" for x in GOALS]
            row += [f"{100 * o[x] / lins:.0f}" if lins else "–" for x in ("self", "other", "group")]
            row += [pct(mean(m["owner_share"] for m in bl)), pct(mean(m["unsup"] for m in bl)), pct(mean(m["redund"] for m in bl)),
                    f"{statistics.mean(m['length'] for m in bl):.0f}",
                    f"{100 * sum('preamble' in m['gen_flags'] for m in bl) / len(bl):.0f}", f"{100 * sum('postamble' in m['gen_flags'] for m in bl) / len(bl):.0f}",
                    f"{100 * sum('ends_with_question' in m['gen_flags'] for m in bl) / len(bl):.0f}", str(sum(m["flagged"] for m in bl))]
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def contrast_table(rows, est, ci, arm, title, note=""):
    out = [f"### {title}\n", note, "| contrast | model | estimate (points) | 95% CI | reading | schemas + / − / n | sign test p |", "|" + " --- |" * 7]
    for kind in KINDS:
        for model in MODELS + ("both",):
            e = est.get((arm, model, kind)); c = ci.get((arm, model, kind)) if ci else None
            pos, neg, n, p = sign(rows, arm, model, kind)
            out.append(f"| {kind} | {model} | {pts(e)} | {ci_s(c)} | {reading(e, c)} | {pos} / {neg} / {n} | {p:.2f} |")
    return "\n".join(out) + "\n"


def per_schema_table(rows, M, arm):
    out = [f"### Per-schema contrasts on owner share, {arm} arm (points)\n", "| schema | domain | valence | model | language | setting | interaction |", "|" + " --- |" * 7]
    meta = {m["schema"]: m for m in M.values()}
    for (s, m, a), v in sorted(rows.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
        if a != arm: continue
        out.append(f"| {s} | {meta[s]['domain']} | {meta[s]['valence']} | {m} | " + " | ".join(pts(None if v[k] is None else 100 * v[k]) for k in KINDS) + " |")
    return "\n".join(out) + "\n"


def figures(M, bids, outdir):
    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, model in enumerate(MODELS):
        ys = [100 * (mean(M[b]["owner_share"] for b in bids if M[b]["cell"] == c and M[b]["model"] == model) or 0) for c in CELLS]
        ax.bar([x + (i - 0.5) * 0.36 for x in range(4)], ys, width=0.34, label={"gpt": "GPT-5.6 Luna", "ds": "DeepSeek V4.1 Flash"}[model])
    ax.set_xticks(range(4)); ax.set_xticklabels(CELLS); ax.set_ylabel("owner share (%)"); ax.legend(frameon=False); ax.set_ylim(0, 50)
    fig.tight_layout(); fig.savefig(outdir / "fig1_owner_share.png", dpi=200); plt.close(fig)
    for name, keys, field, denom in (("fig2_goals", GOALS, "goal", "n_lin"), ("fig3_item_type", TYPES, "type", "n")):
        fig, ax = plt.subplots(figsize=(7, 3.4)); labels = []; bottoms = None; xs = []
        cols = [f"{c}\n{m}" for c in CELLS for m in MODELS]
        data = {k: [] for k in keys}
        for c in CELLS:
            for model in MODELS:
                bl = [M[b] for b in bids if M[b]["cell"] == c and M[b]["model"] == model]
                tot = sum(m[denom] for m in bl); cnt = collections.Counter()
                for m in bl: cnt.update(m[field])
                for k in keys: data[k].append(100 * cnt[k] / tot if tot else 0)
        bottom = [0] * len(cols)
        for k in keys:
            ax.bar(cols, data[k], bottom=bottom, label=k); bottom = [b + d for b, d in zip(bottom, data[k])]
        ax.set_ylabel("% of " + ("linings" if field == "goal" else "items")); ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        ax.tick_params(axis="x", labelsize=7); fig.tight_layout(); fig.savefig(outdir / f"{name}.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- main
def main(codes_csv, out, B, seed):
    M = load(codes_csv); all_b = sorted(M)
    imp = [b for b in all_b if M[b]["arm"] == "imp"]; exp = [b for b in all_b if M[b]["arm"] == "exp"]
    S = setups(M, all_b)
    rows = contrast_rows(M, S, "owner_share"); est = estimates(rows); ci = bootstrap(M, S, "owner_share", B, seed)
    L = [f"# Results on the LLM-coded sample\n", f"Source `{codes_csv}`: {len(all_b)} blocks ({len(imp)} implicit, {len(exp)} explicit), "
         f"{sum(M[b]['n'] for b in all_b)} items; bootstrap {B} draws over blocks within setups, seed {seed}; smallest shift of interest {SESOI:.0f} points.\n",
         f"Parse errors {sum(M[b]['parse_error'] for b in all_b)}; blocks with no lining {sum(M[b]['n_lin'] == 0 for b in all_b)} "
         f"(by cell: " + ", ".join(f"{c} {sum(M[b]['n_lin'] == 0 and M[b]['cell'] == c for b in all_b)}" for c in CELLS) + "); "
         f"flagged blocks (generation n_items / language flags, coder block flags, parse errors) {sum(M[b]['flagged'] for b in all_b)}.\n",
         "## 1. Descriptives\n", descriptives(M, imp, "Implicit arm (main), per cell × model"), descriptives(M, exp, "Explicit arm, per cell × model"),
         "## 2. Primary: owner share contrasts\n",
         contrast_table(rows, est, ci, "imp", "Implicit arm", "Language = Thai text minus English text of the same event, averaged over settings; setting = Thai event minus US event, averaged over languages; interaction = diagonal minus off-diagonal. Equal weight per schema; sign test over schema × model rows.\n"),
         per_schema_table(rows, M, "imp"),
         "## 3. Explicit arm and RQ2\n", contrast_table(rows, est, ci, "exp", "Explicit arm"),
         "### Explicit minus implicit (points, same bootstrap)\n", "| contrast | model | difference | 95% CI |", "|" + " --- |" * 4]
    for kind in KINDS:
        for model in MODELS + ("both",):
            L.append(f"| {kind} | {model} | {pts(est.get(('rq2', model, kind)))} | {ci_s(ci.get(('rq2', model, kind)))} |")
    # leave-one-family-out
    L += ["\n### Leave-one-story-family-out, implicit arm, both models (range of estimates over the six drops)\n", "| contrast | min | max |", "| --- | --- | --- |"]
    for kind in KINDS:
        vals = []
        for fam in range(6):
            sub = [b for b in all_b if M[b]["family"] != fam]
            vals.append(estimates(contrast_rows(M, setups(M, sub), "owner_share"))[("imp", "both", kind)])
        L.append(f"| {kind} | {pts(min(vals))} | {pts(max(vals))} |")
    # wording
    L += ["\n## 4. Secondary\n", "### Wording pairs (implicit arm)\n", "| pair | owner share | consol % | advice % | language contrast | setting contrast |", "|" + " --- |" * 6]
    spread = collections.defaultdict(list)
    for pair in PAIRS:
        sub = [b for b in imp if M[b]["pair"] == pair]; e = estimates(contrast_rows(M, setups(M, sub), "owner_share"))
        spread["language"].append(e[("imp", "both", "language")]); spread["setting"].append(e[("imp", "both", "setting")])
        L.append(f"| {pair} | {pct(mean(M[b]['owner_share'] for b in sub))} | {pct(mean(M[b]['consol'] for b in sub))} | {pct(mean(M[b]['advice'] for b in sub))} | {pts(e[('imp', 'both', 'language')])} | {pts(e[('imp', 'both', 'setting')])} |")
    L.append(f"\nSpread between pairs: language {max(spread['language']) - min(spread['language']):.1f} points, setting {max(spread['setting']) - min(spread['setting']):.1f} points (against the language effect above).\n")
    # valence
    L += ["### Valence (implicit arm, both models, pooled over cells)\n", "| valence | blocks | consol % | accept-type % | advice % | unsupp % | owner share | top goals |", "|" + " --- |" * 8]
    for val in ("pos", "neg", "neu"):
        sub = [M[b] for b in imp if M[b]["valence"] == val]; g = collections.Counter()
        for m in sub: g.update(m["goal"])
        tot = sum(g.values()); top = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in g.most_common(3))
        L.append(f"| {val} | {len(sub)} | {pct(mean(m['consol'] for m in sub))} | {pct(mean(m['accept'] for m in sub))} | {pct(mean(m['advice'] for m in sub))} | {pct(mean(m['unsup'] for m in sub))} | {pct(mean(m['owner_share'] for m in sub))} | {top} |")
    # other summaries as contrasts (descriptive)
    L += ["\n### Other summaries, implicit-arm contrasts (points; descriptive, same bootstrap)\n", "| summary | language | setting | interaction |", "|" + " --- |" * 4]
    for name, metric in (("relational + community share", "relcomm"), ("consolation rate", "consol"), ("acceptance-type rate", "accept"), ("advice rate", "advice"), ("unsupported rate", "unsup")):
        r2 = contrast_rows(M, S, metric); e2 = estimates(r2); c2 = bootstrap(M, S, metric, max(200, B // 4), seed + 1)
        L.append(f"| {name} | " + " | ".join(f"{pts(e2[('imp', 'both', k)])} {ci_s(c2.get(('imp', 'both', k)))}" for k in KINDS) + " |")
    # sensitivity
    L += ["\n## 5. Sensitivity (implicit arm, both models, owner share unless stated)\n", "| variant | blocks | language | setting | interaction |", "|" + " --- |" * 5]
    def sens(name, bids, metric="owner_share", over="blocks"):
        S2 = setups(M, bids); r2 = contrast_rows(M, S2, metric); e2 = estimates(r2); c2 = bootstrap(M, S2, metric, max(200, B // 4), seed + 2, over)
        L.append(f"| {name} | {len(bids)} | " + " | ".join(f"{pts(e2[('imp', 'both', k)])} {ci_s(c2.get(('imp', 'both', k)))} {reading(e2[('imp', 'both', k)], c2.get(('imp', 'both', k)))}" for k in KINDS) + " |")
    sens("all implicit blocks (primary)", imp)
    sens("excluding flagged blocks", [b for b in imp if not M[b]["flagged"]])
    sens("relational + community share as summary", imp, "relcomm")
    sens("designed core subset (one block per text; bootstrap over schemas)", [b for b in imp if M[b]["core"]], over="schemas")
    fd = pathlib.Path(FIGDIR) if FIGDIR else HERE / "figures"; figures(M, imp, fd)
    fn = fd.name if fd.resolve().parent == HERE else str(fd)
    L.append(f"\nFigures: `{fn}/fig1_owner_share.png` (implicit arm, block means by cell and model), `{fn}/fig2_goals.png`, `{fn}/fig3_item_type.png`.\n")
    pathlib.Path(out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}; primary (imp, both): " + ", ".join(f"{k} {pts(est[('imp', 'both', k)])} {ci_s(ci.get(('imp', 'both', k)))} {reading(est[('imp', 'both', k)], ci.get(('imp', 'both', k)))}" for k in KINDS))


if __name__ == "__main__":
    a = sys.argv[1:]
    src = a[0] if a and not a[0].startswith("--") else "llm/all/codes_sol_all.csv"
    out = a[a.index("--out") + 1] if "--out" in a else "results-llm.md"
    B = int(a[a.index("--boot") + 1]) if "--boot" in a else 2000
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 20260919
    FIGDIR = a[a.index("--figdir") + 1] if "--figdir" in a else None
    main(src, out, B, seed)
