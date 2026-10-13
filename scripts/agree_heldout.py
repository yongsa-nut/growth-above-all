# -*- coding: utf-8 -*-
"""Agreement on the 36 held-out lists (codebook v4): the two blind human passes and any LLM coders.

    python agree_heldout.py N=heldout/heldout-N.csv PS=heldout/heldout-PS.csv \
        sol61=llm/sol61-all/codes_sol61_all.csv opus55=llm/opus55-all/codes_opus55_all.csv [--out results-heldout.md]

Every file is cut to the held-out lists (gen/run/blocks.heldout.json), so a coder's full 864-list export can be passed
as is. Writes Markdown with:
  1. completeness per coder;
  2. pairwise agreement (percent and Krippendorff's alpha, nominal), overall and by language and generating model:
     item type (four-way and lining vs not), beneficiary (self vs others, and five-way) and goal (fine, and the
     computed self / social / beyond level) over the items both coders called linings, consolation type;
  3. other-benefit share per coder and cell;
  4. each coder's language / setting / interaction contrasts on these nine schemas (analyze.py's definitions;
     95% bootstrap over schemas), so the human passes give their own estimate of the contrasts;
  5. measurement error of each LLM coder on the beneficiary, against the items on which both humans agree:
     sensitivity and specificity for other-benefit linings, overall and by language, the attenuation factor
     Se + Sp - 1, and what a true 10-point shift would look like through that coder.
Pairs are pairwise complete: a pair uses the items both of its coders coded.
"""
import collections, csv, json, math, pathlib, random, statistics, sys, tempfile
sys.stdout.reconfigure(encoding="utf-8")
import agree as G
import analyze as A

HERE = pathlib.Path(__file__).resolve().parent
HB = json.loads((HERE / "gen/run/blocks.heldout.json").read_text(encoding="utf-8"))["blocks"]
HIDS = {b["block_id"]: b for b in HB}
COARSE = {"relational": "social", "community": "social", "transcendent": "beyond"}
OTHER = ("other", "group")


def load(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["block_id"] in HIDS]
    return {(r["block_id"], int(r["item_idx"])): r for r in rows}, rows


def lang(k): return HIDS[k[0]]["cell"].split("_")[1]
def gmod(k): return HIDS[k[0]]["model"]
def coarse(g): return COARSE.get(g, "self") if g else ""
def ben2(b): return "" if not b else ("others" if b in OTHER else "self")


def pa(pairs):
    if not pairs: return "–"
    p = 100 * sum(a == b for a, b in pairs) / len(pairs)
    try: al = G.alpha(pairs)
    except Exception: al = float("nan")
    return f"{p:.0f}% / α {al:.2f} (n {len(pairs)})"


def split(pairs_by_key, keys):
    """'overall [en x · th y · gpt z · ds w]'"""
    def sub(f): return [pairs_by_key[k] for k in keys if f(k)]
    allp = [pairs_by_key[k] for k in keys]
    if not allp: return "–"
    def short(ps):
        return "–" if not ps else f"{100 * sum(a == b for a, b in ps) / len(ps):.0f}%"
    return (f"{pa(allp)}<br><small>en {short(sub(lambda k: lang(k) == 'en'))} · th {short(sub(lambda k: lang(k) == 'th'))}"
            f" · gpt {short(sub(lambda k: gmod(k) == 'gpt'))} · ds {short(sub(lambda k: gmod(k) == 'ds'))}</small>")


def pair_table(C):
    names = list(C)
    out = ["## 2. Pairwise agreement", "",
           "Percent agreement and Krippendorff's α (nominal); below each, percent agreement by language of the list and by "
           "generating model. Goal and beneficiary are over the items both coders called linings.", "",
           "| Pair | Item type (4-way) | Lining vs not | Beneficiary (self vs others) | Beneficiary (5-way) | Goal (fine) | Goal (self / social / beyond) | Consolation type |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = C[names[i]], C[names[j]]
            ks = sorted(k for k in a if k in b and a[k]["item_type"] and b[k]["item_type"])
            it = {k: (a[k]["item_type"], b[k]["item_type"]) for k in ks}
            ln = {k: ("lining" if a[k]["item_type"] == "lining" else "not", "lining" if b[k]["item_type"] == "lining" else "not") for k in ks}
            both = [k for k in ks if a[k]["item_type"] == b[k]["item_type"] == "lining"]
            b2 = {k: (ben2(a[k]["beneficiary"]), ben2(b[k]["beneficiary"])) for k in both}
            b5 = {k: (a[k]["beneficiary"], b[k]["beneficiary"]) for k in both}
            gf = {k: (a[k]["goal"], b[k]["goal"]) for k in both}
            gc = {k: (coarse(a[k]["goal"]), coarse(b[k]["goal"])) for k in both}
            cons = [k for k in ks if a[k]["item_type"] == b[k]["item_type"] == "consolation"]
            ct = {k: (a[k]["consolation_type"], b[k]["consolation_type"]) for k in cons}
            out.append(f"| {names[i]} vs {names[j]} | {split(it, ks)} | {split(ln, ks)} | {split(b2, both)} | {split(b5, both)} | "
                       f"{split(gf, both)} | {split(gc, both)} | {pa(list(ct.values()))} |")
    return out


def share_table(C):
    out = ["## 3. Other-benefit share per cell", "", "Share of each coder's linings whose beneficiary is a specific other or the group.", "",
           "| Coder | linings | us_en | us_th | th_th | th_en | all |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, c in C.items():
        lin = [k for k in c if c[k]["item_type"] == "lining"]
        def sh(ks):
            return f"{100 * sum(c[k]['beneficiary'] in OTHER for k in ks) / len(ks):.0f}% ({sum(c[k]['beneficiary'] in OTHER for k in ks)}/{len(ks)})" if ks else "–"
        out.append(f"| {n} | {len(lin)} | " + " | ".join(sh([k for k in lin if HIDS[k[0]]["cell"] == cell]) for cell in A.CELLS) + f" | {sh(lin)} |")
    return out


def contrast_table(raw, B=2000, seed=20261012):
    out = ["## 4. Contrasts on the nine held-out schemas", "",
           "Each coder's other-benefit share, contrasted as in `analyze.py` (language: Thai minus English text of the same event; "
           "setting: Thai minus US event; interaction: diagonal minus off-diagonal), one list per cell, so the interval is a "
           "bootstrap over the nine schemas. Nine schemas make these intervals wide; they show whether the human passes and the "
           "LLM coders point the same way, not a test of the hypotheses.", "",
           "| Coder | language | setting | interaction |", "| --- | --- | --- | --- |"]
    tmp = pathlib.Path(tempfile.mkdtemp())
    for n, rows in raw.items():
        p = tmp / f"{n}.csv"
        cols = list(rows[0].keys())
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        M = A.load(str(p)); S = A.setups(M, sorted(M))
        est = A.estimates(A.contrast_rows(M, S, "owner_share"))
        rows_c = A.contrast_rows(M, S, "owner_share")
        schemas = sorted({s for (s, m, a) in rows_c})
        rnd = random.Random(seed); draws = collections.defaultdict(list)
        for _ in range(B):
            pick = [rnd.choice(schemas) for _ in schemas]
            for kind in A.KINDS:
                vals = [rows_c[(s, m, a)][kind] for s in pick for (s2, m, a) in rows_c if s2 == s and rows_c[(s, m, a)][kind] is not None]
                if vals: draws[kind].append(100 * statistics.mean(vals))
        cells = []
        for kind in A.KINDS:
            e = est[("imp", "both", kind)]; d = sorted(draws[kind])
            ci = (d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]) if d else None
            cells.append(f"{A.pts(e)} {A.ci_s(ci)}")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return out


def wilson(x, n):
    if n == 0: return (float("nan"),) * 2
    z = 1.96; p = x / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def error_table(C, humans):
    if len(humans) < 2:
        return ["## 5. Measurement error", "", "Needs both human passes."]
    h1, h2 = (C[h] for h in humans)
    ref = {k: h1[k]["beneficiary"] for k in h1 if k in h2 and h1[k]["item_type"] == h2[k]["item_type"] == "lining"
           and ben2(h1[k]["beneficiary"]) == ben2(h2[k]["beneficiary"]) and h1[k]["beneficiary"]}
    out = ["## 5. Measurement error of the LLM coders on the beneficiary", "",
           f"Reference: the {len(ref)} items both human coders called linings and gave the same beneficiary (self vs others). "
           "Sensitivity = share of reference other-benefit linings the coder also codes as other or group; specificity = share of "
           "reference self linings it codes as self; both over the reference items the coder also calls linings. Under "
           "misclassification that is the same in every cell, a true difference in other-benefit share shrinks by the factor "
           "Se + Sp − 1; a difference between the languages in Se or Sp would instead add a bias to the language contrast.", "",
           "| Coder | subset | Se (95% CI) | Sp (95% CI) | Se + Sp − 1 | a true 10-point shift appears as |", "| --- | --- | --- | --- | --- | --- |"]
    for n, c in C.items():
        if n in humans: continue
        for lab, f in (("all", lambda k: True), ("English lists", lambda k: lang(k) == "en"), ("Thai lists", lambda k: lang(k) == "th")):
            ks = [k for k in ref if f(k) and k in c and c[k]["item_type"] == "lining"]
            pos = [k for k in ks if ref[k] in OTHER]; neg = [k for k in ks if ref[k] not in OTHER]
            tp = sum(c[k]["beneficiary"] in OTHER for k in pos); tn = sum(c[k]["beneficiary"] not in OTHER for k in neg)
            if not pos or not neg:
                out.append(f"| {n} | {lab} | – | – | – | – |"); continue
            se, sp = tp / len(pos), tn / len(neg); lo1, hi1 = wilson(tp, len(pos)); lo2, hi2 = wilson(tn, len(neg))
            fac = se + sp - 1
            out.append(f"| {n} | {lab} | {100 * se:.0f}% ({tp}/{len(pos)}; {100 * lo1:.0f}–{100 * hi1:.0f}) | "
                       f"{100 * sp:.0f}% ({tn}/{len(neg)}; {100 * lo2:.0f}–{100 * hi2:.0f}) | {fac:.2f} | {10 * fac:.1f} points |")
    return out


def main(args):
    out_arg = pathlib.Path(args[args.index("--out") + 1]) if "--out" in args else pathlib.Path("results-heldout.md")
    out_path = out_arg if out_arg.is_absolute() else HERE / out_arg
    specs = [a for a in args if "=" in a and not a.startswith("--")]
    C, raw = {}, {}
    for s in specs:
        n, p = s.split("=", 1)
        C[n], raw[n] = load(HERE / p if not pathlib.Path(p).is_absolute() else p)
    humans = [n for n in C if not any(x in n.lower() for x in ("sol", "opus", "llm", "gpt", "claude"))][:2]
    total = sum(len(b["items"]) for b in HB)
    L = ["# Held-out agreement (codebook v4)", "",
         f"{len(HB)} held-out lists, {total} items; coders: " + ", ".join(f"`{n}`" for n in C) +
         (f"; human passes: {', '.join(humans)}" if humans else ""), "",
         "## 1. Completeness", "", "| Coder | items coded | linings | consolation | action | not a lining |", "| --- | --- | --- | --- | --- | --- |"]
    for n, c in C.items():
        t = collections.Counter(r["item_type"] for r in c.values() if r["item_type"])
        L.append(f"| {n} | {sum(t.values())}/{total} | {t['lining']} | {t['consolation']} | {t['action']} | {t['none']} |")
    L += [""] + pair_table(C) + [""] + share_table(C) + [""] + contrast_table(raw) + [""] + error_table(C, humans)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
