# -*- coding: utf-8 -*-
"""Score a coder's export of the known-answer lists against the answer key.

    python known/score_known.py [llm/v4-known/codes_sol4_known.csv]   # Markdown to known/results-known-<coder>.md
"""
import collections, csv, json, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "llm/v4-known/codes_sol4_known.csv"
K = json.loads((HERE / "known_items.json").read_text(encoding="utf-8"))
NORM = {"could be worse": "could_be_worse", "fate / meaning": "fate_meaning"}
key = {}
for L in K["lists"]:
    for i, it in enumerate(L["items"], 1):
        key[(L["list"], i)] = dict(type=it["type"], cons=NORM.get(it.get("cons", ""), it.get("cons", "")), goal=it.get("goal", ""),
                                   ben=it.get("ben", ""), tests=it["tests"], cultural=bool(it.get("cultural")), th=it["th"], en=it["en"])
rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
coder = rows[0]["coder"]

res = []
for r in rows:
    lst, lang = r["block_id"].rsplit("-", 1); i = int(r["item_idx"]); k = key[(lst, i)]
    got = dict(type=r["item_type"], cons=r["consolation_type"], goal=r["goal"], ben=r["beneficiary"])
    t_ok = got["type"] == k["type"]
    g_ok = (got["goal"] == k["goal"]) if k["type"] == "lining" and t_ok else None
    b_ok = (got["ben"] == k["ben"]) if k["type"] == "lining" and t_ok else None
    c_ok = (got["cons"] == k["cons"]) if k["type"] == "consolation" and t_ok else None
    full = t_ok and all(x is not False for x in (g_ok, b_ok, c_ok))
    res.append(dict(id=f"{lst}-{i}", lang=lang, k=k, got=got, t_ok=t_ok, g_ok=g_ok, b_ok=b_ok, c_ok=c_ok, full=full, note=r["note"]))


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "–"


out = [f"# Known-answer items: {coder} against the key", "",
       f"Source: `{src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}`. 72 items, each coded once in Thai and once in English. "
       "Goal, beneficiary and consolation type are scored over the items whose item type the coder got right.", "",
       "| | Thai | English | Both |", "| --- | --- | --- | --- |"]
for lab, f in (("Item type", "t_ok"), ("Goal (linings)", "g_ok"), ("Beneficiary (linings)", "b_ok"),
               ("Consolation type", "c_ok"), ("Everything right", "full")):
    out.append(f"| {lab} | {pct([r[f] for r in res if r['lang'] == 'th'])} | {pct([r[f] for r in res if r['lang'] == 'en'])} | {pct([r[f] for r in res])} |")
cult = [r for r in res if r["k"]["cultural"]]
out += ["", f"Items with Thai cultural content: everything right on {pct([r['full'] for r in cult if r['lang'] == 'th'])} in Thai "
        f"and {pct([r['full'] for r in cult if r['lang'] == 'en'])} in English.", ""]

out += ["## Item type: key (rows) against the coder (columns)", "", "| key \\ coder | lining | consolation | action | none |", "| --- | --- | --- | --- | --- |"]
cm = collections.Counter((r["k"]["type"], r["got"]["type"]) for r in res)
for a in ("lining", "consolation", "action", "none"):
    out.append(f"| {a} | " + " | ".join(str(cm[(a, b)]) for b in ("lining", "consolation", "action", "none")) + " |")

out += ["", "## Goal: key (rows) against the coder (columns), linings with the right item type", ""]
G = ["transcendent", "community", "relational", "growth", "achievement", "wellbeing", "resources"]
gm = collections.Counter((r["k"]["goal"], r["got"]["goal"]) for r in res if r["g_ok"] is not None)
out += ["| key \\ coder | " + " | ".join(G) + " |", "| --- |" + " --- |" * len(G)]
for a in G:
    out.append(f"| {a} | " + " | ".join(str(gm[(a, b)] or "·") for b in G) + " |")

miss = [r for r in res if not r["full"]]
out += ["", f"## Every miss ({len(miss)})", "",
        "Same item in both languages is listed twice when both missed. Key → coder.", ""]
for r in sorted(miss, key=lambda r: (int(r["id"].split("-")[0][1:]), int(r["id"].split("-")[1]), r["lang"])):
    k, g = r["k"], r["got"]
    def fmt(d):
        if d["type"] == "lining": return f"lining · {d['goal']} · {d['ben']}"
        if d["type"] == "consolation": return f"consolation · {d['cons']}"
        return d["type"] or "(blank)"
    text = k["th"] if r["lang"] == "th" else k["en"]
    out.append(f"- **{r['id']} {r['lang']}**: {fmt(k)} → **{fmt(g)}** · {text[:120]}{'…' if len(text) > 120 else ''}"
               + (f"  \n  *tests: {k['tests']}*" ) + (f" · coder note: {r['note']}" if r['note'] else ""))

dst = HERE / f"results-known-{coder}.md"
dst.write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[:14])); print(f"... {len(miss)} misses; written to {dst}")
