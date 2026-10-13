# -*- coding: utf-8 -*-
"""Agreement between two coding exports in the coder.html layout (human or LLM).

    python agree.py A.csv B.csv          # per question: n, percent agreement, Krippendorff's alpha (nominal)
    python agree.py A.csv B.csv --list   # also every disagreement, one line each

Denominators follow the codebook: item type on all items; consolation type where both said consolation;
grounding, goal, owner, redundancy where both said lining (redundancy as none vs a number). Owner share
(other + group over linings) is shown per coder and cell.
"""
import csv, sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import Counter, defaultdict


def load(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return rows[0]["coder"], {(r["block_id"], r["item_idx"]): r for r in rows}


def alpha(pairs):
    """Krippendorff's alpha, nominal, two coders, from a list of (a, b) value pairs."""
    if len(pairs) < 2:
        return float("nan")
    o = Counter(); n_c = Counter()
    for a, b in pairs:
        o[(a, b)] += 1; o[(b, a)] += 1; n_c[a] += 1; n_c[b] += 1
    n = sum(n_c.values())
    Do = sum(v for (a, b), v in o.items() if a != b)
    De = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n - 1)
    return 1 - Do / De if De else float("nan")


def report(A, B, names, listing):
    keys = sorted(set(A) & set(B), key=lambda k: (k[0], int(k[1])))
    lin = lambda r: r["item_type"] == "lining"
    qs = [("item_type", lambda a, b: True, lambda r: r["item_type"]),
          ("consolation_type", lambda a, b: a["item_type"] == b["item_type"] == "consolation", lambda r: r["consolation_type"]),
          ("grounding", lambda a, b: lin(a) and lin(b), lambda r: r["grounding"]),
          ("goal", lambda a, b: lin(a) and lin(b), lambda r: r["goal"]),
          ("owner", lambda a, b: lin(a) and lin(b), lambda r: r["owner"]),
          ("redundant", lambda a, b: lin(a) and lin(b), lambda r: "none" if r["redundant_with"] in ("", "none") else "yes")]
    print(f"coders {names[0]} vs {names[1]}: {len(keys)} items in common")
    print(f"{'question':18}{'n':>5}{'agree':>8}{'alpha':>8}   by language (n, agree)")
    dis = defaultdict(list)
    for q, cond, val in qs:
        pairs = [(val(A[k]), val(B[k]), k) for k in keys if cond(A[k], B[k])]
        if not pairs:
            print(f"{q:18}{0:>5}"); continue
        agree = sum(a == b for a, b, _ in pairs) / len(pairs)
        al = alpha([(a, b) for a, b, _ in pairs])
        bylang = []
        for lang in ("en", "th"):
            sub = [(a, b) for a, b, k in pairs if A[k]["cell"].endswith(lang)]
            if sub: bylang.append(f"{lang} {len(sub)} {100 * sum(a == b for a, b in sub) / len(sub):.0f}%")
        print(f"{q:18}{len(pairs):>5}{100 * agree:>7.0f}%{al:>8.2f}   {', '.join(bylang)}")
        for a, b, k in pairs:
            if a != b: dis[q].append((k, a, b))
    print("\nowner share (other + group over linings) by cell:")
    for cell in ("us_en", "us_th", "th_th", "th_en"):
        parts = []
        for R, name in ((A, names[0]), (B, names[1])):
            L = [r for k, r in R.items() if k in set(keys) and r["cell"] == cell and lin(r)]
            g = sum(r["owner"] in ("other", "group") for r in L)
            parts.append(f"{name} {g}/{len(L)} = {100 * g / len(L):.0f}%" if L else f"{name} -")
        print(f"  {cell}: " + " | ".join(parts))
    print("\nitem-type confusion (rows " + names[0] + ", columns " + names[1] + "):")
    types = ["lining", "consolation", "advice", "none"]
    conf = Counter((A[k]["item_type"], B[k]["item_type"]) for k in keys)
    print(f"{'':14}" + "".join(f"{t:>13}" for t in types))
    for t in types:
        print(f"{t:14}" + "".join(f"{conf[(t, u)]:>13}" for u in types))
    if listing:
        for q in dis:
            print(f"\n--- {q}: {len(dis[q])} disagreements")
            for (bid, idx), a, b in dis[q]:
                print(f"  [{bid} #{idx}] {names[0]}={a} {names[1]}={b} :: {A[(bid, idx)]['item_text'][:110].replace(chr(10), ' / ')}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit("usage: python agree.py A.csv B.csv [--list]")
    (na, A), (nb, B) = load(a[0]), load(a[1])
    report(A, B, (na, nb), "--list" in a)
