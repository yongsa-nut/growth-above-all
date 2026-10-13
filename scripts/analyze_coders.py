# -*- coding: utf-8 -*-
"""Two-coder reading of the other-benefit contrasts (both must agree).

    python analyze_coders.py sol61=llm/sol61-all/codes_sol61_all.csv opus55=llm/opus55-all/codes_opus55_all.csv \
        [--out results-v4-coders.md] [--boot 2000] [--seed 20260919]

For every contrast (language, setting, interaction; implicit and explicit arm; per generating model and pooled;
and the explicit - implicit change) it reports each coder's estimate and 95% interval exactly as analyze.py computes
them, the pooled estimate (mean over coders) with an interval whose bootstrap resamples lists within setups (the same
resample for every coder) and then picks one coder at random per draw, and the final reading: a shift, or no shift of
practical interest, only when that reading holds under every coder's own interval and under the pooled one;
otherwise inconclusive. Tables only.
"""
import collections, random, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import analyze as A

HERE = pathlib.Path(__file__).resolve().parent


def ci(xs):
    if not xs: return None
    xs = sorted(xs); return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])


def two_coder(Ms, B=2000, seed=20260919, metric="owner_share"):
    """Per-coder estimates and bootstrap draws, the pooled estimate and its coder-resampled draws (also used by
    paper/make_figures_v4.py). Returns names, common lists, est, pooled, own draws, pooled draws, keys."""
    names = list(Ms)
    common = sorted(set.intersection(*(set(M) for M in Ms.values())))
    S = A.setups(Ms[names[0]], common)
    est = {n: A.estimates(A.contrast_rows(Ms[n], S, metric)) for n in names}
    keys = [k for k in est[names[0]] if all(est[n][k] is not None for n in names)]
    pooled = {k: sum(est[n][k] for n in names) / len(names) for k in keys}
    # the resamples use analyze.py's stream (same seed, same order), so each coder's interval equals analyze.py's;
    # the coder picked for the pooled draw comes from a separate stream
    rnd = random.Random(seed); prnd = random.Random(seed + 1)
    own = {n: collections.defaultdict(list) for n in names}; mix = collections.defaultdict(list)
    for _ in range(B):
        sample = {key: [rnd.choice(bl) for _ in bl] for key, bl in S.items()}
        e = {n: A.estimates(A.contrast_rows(Ms[n], S, metric, sample)) for n in names}
        pick = prnd.choice(names)
        for k in keys:
            for n in names:
                if e[n][k] is not None: own[n][k].append(e[n][k])
            if e[pick][k] is not None: mix[k].append(e[pick][k])
    return names, common, est, pooled, own, mix, keys


def main(args):
    specs = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]
    out = HERE / pathlib.Path(args[args.index("--out") + 1] if "--out" in args else "results-v4-coders.md")  # absolute paths kept
    B = int(args[args.index("--boot") + 1]) if "--boot" in args else 2000
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 20260919
    metric = args[args.index("--metric") + 1] if "--metric" in args else "owner_share"   # e.g. relcomm: the social goal level
    names, common, est, pooled, own, mix, keys = two_coder({n: A.load(str(HERE / p)) for n, p in specs}, B, seed, metric)
    title = {"owner_share": "Other-benefit contrasts", "relcomm": "Social-goal (relational + community) contrasts"}.get(metric, f"{metric} contrasts")
    L = [f"# {title} under two coders", "",
         f"Coders: {', '.join(f'`{n}`' for n in names)}; {len(common)} lists coded by all; {B} bootstrap draws, seed {seed}. "
         "Each coder's interval is analyze.py's bootstrap over lists within setups. The pooled estimate is the mean over coders; "
         "its interval uses the same resamples and draws one coder at random per resample, so it carries both sampling and coder "
         "variability. **Final** applies the agreed rule: a reading stands only if it holds under every coder and under the pooled "
         "interval; otherwise inconclusive.", ""]
    hdr = "| arm | model | contrast | " + " | ".join(f"{n}" for n in names) + " | pooled | final |"
    L += [hdr, "|" + " --- |" * (5 + len(names))]
    order = [(arm, m, k) for arm in ("imp", "exp", "rq2") for m in ("both", "gpt", "ds") for k in A.KINDS]
    for k in order:
        if k not in keys: continue
        cells, readings = [], []
        for n in names:
            c = ci(own[n][k]); r = A.reading(est[n][k], c) if k[0] != "rq2" else A.reading(est[n][k], c)
            readings.append(r); cells.append(f"{A.pts(est[n][k])} {A.ci_s(c)} {r}")
        cp = ci(mix[k]); rp = A.reading(pooled[k], cp); readings.append(rp)
        final = readings[0] if len(set(readings)) == 1 else "inconclusive"
        arm = {"imp": "implicit", "exp": "explicit", "rq2": "explicit − implicit"}[k[0]]
        L.append(f"| {arm} | {k[1]} | {k[2]} | " + " | ".join(cells) + f" | {A.pts(pooled[k])} {A.ci_s(cp)} {rp} | **{final}** |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    prim = [(k, pooled[k]) for k in order if k[0] == "imp" and k[1] == "both" and k in keys]
    print(f"wrote {out}; implicit, both models, pooled: " + ", ".join(f"{k[2]} {A.pts(v)}" for k, v in prim))


if __name__ == "__main__":
    main(sys.argv[1:])
