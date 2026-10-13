# -*- coding: utf-8 -*-
"""Shared pieces of the v4 coder runs (llmcode_openai.py, llmcode_claude.py): run directories with a config guard,
raw-response storage, parsing into the llmcode4.py CSV layout, and the cost estimate.

    python coderun.py estimate        # requests, tokens and cost for both coders; no API call
"""
import datetime, json, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")

import llmcode as L1
import llmcode4 as L4

HERE = pathlib.Path(__file__).resolve().parent
SPLIT = "\n\n# List\n"          # the prompt splits here into the shared prefix (instructions + codebook) and the list


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def prompt_parts(block, cb):
    p = L4.prompt_for(block, cb); i = p.index(SPLIT)
    return p[:i], p[i:]          # prefix + suffix == the exact prompt llmcode4.py sends


def rundir(coder, which):
    d = HERE / "llm" / f"{coder}-{which}"; (d / "raw").mkdir(parents=True, exist_ok=True); return d


def check_config(coder, which, blocks, cb, settings):
    d = rundir(coder, which)
    cfg = dict(coder=coder, set=which, **settings, codebook=str(L4.CODEBOOK.relative_to(HERE)), codebook_sha256=L1.sha(cb),
               blocks_sha256=L1.sha(json.dumps(blocks, ensure_ascii=False, sort_keys=True)),
               prompt_sha256=L1.sha(L4.prompt_for(blocks[0], cb)))
    p = d / "config.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old != cfg:
            sys.exit(f"{p} differs ({[k for k in cfg if old.get(k) != cfg[k]]}); move the old run aside first")
    else:
        p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return d


def coerce(text):
    """Keep the outermost {...}: a model may wrap the JSON object in a fence or a sentence."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lstrip().lower().startswith("json"): s = s.lstrip()[4:]
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if i >= 0 and j > i else s


def answer_text(raw):
    """The model's answer from a stored response: OpenAI chat completion or Anthropic message."""
    if "choices" in raw:
        return raw["choices"][0]["message"].get("content") or ""
    return "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")


def write_csv(coder, which, blocks, d):
    import csv
    ts = now(); rows = []; missing = perr = 0
    for b in blocks:
        p = d / "raw" / f"{b['block_id']}.json"
        if not p.exists():
            missing += 1; continue
        shim = {"choices": [{"message": {"content": coerce(answer_text(json.loads(p.read_text(encoding="utf-8"))))}}]}
        items = L4.parse_block(b, shim)
        perr += any(it["note"].startswith("parse_error") for it in items)
        for i, (text, it) in enumerate(zip(b["items"], items), 1):
            rows.append(dict(coder=coder, block_id=b["block_id"], schema_id=b["schema_id"], cell=b["cell"], arm=b["arm"],
                             pair=b["pair"], model=b["model"], core="1" if b["core"] else "", item_idx=i, item_text=text,
                             **it, exported_at=ts))
    out = d / f"codes_{coder}_{which}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=L4.COLS); w.writeheader(); w.writerows(rows)
    print(f"{out.relative_to(HERE)}: {len(rows)} items from {len(blocks) - missing} lists ({missing} missing, {perr} unparsed)")


def log(d, line):
    with open(d / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def estimate():
    pin, pout = [], []
    for m in (HERE / "llm").glob("v4*-*/manifest.jsonl"):
        for line in m.read_text(encoding="utf-8").splitlines():
            u = json.loads(line).get("usage") or {}
            if u: pin.append(u.get("prompt_tokens", 0)); pout.append(u.get("completion_tokens", 0))
    tin, tout = sum(pin) / len(pin), sum(pout) / len(pout)
    cb = L4.codebook_text(); b0 = L4.load_blocks("all")[0]; pre, suf = prompt_parts(b0, cb)
    share = len(pre) / (len(pre) + len(suf))            # character share of the shared prefix, a proxy for its token share
    n = len(L4.load_blocks("known")) + len(L4.load_blocks("all"))
    print(f"{n} lists per coder (known 24 + all 864). Measured with gpt-5.6-sol: {tin:.0f} prompt and {tout:.0f} completion tokens per list;"
          f" the shared prefix is about {100 * share:.0f}% of the prompt.")
    print("Output length of the new coders is unknown until the known set runs: shown at 1x and 3x.\n")
    # GPT-6.1 Sol: $2 / $10 per M (OpenAI cached input assumed at full price here, so this is an upper bound); batch half.
    for lab, mult in (("1x output", 1), ("3x output", 3)):
        sol_live = n * (tin * 2.0 + mult * tout * 10.0) / 1e6
        # Claude Opus 5.5: $4 in, $20 out, cache read $0.20, 1-hour cache write 2x input; batch halves everything.
        pre_t, suf_t = tin * share, tin * (1 - share)
        opus_live = (pre_t * 8.0 + (n - 1) * pre_t * 0.20 + n * suf_t * 4.0 + n * mult * tout * 20.0) / 1e6
        print(f"{lab}: GPT-6.1 Sol live ${sol_live:.2f}, batch ${sol_live / 2:.2f}; Claude Opus 5.5 live ${opus_live:.2f}, batch ${opus_live / 2:.2f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["estimate"]:
        estimate()
    else:
        sys.exit(__doc__)
