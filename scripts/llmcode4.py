# -*- coding: utf-8 -*-
"""LLM coder on codebook v4: gpt-5.6-sol at medium reasoning effort, one list per call.

    python llmcode4.py known|pilot|heldout|all --yes   # calls -> llm/v4-<set>/raw/ + manifest.jsonl + config.json,
                                                       # then llm/v4-<set>/codes_sol4_<set>.csv
    python llmcode4.py known --show                    # print the prompt for the first list; no API call
    python llmcode4.py known --parse                   # rebuild the CSV from stored raw responses; no API call

Sets: known = the 72 known-answer items (known/known_items.json), 12 lists x Thai and English = 24 lists, each
attached to the Thai-setting situation of its schema; pilot = the 32 development lists; heldout = the 36 held-out
lists (gen/run/blocks.heldout.json); all = the 864 lists.

Same discipline as llmcode.py: the coder sees the codebook, the situation and the items only; raw responses are
stored unchanged and never resampled; an answer that does not parse gets blank codes and 'parse_error'; resumable,
and refuses to resume if the codebook, the lists or the settings changed. Key: OPENAI_API_KEY in the root .env.
"""
import csv, datetime, json, pathlib, sys, threading
sys.stdout.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor

import llmcode as L1

HERE = pathlib.Path(__file__).resolve().parent
CODEBOOK = HERE / "reviews-camera-ready" / "codebook-v4-draft.md"
MODEL, EFFORT, CODER = "gpt-5.6-sol", "medium", "sol4"
WORKERS, MAX_TOKENS, ATTEMPTS = 6, 8192, 3
SETS = {"known": "known/known_items.json", "pilot": "gen/run/blocks.pilot.json",
        "heldout": "gen/run/blocks.heldout.json", "all": "gen/run/blocks.json"}
ALLOWED = {
    "item_type": {"lining", "consolation", "action", "none"},
    "consolation_type": {"acceptance", "impermanence", "could_be_worse", "reassurance", "fate_meaning", "other", ""},
    "goal": {"transcendent", "community", "relational", "growth", "achievement", "wellbeing", "resources", "unspecified", ""},
    "beneficiary": {"self", "other", "group", "transcendent", "unspecified", ""},
}
COLS = ["coder", "block_id", "schema_id", "cell", "arm", "pair", "model", "core", "item_idx", "item_text",
        "item_type", "consolation_type", "goal", "beneficiary", "note", "exported_at"]


def codebook_text():
    t = CODEBOOK.read_text(encoding="utf-8")
    t = t.split("\n", 1)[1]                                  # drop the title line
    return t.split("\n## Coarse goal level")[0].rstrip("-\n ").strip()


def known_blocks():
    K = json.loads((HERE / SETS["known"]).read_text(encoding="utf-8"))
    stim = {r["schema_id"]: r for r in csv.DictReader(open(HERE / "stimuli.csv", encoding="utf-8-sig"))}
    out = []
    for Lst in K["lists"]:
        for lang, cell, col in (("th", "th_th", "text_th_th"), ("en", "th_en", "text_th_en")):
            out.append(dict(block_id=f"{Lst['list']}-{lang}", schema_id=Lst["schema"], cell=cell, arm="known", pair="",
                            model="", core=False, situation=stim[Lst["schema"]][col], items=[it[lang] for it in Lst["items"]]))
    return out


def load_blocks(which):
    if which == "known":
        return known_blocks()
    return json.loads((HERE / SETS[which]).read_text(encoding="utf-8"))["blocks"]


def prompt_for(block, cb):
    items = "\n".join(f"{i}. {it}" for i, it in enumerate(block["items"], 1))
    return f"""You are a coder in a content-analysis study. Apply the codebook below to one list exactly as a trained human coder would, using only the situation and the items shown.

# Codebook
{cb}

# List
Situation:
{block["situation"]}

Items:
{items}

# Output
Return only a JSON object, no prose, in this shape:
{{"items": [{{"n": 1, "item_type": "lining | consolation | action | none", "consolation_type": "acceptance | impermanence | could_be_worse | reassurance | fate_meaning | other", "goal": "transcendent | community | relational | growth | achievement | wellbeing | resources | unspecified", "beneficiary": "self | other | group | transcendent | unspecified", "note": ""}}]}}
One entry per item, in order ("none" as item_type means not a lining). A field the codebook does not ask for that item type is an empty string. note is optional and short."""


def run(which, show=False, parse_only=False, yes=False):
    blocks = load_blocks(which)
    cb = codebook_text()
    if show:
        print(prompt_for(blocks[0], cb)); return
    d = HERE / "llm" / f"v4-{which}"; raw = d / "raw"; raw.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_TOKENS, set=which, source=SETS[which],
               codebook_sha256=L1.sha(cb), blocks_sha256=L1.sha(json.dumps(blocks, ensure_ascii=False, sort_keys=True)),
               prompt_sha256=L1.sha(prompt_for(blocks[0], cb)))
    cfgp = d / "config.json"
    if cfgp.exists():
        old = json.loads(cfgp.read_text(encoding="utf-8"))
        if old != cfg:
            sys.exit(f"{cfgp} differs ({[k for k in cfg if old.get(k) != cfg[k]]}); not resuming. Move the old run aside first.")
    else:
        cfgp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    if not parse_only:
        todo = [b for b in blocks if not (raw / f"{b['block_id']}.json").exists()]
        print(f"{which}: {len(blocks)} lists, {len(blocks) - len(todo)} done, {len(todo)} to code with {MODEL} ({EFFORT})")
        if todo:
            if not yes and input("call the API? [y/N] ").strip().lower() != "y":
                sys.exit("aborted")
            call_all(todo, cb, d, raw)
    write_csv(which, blocks, d, raw)


def call_all(todo, cb, d, raw):
    import openai
    client = openai.OpenAI(api_key=L1.env()["OPENAI_API_KEY"], timeout=300, max_retries=0)
    retryable = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
    lock = threading.Lock(); first = [True]; errors = []; usage = []

    def one(b):
        prompt = prompt_for(b, cb)
        line = dict(block_id=b["block_id"], model_requested=MODEL, reasoning_effort=EFFORT, prompt_sha256=L1.sha(prompt),
                    attempts=0, errors=[])
        body = None
        for _ in range(ATTEMPTS):
            line["attempts"] += 1
            try:
                r = client.chat.completions.with_raw_response.create(
                    model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
                body = r.http_response.json(); break
            except retryable as ex:
                line["errors"].append(f"{type(ex).__name__}: {ex}")
            except Exception as ex:
                line["errors"].append(f"{type(ex).__name__}: {ex}"); break
        line["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        if body is None:
            line["status"] = "error"
        else:
            ch = body.get("choices", [{}])[0]
            line.update(status="ok", model_returned=body.get("model"), finish_reason=ch.get("finish_reason"), usage=body.get("usage"))
            (raw / f"{b['block_id']}.json").write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
        with lock:
            with open(d / "manifest.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if line["status"] == "error":
                errors.append(b["block_id"]); print("error:", b["block_id"], line["errors"][-1][:200])
            else:
                usage.append(line["usage"] or {})
                if first[0]:
                    first[0] = False
                    print(f"--- first: model {line['model_returned']!r}, finish {line['finish_reason']!r}, usage {line['usage']}")
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    pin = sum(u.get("prompt_tokens", 0) for u in usage); pout = sum(u.get("completion_tokens", 0) for u in usage)
    print(f"done: {len(todo) - len(errors)} ok, {len(errors)} error {errors if errors else ''}; tokens {pin} in, {pout} out")


def parse_block(b, body):
    n = len(b["items"]); blank = [dict(item_type="", consolation_type="", goal="", beneficiary="", note="") for _ in range(n)]
    try:
        content = body["choices"][0]["message"]["content"]
        items = json.loads(content)["items"]; assert len(items) == n, f"{len(items)} items for {n}"
    except Exception as ex:
        for it in blank: it["note"] = f"parse_error: {str(ex)[:80]}"
        return blank
    out = []
    for it in items:
        row = {k: str(it.get(k, "") or "").strip() for k in ("item_type", "consolation_type", "goal", "beneficiary", "note")}
        for k in [k for k in ALLOWED if row[k] not in ALLOWED[k]]:
            row["note"] = (row["note"] + f" invalid {k}={row[k]!r}").strip(); row[k] = ""
        if row["item_type"] != "consolation": row["consolation_type"] = ""
        if row["item_type"] != "lining": row["goal"] = row["beneficiary"] = ""
        out.append(row)
    return out


def write_csv(which, blocks, d, raw):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"); rows = []; missing = perr = 0
    for b in blocks:
        p = raw / f"{b['block_id']}.json"
        if not p.exists():
            missing += 1; continue
        items = parse_block(b, json.loads(p.read_text(encoding="utf-8")))
        perr += any(it["note"].startswith("parse_error") for it in items)
        for i, (text, it) in enumerate(zip(b["items"], items), 1):
            rows.append(dict(coder=CODER, block_id=b["block_id"], schema_id=b["schema_id"], cell=b["cell"], arm=b["arm"],
                             pair=b["pair"], model=b["model"], core="1" if b["core"] else "", item_idx=i, item_text=text,
                             **it, exported_at=ts))
    out = d / f"codes_{CODER}_{which}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"{out}: {len(rows)} items from {len(blocks) - missing} lists ({missing} missing, {perr} unparsed)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in SETS:
        sys.exit(f"usage: python llmcode4.py {'|'.join(SETS)} [--yes|--show|--parse]")
    run(a[0], show="--show" in a, parse_only="--parse" in a, yes="--yes" in a)
