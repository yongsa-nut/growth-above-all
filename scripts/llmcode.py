# -*- coding: utf-8 -*-
"""LLM coder: gpt-5.6-sol at medium reasoning effort codes blocks with codebook.md, one block per call.

    python llmcode.py pilot|core|all          # calls -> llm/<set>/raw/<block_id>.json + manifest.jsonl + config.json,
                                              # then llm/<set>/codes_sol_<set>.csv in the coder.html export layout
    python llmcode.py pilot --show            # print the prompt for the first block; no API call
    python llmcode.py pilot --parse           # rebuild the CSV from the stored raw responses; no API call

The coder sees what a human coder sees: the codebook, the situation and the items (no cell, model or block id).
Raw responses are stored unchanged and never resampled: an answer that does not parse gets blank codes and the
note 'parse_error'. Resumable (blocks with a raw file are skipped); refuses to resume if codebook.md or the
blocks file changed. Asks before spending. Key: OPENAI_API_KEY in the workspace-root .env.
"""
import csv, datetime, hashlib, json, pathlib, sys, threading
sys.stdout.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
MODEL, EFFORT, CODER = "gpt-5.6-sol", "medium", "sol"
WORKERS, MAX_TOKENS, ATTEMPTS = 4, 4096, 3
SETS = {"pilot": "gen/run/blocks.pilot.json", "core": "gen/run/blocks.core.json", "all": "gen/run/blocks.json"}
ALLOWED = {
    "item_type": {"lining", "consolation", "advice", "none"},
    "consolation_type": {"acceptance", "impermanence", "could_be_worse", "reassurance", "fate_meaning", "other", ""},
    "grounding": {"supported", "unsupported", ""},
    "goal": {"growth", "achievement", "relational", "community", "transcendent", "wellbeing", "practical", "status", "unspecified", ""},
    "owner": {"self", "other", "group", "transcendent", "unspecified", ""},
}
FLAGS = {"wrong_language", "not_six", "refusal", "other"}
COLS = ["coder", "block_id", "schema_id", "cell", "arm", "pair", "model", "core", "block_order", "item_idx", "item_text",
        "item_type", "consolation_type", "grounding", "goal", "owner", "redundant_with", "note", "block_flags", "block_flag_note", "exported_at"]


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()


def codebook_text():
    t = (HERE / "codebook.md").read_text(encoding="utf-8")
    return t.split("\n## Computed, not coded")[0].strip()       # the coder-facing part only


def prompt_for(block, cb):
    items = "\n".join(f"{i}. {it}" for i, it in enumerate(block["items"], 1))
    return f"""You are a coder in a content-analysis study. Apply the codebook below to one block exactly as a trained human coder would, using only the situation and the items shown.

# Codebook
{cb}

# Block
Situation:
{block["situation"]}

Items:
{items}

# Output
Return only a JSON object, no prose, in this shape:
{{"items": [{{"n": 1, "item_type": "lining | consolation | advice | none", "consolation_type": "acceptance | impermanence | could_be_worse | reassurance | fate_meaning | other", "grounding": "supported | unsupported", "goal": "growth | achievement | relational | community | transcendent | wellbeing | practical | status | unspecified", "owner": "self | other | group | transcendent | unspecified", "redundant_with": "none or the earlier item's number", "note": ""}}], "block_flags": [], "block_flag_note": ""}}
One entry per item, in order ("none" as item_type means not a lining). A field the codebook does not ask for that item type is an empty string. block_flags may contain wrong_language, not_six, refusal, other; note is optional and short."""


def env():
    e = {}
    for line in (HERE.parent.parent / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return e


def send(client, prompt):
    r = client.chat.completions.with_raw_response.create(
        model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
    return r.http_response.json()


def run(which, show=False, parse_only=False):
    src = HERE / SETS[which]
    blocks = json.loads(src.read_text(encoding="utf-8"))["blocks"]
    cb = codebook_text()
    if show:
        print(prompt_for(blocks[0], cb)); return
    d = HERE / "llm" / which; raw = d / "raw"; raw.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_TOKENS, set=which, source=SETS[which],
               codebook_sha256=sha(cb), blocks_sha256=sha(src.read_text(encoding="utf-8")), prompt_sha256=sha(prompt_for(blocks[0], cb)))
    cfgp = d / "config.json"
    if cfgp.exists():
        old = json.loads(cfgp.read_text(encoding="utf-8"))
        if old != cfg:
            sys.exit(f"{cfgp} differs from the current codebook / blocks / settings ({[k for k in cfg if old.get(k) != cfg[k]]}); not resuming")
    else:
        cfgp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    if not parse_only:
        todo = [b for b in blocks if not (raw / f"{b['block_id']}.json").exists()]
        print(f"{which}: {len(blocks)} blocks, {len(blocks) - len(todo)} done, {len(todo)} to code with {MODEL} ({EFFORT})")
        if todo:
            if input("call the API? [y/N] ").strip().lower() != "y":
                sys.exit("aborted")
            call_all(todo, cb, d, raw)
    write_csv(which, blocks, d, raw)


def call_all(todo, cb, d, raw):
    import openai
    client = openai.OpenAI(api_key=env()["OPENAI_API_KEY"], timeout=180, max_retries=0)
    retryable = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
    lock = threading.Lock(); first = [True]; errors = []

    def one(b):
        prompt = prompt_for(b, cb); line = dict(block_id=b["block_id"], model_requested=MODEL, reasoning_effort=EFFORT,
                                                prompt_sha256=sha(prompt), attempts=0, errors=[])
        body = None
        for _ in range(ATTEMPTS):
            line["attempts"] += 1
            try:
                body = send(client, prompt); break
            except retryable as ex:
                line["errors"].append(f"{type(ex).__name__}: {ex}")
            except Exception as ex:                       # anything else: record and stop
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
                errors.append(b["block_id"])
            elif first[0]:
                first[0] = False
                print(f"--- first response: model {line['model_returned']!r}, finish {line['finish_reason']!r}, usage {line['usage']}")
                print("content:", ch.get("message", {}).get("content", "")[:300])
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    print(f"done: {len(todo) - len(errors)} ok, {len(errors)} error {errors if errors else ''}")


def parse_block(b, body):
    """The stored response -> one dict per item, in the export layout; blanks + a note when it does not parse."""
    n = len(b["items"]); blank = [dict(item_type="", consolation_type="", grounding="", goal="", owner="", redundant_with="", note="") for _ in range(n)]
    flags, flag_note = [], ""
    try:
        content = body["choices"][0]["message"]["content"]
        obj = json.loads(content)
        items = obj["items"]; assert len(items) == n, f"{len(items)} items for {n}"
    except Exception as ex:
        for it in blank: it["note"] = f"parse_error: {str(ex)[:80]}"
        return blank, flags, flag_note
    out = []
    for i, it in enumerate(items):
        row = {k: str(it.get(k, "") or "").strip() for k in ("item_type", "consolation_type", "grounding", "goal", "owner", "redundant_with", "note")}
        bad = [k for k in ALLOWED if row[k] not in ALLOWED[k]]
        for k in bad:
            row["note"] = (row["note"] + f" invalid {k}={row[k]!r}").strip(); row[k] = ""
        if row["item_type"] != "consolation": row["consolation_type"] = ""
        if row["item_type"] != "lining": row["grounding"] = row["goal"] = row["owner"] = row["redundant_with"] = ""
        elif row["redundant_with"] not in ["none"] + [str(k) for k in range(1, i + 1)]:
            row["note"] = (row["note"] + f" invalid redundant_with={row['redundant_with']!r}").strip(); row["redundant_with"] = ""
        out.append(row)
    flags = [f for f in (obj.get("block_flags") or []) if f in FLAGS]
    flag_note = str(obj.get("block_flag_note") or "")[:200]
    return out, flags, flag_note


def write_csv(which, blocks, d, raw):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"); rows = []; missing = 0; perr = 0
    for pos, b in enumerate(blocks, 1):
        p = raw / f"{b['block_id']}.json"
        if not p.exists():
            missing += 1; continue
        items, flags, flag_note = parse_block(b, json.loads(p.read_text(encoding="utf-8")))
        perr += any(it["note"].startswith("parse_error") for it in items)
        for i, (text, it) in enumerate(zip(b["items"], items), 1):
            rows.append(dict(coder=CODER, block_id=b["block_id"], schema_id=b["schema_id"], cell=b["cell"], arm=b["arm"], pair=b["pair"],
                             model=b["model"], core="1" if b["core"] else "", block_order=pos, item_idx=i, item_text=text, **it,
                             block_flags="|".join(flags), block_flag_note=flag_note, exported_at=ts))
    out = d / f"codes_{CODER}_{which}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"{out}: {len(rows)} items from {len(blocks) - missing} blocks ({missing} missing, {perr} unparsed)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in SETS:
        sys.exit("usage: python llmcode.py pilot|core|all [--show|--parse]")
    run(a[0], show="--show" in a, parse_only="--parse" in a)
