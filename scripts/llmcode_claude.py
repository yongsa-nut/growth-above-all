# -*- coding: utf-8 -*-
"""Codebook v4 coder: Claude Opus 5.5 at medium effort, through the Anthropic API (official Python SDK).

    python llmcode_claude.py show   known                  # print the request for the first list; no API call
    python llmcode_claude.py submit all [--yes]            # one Message Batch (50% price) -> llm/opus55-all/batch.json
    python llmcode_claude.py poll   all                    # status; once ended, stores raw responses and writes the CSV
    python llmcode_claude.py live   known [--limit N] [--yes]   # live calls, resumable; stops when the credit runs out
    python llmcode_claude.py parse  all                    # rebuild the CSV from stored raw responses; no API call

Same prompt text as llmcode4.py, sent as one user turn in two blocks: the instructions and codebook (identical in
every request, cached for an hour) and the list. Thinking stays at the model default (adaptive); effort is set to
medium explicitly. No refusal fallback: the Batches API rejects it, and a second model would change the coder; a
refusal is stored, counted and left blank. Raw responses are stored unchanged and never resampled.
Key: ANTHROPIC_API_KEY in the workspace-root .env.
"""
import json, sys, threading
sys.stdout.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor

import anthropic
import llmcode as L1
import llmcode4 as L4
import coderun as R

CODER, MODEL, EFFORT, MAX_TOKENS, WORKERS = "opus55", "claude-opus-5-5", "medium", 16000, 6
SETTINGS = dict(model=MODEL, effort=EFFORT, max_tokens=MAX_TOKENS, provider="anthropic", cache="prefix block, ttl 1h")


def client():
    return anthropic.Anthropic(api_key=L1.env()["ANTHROPIC_API_KEY"], max_retries=4, timeout=600)


def params(block, cb):
    pre, suf = R.prompt_parts(block, cb)
    return dict(model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": pre, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"type": "text", "text": suf}]}])


def setup(which):
    blocks = L4.load_blocks(which); cb = L4.codebook_text()
    return blocks, cb, R.check_config(CODER, which, blocks, cb, SETTINGS)


def store(d, bid, msg, extra):
    raw = msg if isinstance(msg, dict) else msg.to_dict()
    (d / "raw" / f"{bid}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    R.log(d, dict(block_id=bid, status="ok", model_returned=raw.get("model"), stop_reason=raw.get("stop_reason"),
                  usage=raw.get("usage"), timestamp=R.now(), **extra))
    if raw.get("stop_reason") == "refusal":
        print(f"refusal: {bid} {raw.get('stop_details')}")


def submit(which, yes=False):
    blocks, cb, d = setup(which)
    if (d / "batch.json").exists():
        sys.exit(f"{d.relative_to(R.HERE)}/batch.json exists: poll it, or move it aside to resubmit")
    todo = [b for b in blocks if not (d / "raw" / f"{b['block_id']}.json").exists()]
    print(f"{CODER} {which}: {len(todo)} requests to {MODEL} (effort {EFFORT})")
    if not todo: R.write_csv(CODER, which, blocks, d); return
    if not yes and input("submit the batch? [y/N] ").strip().lower() != "y": sys.exit("aborted")
    batch = client().messages.batches.create(requests=[{"custom_id": b["block_id"], "params": params(b, cb)} for b in todo])
    (d / "batch.json").write_text(json.dumps(dict(id=batch.id, status=batch.processing_status, submitted_at=R.now(), n=len(todo)), indent=1), encoding="utf-8")
    print(f"submitted batch {batch.id} ({batch.processing_status}); run `python llmcode_claude.py poll {which}`")


def poll(which):
    blocks, cb, d = setup(which)
    rec = json.loads((d / "batch.json").read_text(encoding="utf-8"))
    if rec.get("collected_at"):
        print(f"{CODER} {which}: batch {rec['id']} already collected ({rec['collected_at']}); `parse` rebuilds the CSV"); return
    c = client()
    batch = c.messages.batches.retrieve(rec["id"])
    print(f"{CODER} {which}: batch {batch.id} {batch.processing_status}; {batch.request_counts}")
    if batch.processing_status != "ended": return
    ok = bad = 0
    for res in c.messages.batches.results(batch.id):
        bid = res.custom_id
        if (d / "raw" / f"{bid}.json").exists():
            R.log(d, dict(block_id=bid, status="skipped_existing", batch_id=batch.id, timestamp=R.now())); continue
        if res.result.type == "succeeded":
            store(d, bid, res.result.message, dict(batch_id=batch.id)); ok += 1
        else:
            R.log(d, dict(block_id=bid, status=res.result.type, batch_id=batch.id, timestamp=R.now(),
                          error=json.dumps(res.result.to_dict(), ensure_ascii=False)[:400])); bad += 1
    rec.update(status="ended", collected_at=R.now(), ok=ok, not_ok=bad)
    (d / "batch.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"collected {ok} responses, {bad} not succeeded")
    R.write_csv(CODER, which, blocks, d)


def live(which, limit=None, yes=False):
    blocks, cb, d = setup(which)
    todo = [b for b in blocks if not (d / "raw" / f"{b['block_id']}.json").exists()][:limit or None]
    print(f"{CODER} {which}: {len(blocks)} lists, {len(todo)} to code live with {MODEL} (effort {EFFORT})")
    if todo and not yes and input("call the API? [y/N] ").strip().lower() != "y": sys.exit("aborted")
    c = client(); stop = threading.Event(); lock = threading.Lock(); first = [True]; usage = []

    def one(b):
        if stop.is_set(): return
        try:
            msg = c.messages.create(**params(b, cb))
        except anthropic.BadRequestError as e:
            if "credit balance" in str(e).lower():
                stop.set(); print("out of credit; stopping. Top up and rerun the same command to resume."); return
            R.log(d, dict(block_id=b["block_id"], status="error", error=str(e)[:300], timestamp=R.now())); return
        except anthropic.APIStatusError as e:
            R.log(d, dict(block_id=b["block_id"], status="error", error=f"{e.status_code}: {str(e)[:300]}", timestamp=R.now())); return
        except anthropic.APIConnectionError as e:
            R.log(d, dict(block_id=b["block_id"], status="error", error=f"connection: {e}", timestamp=R.now())); return
        with lock:
            store(d, b["block_id"], msg, {})
            usage.append(msg.usage)
            if first[0]:
                first[0] = False
                print(f"--- first: model {msg.model!r}, stop {msg.stop_reason!r}, usage {msg.usage}")
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    if usage:
        tot = lambda f: sum(getattr(u, f, 0) or 0 for u in usage)
        print(f"done: {len(usage)} ok; input {tot('input_tokens')}, cache write {tot('cache_creation_input_tokens')}, "
              f"cache read {tot('cache_read_input_tokens')}, output {tot('output_tokens')}")
    R.write_csv(CODER, which, blocks, d)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2 or a[0] not in ("show", "submit", "poll", "live", "parse") or a[1] not in L4.SETS:
        sys.exit(__doc__)
    cmd, which = a[0], a[1]
    if cmd == "show":
        p = params(L4.load_blocks(which)[0], L4.codebook_text())
        print(json.dumps({k: v for k, v in p.items() if k != "messages"}, indent=1))
        print("block 1 (cached):", len(p["messages"][0]["content"][0]["text"]), "chars; block 2:", p["messages"][0]["content"][1]["text"][:400])
    elif cmd == "submit": submit(which, yes="--yes" in a)
    elif cmd == "poll": poll(which)
    elif cmd == "live": live(which, limit=int(a[a.index("--limit") + 1]) if "--limit" in a else None, yes="--yes" in a)
    elif cmd == "parse":
        blocks, cb, d = setup(which); R.write_csv(CODER, which, blocks, d)
