# -*- coding: utf-8 -*-
"""Codebook v4 coder: GPT-6.1 Sol at medium reasoning effort, through the OpenAI API (official Python SDK).

    python llmcode_openai.py show   known                  # print the request for the first list; no API call
    python llmcode_openai.py submit all [--yes]            # one OpenAI Batch (50% price, 24 h window) -> llm/sol61-all/batch.json
    python llmcode_openai.py poll   all                    # status; once completed, stores raw responses and writes the CSV
    python llmcode_openai.py live   known [--limit N] [--yes]   # live calls, resumable; stops when the quota runs out
    python llmcode_openai.py parse  all                    # rebuild the CSV from stored raw responses; no API call

Same prompt, JSON mode and parse rules as llmcode4.py; only the model changes. The instructions and codebook come
first in every prompt, so OpenAI's automatic prefix caching applies. Raw responses are stored unchanged and never
resampled. Key: OPENAI_API_KEY in the workspace-root .env.
"""
import io, json, sys, threading
sys.stdout.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor

import openai
import llmcode as L1
import llmcode4 as L4
import coderun as R

CODER, MODEL, EFFORT, MAX_TOKENS, WORKERS = "sol61", "gpt-6.1-sol", "medium", 16384, 6
SETTINGS = dict(model=MODEL, effort=EFFORT, max_tokens=MAX_TOKENS, provider="openai", response_format="json_object")


def client():
    return openai.OpenAI(api_key=L1.env()["OPENAI_API_KEY"], max_retries=4, timeout=600)


def body(block, cb):
    return dict(model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_TOKENS, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": L4.prompt_for(block, cb)}])


def setup(which):
    blocks = L4.load_blocks(which); cb = L4.codebook_text()
    return blocks, cb, R.check_config(CODER, which, blocks, cb, SETTINGS)


def store(d, bid, raw, extra):
    (d / "raw" / f"{bid}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    ch = (raw.get("choices") or [{}])[0]
    R.log(d, dict(block_id=bid, status="ok", model_returned=raw.get("model"), finish_reason=ch.get("finish_reason"),
                  usage=raw.get("usage"), timestamp=R.now(), **extra))


def submit(which, yes=False):
    blocks, cb, d = setup(which)
    if (d / "batch.json").exists():
        sys.exit(f"{d.relative_to(R.HERE)}/batch.json exists: poll it, or move it aside to resubmit")
    todo = [b for b in blocks if not (d / "raw" / f"{b['block_id']}.json").exists()]
    print(f"{CODER} {which}: {len(todo)} requests to {MODEL} (reasoning {EFFORT})")
    if not todo: R.write_csv(CODER, which, blocks, d); return
    if not yes and input("submit the batch? [y/N] ").strip().lower() != "y": sys.exit("aborted")
    lines = "\n".join(json.dumps({"custom_id": b["block_id"], "method": "POST", "url": "/v1/chat/completions", "body": body(b, cb)},
                                 ensure_ascii=False) for b in todo)
    (d / "batch_input.jsonl").write_text(lines, encoding="utf-8")
    c = client()
    f = c.files.create(file=("batch_input.jsonl", io.BytesIO(lines.encode("utf-8"))), purpose="batch")
    b = c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
    (d / "batch.json").write_text(json.dumps(dict(id=b.id, input_file_id=f.id, status=b.status, submitted_at=R.now(), n=len(todo)), indent=1), encoding="utf-8")
    print(f"submitted batch {b.id} ({b.status}); run `python llmcode_openai.py poll {which}`")


def poll(which):
    blocks, cb, d = setup(which)
    rec = json.loads((d / "batch.json").read_text(encoding="utf-8"))
    if rec.get("collected_at"):
        print(f"{CODER} {which}: batch {rec['id']} already collected ({rec['collected_at']}); `parse` rebuilds the CSV"); return
    c = client()
    b = c.batches.retrieve(rec["id"])
    print(f"{CODER} {which}: batch {b.id} {b.status}; {b.request_counts}")
    if b.status not in ("completed", "failed", "expired", "cancelled"): return
    ok = bad = 0
    for fid, kind in ((b.output_file_id, "output"), (b.error_file_id, "error")):
        if not fid: continue
        text = c.files.content(fid).text
        (d / f"batch_{kind}.jsonl").write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if not line.strip(): continue
            res = json.loads(line); bid = res.get("custom_id"); r = res.get("response") or {}
            if (d / "raw" / f"{bid}.json").exists():
                R.log(d, dict(block_id=bid, status="skipped_existing", batch_id=b.id, timestamp=R.now())); continue
            if r.get("status_code") == 200 and r.get("body"):
                store(d, bid, r["body"], dict(batch_id=b.id)); ok += 1
            else:
                R.log(d, dict(block_id=bid, status="error", batch_id=b.id, timestamp=R.now(),
                              error=json.dumps(res.get("error") or r, ensure_ascii=False)[:400])); bad += 1
    rec.update(status=b.status, collected_at=R.now(), ok=ok, errors=bad)
    (d / "batch.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"collected {ok} responses, {bad} errors")
    R.write_csv(CODER, which, blocks, d)


def live(which, limit=None, yes=False):
    blocks, cb, d = setup(which)
    todo = [b for b in blocks if not (d / "raw" / f"{b['block_id']}.json").exists()][:limit or None]
    print(f"{CODER} {which}: {len(blocks)} lists, {len(todo)} to code live with {MODEL} (reasoning {EFFORT})")
    if todo and not yes and input("call the API? [y/N] ").strip().lower() != "y": sys.exit("aborted")
    c = client(); stop = threading.Event(); lock = threading.Lock(); first = [True]; usage = []

    def one(b):
        if stop.is_set(): return
        try:
            raw = c.chat.completions.with_raw_response.create(**body(b, cb)).http_response.json()
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e):
                stop.set(); print("out of quota; stopping. Top up and rerun the same command to resume."); return
            R.log(d, dict(block_id=b["block_id"], status="error", error=str(e)[:300], timestamp=R.now())); return
        except openai.APIError as e:
            R.log(d, dict(block_id=b["block_id"], status="error", error=str(e)[:300], timestamp=R.now())); return
        with lock:
            store(d, b["block_id"], raw, {})
            usage.append(raw.get("usage") or {})
            if first[0]:
                first[0] = False
                print(f"--- first: model {raw.get('model')!r}, usage {raw.get('usage')}")
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    if usage:
        print(f"done: {len(usage)} ok; prompt {sum(u.get('prompt_tokens', 0) for u in usage)}, "
              f"completion {sum(u.get('completion_tokens', 0) for u in usage)}")
    R.write_csv(CODER, which, blocks, d)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2 or a[0] not in ("show", "submit", "poll", "live", "parse") or a[1] not in L4.SETS:
        sys.exit(__doc__)
    cmd, which = a[0], a[1]
    if cmd == "show":
        bd = body(L4.load_blocks(which)[0], L4.codebook_text())
        print(json.dumps({k: v for k, v in bd.items() if k != "messages"}, indent=1)); print(bd["messages"][0]["content"][-400:])
    elif cmd == "submit": submit(which, yes="--yes" in a)
    elif cmd == "poll": poll(which)
    elif cmd == "live": live(which, limit=int(a[a.index("--limit") + 1]) if "--limit" in a else None, yes="--yes" in a)
    elif cmd == "parse":
        blocks, cb, d = setup(which); R.write_csv(CODER, which, blocks, d)
