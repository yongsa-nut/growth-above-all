# -*- coding: utf-8 -*-
"""Generation for Silver Linings Exp 1. One file, four commands.

  python gen.py identify     # setting-identifiability: 72 texts x 2 models x 5 samples  -> gen/identify/
  python gen.py pilot        # S5 + S14, 4 cells x 3 pairs x 2 models, implicit arm     -> gen/pilot/
  python gen.py run          # all 864 blocks, both arms                                -> gen/run/
  python gen.py parse DIR    # no API. raw -> DIR/blocks.json (+ blocks.core.json) + DIR/parse_report.csv;
                             #         for gen/identify: DIR/identify_report.csv

Reads stimuli.csv and prompt_templates.json. Every API response is stored as received (the HTTP body, one file
per call) with one manifest line per call; parsing is a separate step; malformed responses are flagged, never
resampled. A run that stops can be started again: calls already ok in the manifest are skipped, and the run
refuses to resume or parse if stimuli.csv or prompt_templates.json changed since it started (start a new
directory instead). Keys come from the workspace-root .env (OPENAI_API_KEY, OPENROUTER_API_KEY).
"""
import csv, datetime, hashlib, json, pathlib, random, re, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Thai on a Windows console
SEED = 20260918
WORKERS = 4
CELLS = ["us_en", "us_th", "th_th", "th_en"]
MODELS = ["gpt", "ds"]
DOM = {"social": 0, "sport": 1, "academic": 2}
VAL = {"pos": 0, "neg": 1, "neu": 2}


# ---------------------------------------------------------------- inputs
def load():
    rows = list(csv.DictReader(open(HERE / "stimuli.csv", encoding="utf-8-sig")))
    tpl = json.load(open(HERE / "prompt_templates.json", encoding="utf-8"))
    return rows, tpl


def input_hashes():
    """Hashes of what shapes the prompts (texts, personas, wordings); notes, statuses and generation params are
    recorded elsewhere (config.json, manifest) and may change without invalidating a run."""
    rows, tpl = load()
    texts = [[r["schema_id"], r["domain"], r["valence"]] + [r["text_" + c] for c in CELLS] for r in rows]
    prompts = [tpl["template"], tpl["persona"]["en"], tpl["persona"]["th"], tpl["persona_explicit"]["en"],
               tpl["persona_explicit"]["th"], [[p["id"], p["en"], p["th"]] for p in tpl["instruction_pairs"]],
               tpl["identifiability_check"]["en"], tpl["identifiability_check"]["th"]]
    return {"stimuli_sha256": sha(json.dumps(texts, ensure_ascii=False)),
            "prompts_sha256": sha(json.dumps(prompts, ensure_ascii=False))}


def core_of(row):
    """Human core: all four cells of a schema share one model and one wording pair."""
    d, v = DOM[row["domain"]], VAL[row["valence"]]
    fam = 0 if int(row["schema_id"][1:]) <= 9 else 1
    m = (d + fam) % 2
    return MODELS[m], f"P{(d + v + m) % 3 + 1}"


def persona(tpl, arm, lang, setting):
    return tpl["persona"][lang] if arm == "imp" else tpl["persona_explicit"][lang][setting]


def jobs(mode, rows, tpl):
    out = []
    for row in rows:
        if mode == "pilot" and row["schema_id"] not in ("S5", "S14"):
            continue
        for cell in CELLS:
            setting, lang = cell.split("_")
            text = row["text_" + cell]
            for m in MODELS:
                if mode == "identify":
                    p = f"{tpl['persona'][lang]} {text} {tpl['identifiability_check'][lang]}"
                    for i in range(1, 6):
                        out.append(dict(id=f"{row['schema_id']}-{cell}-{m}-{i}", schema_id=row["schema_id"], cell=cell,
                                        arm="", pair="", sample=i, model=m, prompt=p, max_tokens=32))
                    continue
                for arm in (["imp"] if mode == "pilot" else ["imp", "exp"]):
                    for pr in tpl["instruction_pairs"]:
                        p = tpl["template"].format(persona=persona(tpl, arm, lang, setting), situation=text,
                                                   instruction=pr[lang])
                        out.append(dict(id=f"{row['schema_id']}-{cell}-{arm}-{m}-{pr['id']}", schema_id=row["schema_id"],
                                        cell=cell, arm=arm, pair=pr["id"], model=m, prompt=p,
                                        max_tokens=tpl["generation"]["common"]["max_tokens"]))
    random.Random(SEED).shuffle(out)  # interleave cells, models, pairs and arms
    return out


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- calls
def env():
    e = {}
    for line in (HERE.parent.parent / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return e


def clients():
    from openai import OpenAI
    e = env()
    return {"gpt": OpenAI(api_key=e["OPENAI_API_KEY"], timeout=120, max_retries=0),
            "ds": OpenAI(api_key=e["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1",
                         timeout=120, max_retries=0)}


def request(gen, m, max_tokens):
    """Request parameters as sent (recorded in the manifest)."""
    if m == "gpt":
        g = gen["gpt"]
        return dict(model=g["model"], temperature=g["temperature"], reasoning_effort=g["reasoning_effort"],
                    max_completion_tokens=max_tokens)
    g = gen["deepseek"]
    return dict(model=g["model"], temperature=g["temperature"], max_tokens=max_tokens,
                extra_body={"provider": g["provider"], "reasoning": g["reasoning"]})


def send(client, params, prompt):
    """One user turn, no system prompt, fresh context. Returns the HTTP body as received."""
    kw = dict(params)
    extra = kw.pop("extra_body", None)
    r = client.chat.completions.with_raw_response.create(messages=[{"role": "user", "content": prompt}],
                                                         extra_body=extra, **kw)
    return r.http_response.json()


def execute(mode, js, tpl):
    import openai
    retryable = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
    outdir = HERE / "gen" / mode
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cfg_path, mf, hashes = outdir / "config.json", outdir / "manifest.jsonl", input_hashes()
    if cfg_path.exists() and {k: json.loads(cfg_path.read_text(encoding="utf-8")).get(k) for k in hashes} != hashes:
        sys.exit(f"stimuli.csv or prompt_templates.json changed since gen/{mode} started: start a new directory")
    done = set()
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["status"] == "ok":
                done.add(d["block_id"])
    orphans = sorted(p.stem for p in raw.glob("*.json") if p.stem not in done)
    if orphans:
        sys.exit(f"{len(orphans)} raw files have no ok manifest line (interrupted mid-write?): {orphans[:5]} — "
                 f"move them out or add their lines before resuming")
    todo = [j for j in js if j["id"] not in done]
    print(f"{mode}: {len(js)} calls, {len(done)} already done, {len(todo)} to make ({dict(Counter(j['model'] for j in todo))})")
    if mode == "run":
        rep = HERE / "gen" / "identify" / "identify_report.csv"
        if not rep.exists():
            print("WARNING: no gen/identify/identify_report.csv — every text must pass the identifiability check first")
        else:
            fails = [r for r in csv.DictReader(open(rep, encoding="utf-8")) if r["pass"] != "True"]
            print(f"identifiability report: {len(fails)} text x model checks below 4/5")
    if not todo:
        return
    if input("call the APIs? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(dict(mode=mode, seed=SEED, workers=WORKERS, generation=tpl["generation"],
                                            started=now(), **hashes), ensure_ascii=False, indent=1), encoding="utf-8")
    cl, gen = clients(), tpl["generation"]
    lock, first = threading.Lock(), set()

    def one(j):
        params = request(gen, j["model"], j["max_tokens"])
        body, errors = None, []
        for attempt in range(1, 4):  # transport retries only; the content is never inspected
            try:
                body = send(cl[j["model"]], params, j["prompt"])
                break
            except retryable as ex:
                errors.append(f"{attempt}: {type(ex).__name__}: {ex}"[:300])
                time.sleep(3 * attempt)
            except Exception as ex:  # not retried
                errors.append(f"{attempt}: {type(ex).__name__}: {ex}"[:300])
                break
        line = dict(block_id=j["id"], schema_id=j["schema_id"], cell=j["cell"], arm=j["arm"], pair=j["pair"],
                    model_short=j["model"], mode=mode, status="ok" if body is not None else "error",
                    model_requested=params["model"], model_returned=None, provider=None, params=params,
                    finish_reason=None, usage=None, raw_path=None, attempts=attempt, errors=errors,
                    prompt_sha256=sha(j["prompt"]), timestamp=now())
        if j.get("sample"):
            line["sample"] = j["sample"]
        if body is not None:
            path = raw / f"{j['id']}.json"
            ch = (body.get("choices") or [{}])[0]
            line.update(model_returned=body.get("model"),
                        provider=body.get("provider") or ("openai" if j["model"] == "gpt" else None),
                        finish_reason=ch.get("finish_reason"), usage=body.get("usage"),
                        raw_path=path.relative_to(HERE).as_posix())
        with lock:
            if body is not None:
                path.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            with open(mf, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if body is not None and j["model"] not in first:  # first call per model: check by hand
                first.add(j["model"])
                msg = ch.get("message") or {}
                print(f"\n--- first {j['model']}: model {line['model_returned']!r}, provider {line['provider']!r}, "
                      f"finish {line['finish_reason']!r}, usage {line['usage']}")
                print("message fields:", {k: (v[:300] if k == "content" else v) for k, v in msg.items()
                                          if v not in (None, "", [], {})}, "\n")
        return line["status"]

    with ThreadPoolExecutor(WORKERS) as ex:
        st = Counter(ex.map(one, todo))
    print(f"{mode} finished: {st['ok']} ok, {st['error']} error -> gen/{mode}")
    sys.exit(0 if st["error"] == 0 else 1)


# ---------------------------------------------------------------- parse
MARK = re.compile(r"^(\s*)(?:(\d{1,2})[.)]|[-*•–])\s+(.*)$")
HEAD = re.compile(r"^(\s*)(?:\*\*)?\s*(?:มุม(?:มอง)?ที่|ข้อที่|ข้อ|มุม|angle|perspective|point|way|side|reason|thing)\s*(\d{1,2})\s*(?:[:.)—–-]|\*\*)*\s*(.*)$", re.I)
RULE = re.compile(r"^\s*[-*_]{3,}\s*$")


def marker(l):
    """(indent, numbered, text) for a list-marker line — '1. x', '- x', or a headed item like 'มุมที่ 1: x' / 'Angle 1 — x'."""
    if RULE.match(l):
        return None
    m = MARK.match(l)
    if m:
        return len(m.group(1)), bool(m.group(2)), m.group(3)
    m = HEAD.match(l)
    if m:
        return len(m.group(1)), True, m.group(3).strip("* ")
    return None


def norm(line):
    """**1.** x -> 1. x ; 1.**x** -> 1. x ; drop markdown headers."""
    line = re.sub(r"^\s*#+\s*", "", line.expandtabs(4)).rstrip()
    line = re.sub(r"^(\s*)(?:\*\*|__)\s*(\d{1,2}[.)])\s*(?:\*\*|__)?\s*", r"\1\2 ", line)
    return re.sub(r"^(\s*\d{1,2}[.)])(?=\*\*|__)", r"\1 ", line)  # 1.**x** -> 1. **x** (bold header kept for clean)


def clean(parts):
    """Item lines -> one string. A leading bold header is separated from its body by ' — ' unless it ends a sentence."""
    s = "\n".join(p.strip() for p in parts if p.strip())
    m = re.match(r"^\*\*(.+?)\*\*\s*[:：—–-]?\s*(.*)$", s, re.S)
    if m and m.group(2).strip():
        head, body = m.group(1).strip(), m.group(2).strip()
        s = head + (" " if re.search(r"[.!?:。]$", head) else " — ") + body
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"(?<!\w)[*_](?=\S)([^*_\n]+?)(?<=\S)[*_](?!\w)", r"\1", s)  # *this* -> this
    return s.strip(" :-–")


def parse_items(text):
    """Top-level list markers -> items (never demoted, blank lines included); text before the first -> preamble;
    after the last, an unindented paragraph past a blank line -> postamble unless earlier items were laid out that way."""
    lines = [norm(l) for l in text.splitlines()]
    marks = {i: mk for i, l in enumerate(lines) if (mk := marker(l))}
    if not marks:
        return [], " ".join(l.strip() for l in lines if l.strip()), ""
    base = min(ind for ind, _, _ in marks.values())
    numbered = any(num for ind, num, _ in marks.values() if ind <= base + 1)
    top = [i for i, (ind, num, _) in marks.items() if ind <= base + 1 and (num or not numbered)]  # in a numbered list, bullets are not items
    first, last = top[0], top[-1]
    pre = [l.strip() for l in lines[:first] if l.strip() and not RULE.match(l)]
    items, post, blank, blank_cont = [], [], False, False
    for i in range(first, len(lines)):
        l = lines[i]
        if i in top:
            items.append([marks[i][2]])
            blank = False
        elif not l.strip():
            blank = True
        elif RULE.match(l):
            continue
        elif post or (i > last and blank and len(l) - len(l.lstrip()) <= 1 and not blank_cont):
            post.append(l.strip())
        else:
            if blank and i < last and len(l) - len(l.lstrip()) <= 1:
                blank_cont = True  # this response separates item bodies from their markers by blank lines
            items[-1].append(re.sub(r"^(?:[-*•–]\s+|>\s*)", "", l.strip()))  # a sub-bullet or quote line folds into its item
            blank = False
    return [clean(p) for p in items], " ".join(pre), " ".join(post)


def scripts(text):
    th = len(re.findall(r"[฀-๿]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    other = len(re.findall(r"[^\W\d_]", text)) - th - en
    return th, en, other


def run_dir(dirname):
    d = pathlib.Path(dirname)
    d = d if d.is_absolute() else HERE / d
    if not d.is_dir() or not (d / "config.json").exists():
        sys.exit(f"{dirname}: not a run directory (no config.json)")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    if {k: cfg.get(k) for k in input_hashes()} != input_hashes():
        sys.exit("stimuli.csv or prompt_templates.json changed since this run: restore them to parse it")
    lines = {}
    for l in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        ln = json.loads(l)
        lines[ln["block_id"]] = ln  # last line per call wins (a retried error keeps its history in `errors`)
    return d, cfg, list(lines.values())


def parse(dirname):
    d, cfg, lines = run_dir(dirname)
    rows, tpl = load()
    R = {r["schema_id"]: r for r in rows}
    expected = {j["id"]: j for j in jobs(cfg["mode"], rows, tpl)}
    missing = sorted(set(expected) - {ln["block_id"] for ln in lines})
    if missing:
        print(f"WARNING: {len(missing)} calls never made: {missing[:8]}")
    for ln in lines:  # provenance: the prompt rebuilt from today's inputs must be the prompt that was sent
        assert ln["block_id"] in expected and sha(expected[ln["block_id"]]["prompt"]) == ln["prompt_sha256"], \
            f"prompt mismatch for {ln['block_id']}"
    if cfg["mode"] == "identify":
        return identify_report(d, lines)
    gen = cfg["generation"]
    blocks = []
    for ln in lines:
        row = R[ln["schema_id"]]
        b = dict(block_id=ln["block_id"], schema_id=ln["schema_id"], cell=ln["cell"], arm=ln["arm"], pair=ln["pair"],
                 model=ln["model_short"], core=ln["arm"] == "imp" and core_of(row) == (ln["model_short"], ln["pair"]),
                 situation=row["text_" + ln["cell"]], items=[], flags=[], n_items=0, preamble="", postamble="")
        f = b["flags"]
        if ln["status"] != "ok":
            f.append("call_error")
        else:
            resp = json.loads((HERE / ln["raw_path"]).read_text(encoding="utf-8"))
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            text = msg.get("content") or ""
            items, pre, post = parse_items(text)
            b.update(items=items, n_items=len(items), preamble=pre, postamble=post)
            if not items:
                f.append("no_list")
            elif len(items) != 6:
                f.append("n_items")
            if ln.get("finish_reason") == "length":
                f.append("truncated")
            th, en, other = scripts(" ".join(items))
            if items and ("th" if th >= en else "en") != ln["cell"].split("_")[1]:
                f.append("lang_mismatch")
            if scripts(text)[2] >= 5:
                f.append("other_script")
            if pre:
                f.append("preamble")
            if post:
                f.append("postamble")
            if text.rstrip().endswith("?"):
                f.append("ends_with_question")
            if msg.get("refusal"):
                f.append("refusal")
            if any(msg.get(k) for k in ("reasoning", "reasoning_content", "reasoning_details")):
                f.append("reasoning_field")
            want = gen["gpt"]["model"] if ln["model_short"] == "gpt" else gen["deepseek"]["model"]
            if ln["model_returned"] != want or (ln["model_short"] == "ds" and ln["provider"] != gen["deepseek"]["provider"]["order"][0]):
                f.append("provider_or_model")
        blocks.append(b)
    blocks.sort(key=lambda b: (int(b["schema_id"][1:]), CELLS.index(b["cell"]), b["arm"], b["model"], b["pair"]))
    sample = json.load(open(HERE / "blocks.sample.json", encoding="utf-8"))["blocks"][0]
    for b in blocks:  # shape as blocks.sample.json
        if not (set(sample) <= set(b) and isinstance(b["core"], bool) and all(isinstance(i, str) for i in b["items"])):
            sys.exit(f"{b['block_id']}: block does not match blocks.sample.json")
    core = [b for b in blocks if b["core"]]
    if cfg["mode"] == "run" and not missing:
        assert len(blocks) == 864 and len(core) == 72, (len(blocks), len(core))
        assert Counter(b["model"] for b in core) == {"gpt": 36, "ds": 36} and all(n == 24 for n in Counter(b["pair"] for b in core).values())
    FLAGS = ["call_error", "no_list", "n_items", "truncated", "lang_mismatch", "other_script", "preamble", "postamble",
             "ends_with_question", "refusal", "reasoning_field", "provider_or_model"]
    def meta(bs, note=""):
        return dict(study="silver-linings-exp1-v1", source=d.relative_to(HERE).as_posix(), parsed_at=now(), note=note,
                    n_blocks=len(bs), n_core=sum(b["core"] for b in bs), flag_counts={k: v for k, v in Counter(f for b in bs for f in b["flags"]).items()})
    (d / "blocks.json").write_text(json.dumps(dict(meta=meta(blocks), blocks=blocks), ensure_ascii=False, indent=1), encoding="utf-8")
    if core:
        (d / "blocks.core.json").write_text(json.dumps(dict(meta=meta(core, "human core only"), blocks=core), ensure_ascii=False, indent=1), encoding="utf-8")
    groups = {}
    for b in blocks:
        g = groups.setdefault((b["arm"], b["cell"], b["model"]), Counter())
        g["blocks"] += 1
        g["six_clean"] += b["n_items"] == 6 and not ({"truncated", "lang_mismatch", "call_error", "refusal"} & set(b["flags"]))
        g.update(b["flags"])
    with open(d / "parse_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cell", "model", "blocks", "six_clean"] + FLAGS)
        for (arm, cell, model), g in sorted(groups.items()):
            w.writerow([arm, cell, model, g["blocks"], g["six_clean"]] + [g[k] for k in FLAGS])
    print(f"{len(blocks)} blocks ({len(core)} core) -> {d.name}/blocks.json; flags: {meta(blocks)['flag_counts']}")
    print(open(d / "parse_report.csv", encoding="utf-8").read())
    sys.exit(1 if missing or any("call_error" in b["flags"] for b in blocks) else 0)


US_RE = re.compile(r"\b(usa|u\.s\.a?\.?|us|united states( of america)?|americ(a|an))\b|สหรัฐ|อเมริกา", re.I)
US_NOT = re.compile(r"\b(south|latin|central)\s+americ|อเมริกาใต้|ลาตินอเมริกา", re.I)
TH_RE = re.compile(r"\bthai(land)?\b|ประเทศไทย|ไทย", re.I)
NEG = re.compile(r"\bnot\b|n't\b|ไม่ใช่|ไม่น่าใช่", re.I)


def country_hit(answer, cell):
    a = answer.strip()
    if NEG.search(a):
        return False
    us, th = bool(US_RE.search(a)) and not US_NOT.search(a), bool(TH_RE.search(a))
    return (us and not th) if cell.startswith("us") else (th and not us)


def identify_report(d, lines):
    checks = {}
    for ln in lines:
        ans = ""
        if ln["status"] == "ok":
            resp = json.loads((HERE / ln["raw_path"]).read_text(encoding="utf-8"))
            ans = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        h = checks.setdefault((ln["schema_id"], ln["cell"], ln["model_short"]), dict(n=0, hits=0, answers=[]))
        h["n"] += 1
        h["hits"] += country_hit(ans, ln["cell"])
        h["answers"].append(ans.strip().replace("\n", " ") or "<error>")
    with open(d / "identify_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schema_id", "cell", "model", "n", "hits", "pass", "answers"])
        for (s, c, m), h in sorted(checks.items(), key=lambda kv: (int(kv[0][0][1:]), CELLS.index(kv[0][1]), kv[0][2])):
            w.writerow([s, c, m, h["n"], h["hits"], h["n"] == 5 and h["hits"] >= 4, " | ".join(h["answers"])])
    texts = {}
    for (s, c, m), h in checks.items():
        texts.setdefault((s, c), []).append(h["n"] == 5 and h["hits"] >= 4)
    failing = sorted((s, c) for (s, c), ok in texts.items() if not all(ok) or len(ok) < 2)
    print(f"{len(texts)} texts x {len(MODELS)} models -> {d.name}/identify_report.csv; texts failing on either model: {len(failing)}")
    for s, c in failing:
        print("  ", s, c, "  ".join(f"{m}: {checks[(s, c, m)]['hits']}/{checks[(s, c, m)]['n']}" for m in MODELS if (s, c, m) in checks))
    sys.exit(1 if failing else 0)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("identify", "pilot", "run"):
        rows, tpl = load()
        execute(cmd, jobs(cmd, rows, tpl), tpl)
    elif cmd == "parse" and len(sys.argv) > 2:
        parse(sys.argv[2])
    else:
        sys.exit(__doc__)
