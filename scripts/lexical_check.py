# -*- coding: utf-8 -*-
"""Coder-free convergence check: word counts on the raw generated items (no API calls).

    python lexical_check.py [--boot 2000] [--seed 20260919] [--out results-lexical.md]

Reads gen/run/blocks.json (items exactly as parsed), stimuli.csv (valence, domain) and, for the comparison with the
coder only, llm/all/codes_sol_all.csv. Writes results-lexical.md.

1. Other-mention (per item, binary): does the item name a person or group other than the asker? English: whole-word,
   case-insensitive regex over listed word forms (plurals and possessives). Thai: pythainlp newmm tokens matched
   against a token list (never raw substrings). Three list variants: core (primary), core + school, broad.
2. Religion / fate / acceptance / gratitude terms: every hit printed in context and classified by hand (CLASS below).
3. Contrasts on the per-list other-mention rate (share of a list's items with a mention; all items) with the
   definitions, weighting, bootstrap and seed of analyze.py (its functions are imported, and the setups are built
   in the same order, so the resamples are the ones behind results-llm.md).
4. Item-level cross-tab of the lexical flag against the coder's beneficiary on linings; per-cell comparison.
5. Valence check (the measure's own validity check).
"""
import csv, json, sys, re, random, statistics, pathlib, collections, math
sys.stdout.reconfigure(encoding="utf-8")
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze as A  # setups, contrast_rows, estimates, bootstrap, reading, sign: the paper's analysis code
from pythainlp.tokenize import word_tokenize
import pythainlp

TH_CHAR = re.compile(r"[\u0E00-\u0E7F]")
CELLS, MODELS, ARMS, KINDS = A.CELLS, A.MODELS, A.ARMS, A.KINDS

# ======================================================================== word lists
# English: each entry is a list of exact word forms; a match is the form, optionally followed by 's / ' (straight or
# curly), with no letter on either side (so "teamwork", "regroup", "friendly", "coaching" do not match).
EN_CORE = [
    "family", "families", "parent", "parents", "mom", "moms", "mother", "mothers", "dad", "dads", "father", "fathers",
    "sibling", "siblings", "brother", "brothers", "sister", "sisters",
    "grandparent", "grandparents", "grandma", "grandmas", "grandpa", "grandpas", "grandmother", "grandmothers",
    "grandfather", "grandfathers",
    "friend", "friends", "friendship", "friendships",
    "teammate", "teammates", "team", "teams", "classmate", "classmates", "teacher", "teachers", "coach", "coaches",
    "community", "communities", "others", "people", "everyone", "everybody", "someone", "somebody",
    "peer", "peers", "group", "groups",
]
EN_SCHOOL = ["school", "schools"]
EN_BROAD_EXTRA = [
    "person", "persons", "student", "students", "player", "players", "judge", "judges", "adult", "adults",
    "counselor", "counselors", "crowd", "crowds", "member", "members", "mentor", "mentors", "audience", "audiences",
    "opponent", "opponents", "staff", "neighbor", "neighbors", "guy", "guys", "kid", "kids", "fan", "fans",
    "captain", "captains", "tutor", "tutors", "alumni", "upperclassman", "upperclassmen", "underclassman",
    "underclassmen", "freshman", "freshmen", "sophomore", "sophomores", "juniors", "seniors", "squad", "squads",
    "crew", "crews", "club", "clubs",
]

# Thai: tokens are normalized by removing spaces and the repetition mark ๆ (เพื่อน ๆ -> เพื่อน, คนอื่นๆ -> คนอื่น), the
# Thai counterpart of an English plural. Compound tokens that newmm keeps whole and that denote a person or group are
# listed beside their head word (เพื่อนสนิท "close friend" as English "close friend" matches "friend"); compounds that
# do not denote people are left out (ทีมเวิร์ก "teamwork", สักครู่ "a moment", แม่นยำ "accurate"). The table of
# excluded compound tokens in the report shows every token containing a list word that does not count.
TH_CORE = [
    "ครอบครัว", "พ่อ", "แม่", "พ่อแม่", "ผู้ปกครอง",
    "พี่", "น้อง", "พี่น้อง", "พี่ชาย", "พี่สาว", "น้องชาย", "น้องสาว",
    "ปู่", "ย่า", "ยาย", "ปู่ย่า", "ตายาย", "ปู่ย่าตายาย",              # ตา dropped: see TH_DROPPED
    "เพื่อน", "เพื่อนสนิท", "เป็นเพื่อน", "ความเป็นเพื่อน", "เพื่อนรัก", "เพื่อนกิน", "เพื่อนแท้", "เพื่อนร่วมห้อง",
    "เพื่อนร่วมทาง", "เพื่อนร่วมชั้น", "เพื่อนร่วมทีม", "กว่าเพื่อน", "ก่อนเพื่อน", "มิตรภาพ",
    "ทีม", "ทีมงาน", "ครู", "คุณครู", "ครูประจำชั้น", "ครูฝึก", "อาจารย์", "อาจารย์ที่ปรึกษา", "โค้ช", "ชุมชน",
    "คนอื่น", "ผู้อื่น", "ผู้คน", "หลายคน", "บางคน", "ทุกคน", "ใครสักคน", "ใครคนหนึ่ง", "ใครบางคน", "คนใดคนหนึ่ง",
    "กลุ่ม", "กลุ่มคน", "รวมกลุ่ม", "เข้ากลุ่ม",
]
TH_DROPPED = {"ตา": "never a standalone token in these items; every token containing it is another word "
                    "(ตาม 'follow', ตาราง 'timetable', น้ำตา 'tears', หลับตา 'close one's eyes')"}
TH_SCHOOL = ["โรงเรียน", "โรงเรียนมัธยม", "รถโรงเรียน"]
TH_BROAD_EXTRA = [
    "คน", "นักเรียน", "เด็กนักเรียน", "นักกีฬา", "ผู้เล่น", "กรรมการ", "คณะกรรมการ", "ผู้ใหญ่", "ที่ปรึกษา",
    "สมาชิก", "เป็นสมาชิก", "พี่เลี้ยง", "พี่ใหญ่", "คู่แข่ง", "แฟน", "กองเชียร์", "ผู้ชม", "คนดู", "ฝูงชน", "กัปตัน",
    "ติวเตอร์", "ศิษย์เก่า", "รุ่นพี่", "รุ่นน้อง", "แต่ละคน", "คนรู้จัก", "คนใน", "คนนอก", "เจ้าหน้าที่", "ญาติ",
    "ชมรม", "เด็ก",
]
# head words whose containing tokens are audited (which compounds count, which do not)
TH_AUDIT_HEADS = ["พ่อ", "แม่", "พี่", "น้อง", "ปู่", "ย่า", "ตา", "ยาย", "เพื่อน", "ทีม", "ครู", "อาจารย์", "โค้ช",
                  "ชุมชน", "คนอื่น", "ทุกคน", "กลุ่ม", "ครอบครัว"]

# ------------------------------------------------------------------------ religion / fate / acceptance / gratitude
EN_REL = [  # (label, regex)
    ("karma", r"karma|karmic"), ("merit", r"merits?"), ("fate", r"fated?"), ("destiny", r"destiny|destined"),
    ("meant to be", r"meant\s+to\s+be"), ("god", r"god|gods|god['’]s"),
    ("pray", r"pray|prays|prayed|praying|prayers?"), ("blessing", r"bless|blessed|blessings?"),
    ("temple", r"temples?"), ("monk", r"monks?"), ("buddha", r"buddha|buddhist|buddhism"),
    ("let go", r"let(?:s|ting)?\s+(?:it\s+|that\s+|this\s+|things\s+|them\s+)?go"),
    ("accept", r"accept|accepts|accepted|accepting|acceptance"),
    ("happens for a reason", r"happens?\s+for\s+a\s+reason"),
    ("gratitude", r"gratitude|grateful|gratefully"),
    ("thank", r"thank|thanks|thankful|thanked|thanking"),               # added: counterpart of ขอบคุณ
    ("appreciate", r"appreciate|appreciates|appreciated|appreciating|appreciation"),  # added: counterpart of ซาบซึ้ง
]
TH_REL = ["กรรม", "บุญ", "ทำบุญ", "บุญคุณ", "กตัญญู", "ความกตัญญู", "โชคชะตา", "ชะตา", "ชะตากรรม", "พระ", "วัด",
          "ธรรมะ", "ปล่อยวาง", "ทำใจ", "ยอมรับ", "การยอมรับ", "เป็นที่ยอมรับ", "เกรงใจ", "ความเกรงใจ", "ขอบคุณ",
          "ขอบใจ", "ซาบซึ้ง", "ซาบซึ้งใจ"]

# Hand classification of every religion-term hit, read in context (hit id = block#item#k, k = order within the item).
# G = genuine religious / fate / acceptance / gratitude / kreng-jai framing; N = not (another sense of the word).
# subtype: religion, fate, acceptance, gratitude, krengjai.
_ACC_OUT = "accepting the outcome / disappointment"
_RECOG = "being accepted or recognised by others"
_MEAS = "วัด = measure"
_JUDGE = "newmm split กรรมการ (judges) into กรรม + การ"
_KJ_FREE = "kreng-jai sense; the lining is freedom from it"
CLASS = {
    "S1-us_en-imp-gpt-P3#5#0": ("N", "", "accepting the invitation"),
    "S1-us_th-exp-ds-P2#1#0": ("N", "", _RECOG), "S1-us_th-imp-gpt-P1#4#0": ("N", "", _RECOG),
    "S1-us_th-imp-ds-P3#6#0": ("N", "", "politeness formula in a sample refusal (thanks for inviting me)"),
    "S1-th_th-imp-ds-P2#1#0": ("N", "", _RECOG), "S1-th_th-imp-gpt-P3#6#0": ("N", "", _RECOG),
    "S2-us_en-exp-gpt-P1#6#0": ("N", "", "others appreciating being included"),
    "S2-us_en-exp-gpt-P2#6#0": ("N", "", "being valued by others"),
    "S2-us_th-exp-ds-P3#4#0": ("G", "acceptance", "accepting that friends do not owe you every outing"),
    "S2-us_th-exp-gpt-P3#3#0": ("N", "", _MEAS),
    "S2-th_th-exp-ds-P2#5#0": ("G", "acceptance", "ปล่อยวาง: letting go, not dwelling on it"),
    "S2-th_th-exp-gpt-P3#3#0": ("N", "", _MEAS), "S2-th_th-exp-gpt-P3#6#0": ("N", "", _RECOG),
    "S3-us_en-exp-ds-P3#6#0": ("G", "gratitude", "appreciate the friendship rather than take it for granted"),
    "S3-us_en-imp-ds-P2#3#0": ("N", "", "idiom: future-you will thank you"),
    "S3-us_en-imp-ds-P3#1#0": ("N", "", "idiom: future-you will appreciate it"),
    "S3-th_th-exp-ds-P3#5#0": ("N", "", _MEAS), "S3-th_th-imp-ds-P3#5#0": ("N", "", _MEAS), "S3-th_th-imp-ds-P3#5#1": ("N", "", _MEAS),
    "S3-th_en-exp-gpt-P3#6#0": ("G", "gratitude", "appreciate your routine, friends and school"),
    "S3-th_en-imp-ds-P1#6#0": ("N", "", "idiom: future you will appreciate it"),
    "S4-us_en-imp-gpt-P3#2#0": ("G", "gratitude", "appreciate the teammates' support"),
    "S5-us_en-imp-gpt-P3#4#0": ("G", "gratitude", "appreciate the relationships you built"),
    "S5-us_th-exp-gpt-P3#5#0": ("G", "acceptance", _ACC_OUT), "S5-us_th-imp-gpt-P2#5#0": ("G", "acceptance", _ACC_OUT),
    "S5-th_th-exp-gpt-P3#4#0": ("G", "acceptance", _ACC_OUT), "S5-th_th-imp-ds-P1#5#0": ("G", "acceptance", "accepting defeat (sportsmanship)"),
    "S5-th_en-exp-gpt-P1#6#0": ("G", "acceptance", "accepting defeat"),
    "S6-th_th-exp-ds-P3#1#0": ("N", "", "recognition that you made the team"),
    "S6-th_th-exp-ds-P3#4#0": ("G", "acceptance", "accepting one's role on the team"),
    "S6-th_en-exp-ds-P1#2#0": ("N", "", "coaches appreciate reliability"),
    "S8-us_th-imp-ds-P3#1#0": ("N", "", _MEAS), "S8-us_th-imp-ds-P3#1#1": ("N", "", _MEAS),
    "S8-us_th-imp-gpt-P1#4#0": ("G", "acceptance", "accepting one's feelings"),
    "S8-us_th-imp-gpt-P3#6#0": ("N", "", _MEAS), "S8-th_th-exp-ds-P2#5#0": ("N", "", _MEAS),
    "S8-th_th-exp-ds-P3#1#0": ("N", "", _MEAS), "S8-th_th-exp-ds-P3#1#1": ("N", "", _MEAS),
    "S8-th_th-exp-gpt-P2#4#0": ("N", "", _MEAS), "S8-th_th-exp-gpt-P2#5#0": ("N", "", _MEAS),
    "S8-th_th-exp-gpt-P3#5#0": ("G", "acceptance", _ACC_OUT), "S8-th_th-imp-ds-P3#1#0": ("N", "", _MEAS),
    "S9-us_en-imp-ds-P3#6#0": ("G", "acceptance", "let it go: the form is out of your hands"),
    "S9-us_th-exp-ds-P3#5#0": ("N", "", _MEAS),
    "S9-th_en-exp-ds-P2#6#0": ("G", "acceptance", "let go: not everything needs to be controlled"),
    "S9-th_en-exp-ds-P3#4#0": ("G", "acceptance", "letting go of what is out of your hands"),
    "S10-th_en-exp-ds-P3#6#0": ("G", "gratitude", "everyone said yes: worth appreciating"),
    "S11-us_en-imp-gpt-P1#6#0": ("N", "", "others will appreciate being included"),
    "S11-us_th-exp-ds-P1#3#0": ("G", "krengjai", _KJ_FREE), "S11-us_th-exp-ds-P2#1#0": ("G", "krengjai", _KJ_FREE),
    "S11-us_th-imp-ds-P1#1#0": ("G", "krengjai", _KJ_FREE), "S11-us_th-imp-ds-P2#2#0": ("G", "krengjai", _KJ_FREE),
    "S11-th_th-exp-ds-P1#2#0": ("G", "krengjai", _KJ_FREE), "S11-th_th-exp-ds-P2#1#0": ("G", "krengjai", _KJ_FREE),
    "S11-th_th-imp-ds-P1#1#0": ("G", "krengjai", _KJ_FREE), "S11-th_th-imp-ds-P2#1#0": ("G", "krengjai", _KJ_FREE),
    "S11-th_th-imp-ds-P3#1#0": ("G", "krengjai", _KJ_FREE),
    "S11-th_th-imp-gpt-P3#1#0": ("G", "acceptance", "accepting uncertainty"),
    "S12-us_th-imp-ds-P3#1#0": ("G", "krengjai", "kreng-jai sense; close enough not to need too much of it"),
    "S12-th_th-exp-ds-P3#3#0": ("G", "gratitude", "a convenience worth being thankful for"),
    "S12-th_th-imp-ds-P3#4#0": ("G", "krengjai", "kreng-jai valued: friends who are considerate of each other"),
    "S13-us_th-exp-ds-P2#5#0": ("N", "", "recognition from the coach"), "S13-us_th-exp-gpt-P2#2#0": ("N", "", "recognition from the coach"),
    "S14-us_th-exp-ds-P3#3#0": ("N", "", _MEAS),
    "S14-th_th-exp-gpt-P2#6#0": ("G", "acceptance", _ACC_OUT),
    "S14-th_th-imp-ds-P1#6#0": ("N", "", "wanting to be accepted"), "S14-th_th-imp-ds-P2#5#0": ("N", "", "wanting others to accept you"),
    "S14-th_th-imp-gpt-P2#5#0": ("G", "acceptance", _ACC_OUT), "S14-th_th-imp-gpt-P3#6#0": ("G", "acceptance", _ACC_OUT),
    "S15-th_th-imp-ds-P3#6#0": ("N", "", "narrative: nobody said thank you"),
    "S16-us_en-exp-gpt-P1#4#0": ("N", "", "others appreciate your effort (recognition)"),
    "S16-us_th-imp-ds-P3#4#0": ("N", "", "acknowledging one's own success"),
    "S16-us_th-imp-gpt-P1#5#0": ("N", "", _RECOG), "S16-us_th-imp-gpt-P3#5#0": ("N", "", _MEAS),
    "S16-th_th-exp-ds-P3#3#0": ("G", "gratitude", "thank the people behind the project"),
    "S16-th_th-imp-ds-P3#3#0": ("G", "gratitude", "a chance to express thanks to the team and teacher"),
    "S16-th_th-imp-gpt-P3#3#0": ("N", "", _MEAS),
    "S16-th_en-exp-ds-P3#4#0": ("N", "", "merit = quality of the work"),
    "S17-us_th-exp-ds-P1#4#0": ("N", "", _MEAS),
    "S17-us_th-exp-ds-P2#1#0": ("N", "", _JUDGE), "S17-us_th-exp-ds-P2#2#0": ("N", "", _JUDGE),
    "S17-us_th-exp-gpt-P2#3#0": ("G", "acceptance", _ACC_OUT), "S17-us_th-exp-gpt-P2#6#0": ("N", "", _MEAS),
    "S17-us_th-imp-ds-P1#3#0": ("G", "acceptance", _ACC_OUT), "S17-us_th-imp-ds-P3#1#0": ("N", "", _JUDGE),
    "S17-us_th-imp-gpt-P2#6#0": ("N", "", _MEAS), "S17-us_th-imp-gpt-P3#3#0": ("G", "acceptance", _ACC_OUT),
    "S17-th_th-exp-ds-P3#2#0": ("N", "", _JUDGE), "S17-th_th-exp-ds-P3#6#0": ("N", "", "work gaining recognition"),
    "S17-th_th-exp-gpt-P1#5#0": ("G", "acceptance", _ACC_OUT), "S17-th_th-exp-gpt-P3#2#0": ("N", "", _MEAS),
    "S17-th_th-exp-gpt-P3#6#0": ("N", "", "a success that deserves acknowledgement"),
    "S17-th_th-imp-ds-P3#3#0": ("N", "", _MEAS), "S17-th_th-imp-gpt-P1#6#0": ("N", "", _MEAS),
    "S17-th_th-imp-gpt-P2#4#0": ("G", "acceptance", _ACC_OUT), "S17-th_th-imp-gpt-P3#2#0": ("N", "", _MEAS),
    "S17-th_en-exp-gpt-P3#5#0": ("N", "", "accepting evaluation (being judged), not coping"),
    "S18-us_th-exp-ds-P3#4#0": ("N", "", "idiom เพื่อนร่วมชะตากรรม: people in the same boat"),
    "S18-us_th-imp-ds-P3#3#0": ("N", "", "signing as committing to the term"),
    "S18-th_th-exp-ds-P3#6#0": ("N", "", "signing as committing to the term"),
    "S18-th_th-exp-ds-P3#6#1": ("N", "", "signing as committing to the term"),
    "S18-th_th-exp-ds-P3#6#2": ("N", "", "signing as committing to the term"),
    "S18-th_en-exp-ds-P3#4#0": ("G", "gratitude", "free textbooks: something to be grateful for"),
}


# ======================================================================== matching
def en_regex(forms):
    alt = "|".join(sorted((re.escape(f) for f in forms), key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z])(?:" + alt + r")(?:['’]s|['’])?(?![A-Za-z])", re.I)


def norm(tok):
    return tok.replace(" ", "").replace("\u200b", "").replace("ๆ", "")


def tokenize(text):
    toks = word_tokenize(text, engine="newmm", keep_whitespace=True)
    spans, pos = [], 0
    for t in toks:
        j = text.find(t, pos)
        if j < 0: j = pos
        spans.append((t, j, j + len(t))); pos = j + len(t)
    return spans


def th_phrases(terms):
    """Single-token set (normalized) and multi-token phrases (in case newmm splits a listed term in context)."""
    single, phrases = set(), []
    for t in terms:
        single.add(norm(t))
        tt = [norm(x) for x in word_tokenize(t, engine="newmm") if norm(x)]
        if len(tt) > 1: phrases.append((t, tt))
    return single, phrases


def th_match(spans, single, phrases):
    """Return [(surface, start, end)] of token matches."""
    toks = [(norm(t), t, a, b) for t, a, b in spans if norm(t)]
    out, used = [], set()
    for label, seq in phrases:
        n = len(seq)
        for i in range(len(toks) - n + 1):
            if [x[0] for x in toks[i:i + n]] == seq and not any(k in used for k in range(i, i + n)):
                out.append((label, toks[i][2], toks[i + n - 1][3])); used.update(range(i, i + n))
    for i, (nt, t, a, b) in enumerate(toks):
        if i not in used and nt in single: out.append((t, a, b))
    return sorted(out, key=lambda x: x[1])


VARIANTS = {
    "core": (EN_CORE, TH_CORE),
    "school": (EN_CORE + EN_SCHOOL, TH_CORE + TH_SCHOOL),
    "broad": (EN_CORE + EN_SCHOOL + EN_BROAD_EXTRA, TH_CORE + TH_SCHOOL + TH_BROAD_EXTRA),
}
VAR_LABEL = {"core": "core lists (primary)", "school": "core + school / โรงเรียน", "broad": "broad lists"}


# ======================================================================== load
def load():
    stim = {r["schema_id"]: r for r in csv.DictReader(open(HERE / "stimuli.csv", encoding="utf-8-sig"))}
    blocks = json.load(open(HERE / "gen/run/blocks.json", encoding="utf-8"))["blocks"]
    codes = {(r["block_id"], int(r["item_idx"])): r for r in csv.DictReader(open(HERE / "llm/all/codes_sol_all.csv", encoding="utf-8-sig"))}
    en_re = {v: en_regex(en) for v, (en, th) in VARIANTS.items()}
    th_sets = {v: th_phrases(th) for v, (en, th) in VARIANTS.items()}
    rel_en = [(lab, re.compile(r"(?<![A-Za-z])(?:" + rx + r")(?![A-Za-z])", re.I)) for lab, rx in EN_REL]
    rel_th = th_phrases(TH_REL)
    items, M = [], {}
    for b in blocks:
        sch = b["schema_id"]; recs = []
        for i, text in enumerate(b["items"], start=1):
            thai = bool(TH_CHAR.search(text))
            spans = tokenize(text) if thai else None
            rec = dict(block_id=b["block_id"], idx=i, schema=sch, cell=b["cell"], arm=b["arm"], model=b["model"], pair=b["pair"],
                       valence=stim[sch]["valence"], domain=stim[sch]["domain"], lang=b["cell"][-2:], script="th" if thai else "latin",
                       text=text, spans=spans, code=codes.get((b["block_id"], i)))
            for v in VARIANTS:
                hits = [(m.group(0), m.start(), m.end()) for m in en_re[v].finditer(text)]
                if thai: hits += th_match(spans, *th_sets[v])
                rec["hits_" + v] = sorted(hits, key=lambda x: x[1]); rec[v] = bool(hits)
            rh = [(lab, m.group(0), m.start(), m.end()) for lab, rx in rel_en for m in rx.finditer(text)]
            if thai: rh += [(norm(s), s, a, e) for s, a, e in th_match(spans, *rel_th)]
            rh.sort(key=lambda x: x[2])
            rec["rel"] = [dict(hid=f"{b['block_id']}#{i}#{k}", label=lab, surface=s, start=a, end=e) for k, (lab, s, a, e) in enumerate(rh)]
            recs.append(rec); items.append(rec)
        m = dict(schema=sch, cell=b["cell"], model=b["model"], arm=b["arm"], pair=b["pair"], valence=stim[sch]["valence"],
                 domain=stim[sch]["domain"], family=(int(sch[1:]) - 1) // 3, n=len(recs))
        for v in VARIANTS: m[v] = sum(r[v] for r in recs) / len(recs)
        lin = [r for r in recs if r["code"] and r["code"]["item_type"] == "lining"]
        m["owner_share"] = (sum(r["code"]["owner"] in ("other", "group") for r in lin) / len(lin)) if lin else None
        m["core_lin"] = (sum(r["core"] for r in lin) / len(lin)) if lin else None
        M[b["block_id"]] = m
    return items, M


# ======================================================================== helpers
def esc(s):
    return s.replace("|", "\\|").replace("*", "\\*").replace("\n", " ").replace("_", "\\_")


def highlight(text, hits):
    spans = []
    for h in sorted(hits, key=lambda x: x[1]):  # merge touching spans so Thai runs like เพื่อนกลุ่ม bold as one
        a, e = h[1], h[2]
        if spans and a <= spans[-1][1]: spans[-1][1] = max(spans[-1][1], e)
        else: spans.append([a, e])
    out, pos = [], 0
    for a, e in spans:
        out.append(esc(text[pos:a])); out.append("**" + esc(text[a:e]) + "**"); pos = e
    out.append(esc(text[pos:]))
    return "".join(out)


def context(text, a, e, w=40):
    return esc(text[max(0, a - w):a]) + "**" + esc(text[a:e]) + "**" + esc(text[e:e + w])


def rescaled(M, S, metric):
    """analyze.py bootstrap with within-setup residuals inflated by sqrt(k/(k-1)) (paper/robustness_recompute.py)."""
    M2 = {b: dict(v) for b, v in M.items()}
    for key, bl in S.items():
        xs = [(b, M[b][metric]) for b in bl if M[b][metric] is not None]
        k = len(xs)
        if k < 2: continue
        mu = statistics.mean(x for _, x in xs); f = math.sqrt(k / (k - 1))
        for b, x in xs: M2[b][metric] = mu + f * (x - mu)
    return M2


def kappa(tab):
    """Cohen's kappa for a 2x2 dict {(a, b): n} with a, b in {0, 1}."""
    n = sum(tab.values())
    if not n: return float("nan")
    po = (tab[(0, 0)] + tab[(1, 1)]) / n
    pa1 = (tab[(1, 0)] + tab[(1, 1)]) / n; pb1 = (tab[(0, 1)] + tab[(1, 1)]) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def p0(x): return "–" if x is None else f"{100 * x:.0f}"
def p1(x): return "–" if x is None else f"{100 * x:.1f}"


# ======================================================================== report sections
def section_lists(items, L):
    L += ["## 1. Word lists and audit\n",
          "**English** (whole word, case-insensitive; each form may carry 's or '; a letter on either side blocks the match, so "
          "*teamwork*, *regroup*, *friendly* and *coaching* do not count). Core: " + ", ".join(EN_CORE) + ".\n",
          "Departures from the requested list: *other people* is covered by *people*; *everybody*, *somebody*, *grandmother*, "
          "*grandfather* and *friendship(s)* are added as variants (*friendship* mirrors Thai มิตรภาพ / ความเป็นเพื่อน); *school* "
          "is left out of the core (below).\n",
          "**Thai** (pythainlp " + pythainlp.__version__ + " `word_tokenize(engine=\"newmm\")`; a token counts when, with spaces and ๆ removed, it equals a "
          "list entry; listed entries that newmm splits in context are matched as token sequences). Core: " + ", ".join(TH_CORE) + ".\n",
          "Departures from the requested list: compound tokens that newmm keeps whole and that denote people were added beside their "
          "head word (เพื่อนสนิท, รุ่น- forms excepted, see broad), as were มิตรภาพ (friendship), ผู้คน / บางคน / หลายคน (people), "
          "ใครคนหนึ่ง / ใครบางคน / คนใดคนหนึ่ง (someone) and กลุ่มคน / รวมกลุ่ม / เข้ากลุ่ม (group). "
          + " ".join(f"**{k} is dropped**: {v}." for k, v in TH_DROPPED.items())
          + " ปู่, ย่า and ยาย stay in the list but never occur as tokens (ย่า and ยาย occur only inside อย่าง 'manner' and พยายาม 'try', which token matching correctly skips). "
          "Bare คน is left out of the core: it is mostly a classifier (หก คน) or a self-description (เป็นคนที่อดทนขึ้น 'become a more patient person'), "
          "the counterpart of English *person*, which the core also leaves out.\n",
          "**School.** *School* / โรงเรียน is left out of the core lists: in these items it mostly names the place or the institution (\"at school\", "
          "\"high school\", \"school day\", the invented school names in the situations), not people. The variant *core + school* adds it; "
          "the *broad* variant adds it together with bare คน / *person* and other person nouns in both languages: "
          + ", ".join(EN_BROAD_EXTRA) + " / " + ", ".join(TH_BROAD_EXTRA) + ".\n",
          "Pronouns (*they*, *them*, เขา, พวกเขา) and relationship words (*relationship*, ความสัมพันธ์) are in no list.\n"]
    th_items = [r for r in items if r["script"] == "th"]
    th_en_word = [r for r in th_items if any(re.match(r"[A-Za-z]", s) for s, a, e in r["hits_core"])]
    th_en_only = [r for r in th_en_word if all(re.match(r"[A-Za-z]", s) for s, a, e in r["hits_core"])]
    L += ["The English regex runs on every item and the Thai token list on every item containing Thai script. English cells hold "
          f"{sum(r['lang'] == 'en' for r in items)} Latin-script items; Thai cells hold {len(th_items)} Thai-script items and the "
          f"{sum(r['lang'] == 'th' and r['script'] != 'th' for r in items)} Chinese items of S10-th_th-exp-ds-P3 (explicit arm), which match nothing. "
          f"{len(th_en_word)} Thai-script items carry an English list word written in Latin script (*team*, *group*, *everyone*); "
          f"items flagged through such a word alone: {len(th_en_only)}.\n"]
    # audit: top matched tokens per language
    for lang, name in (("latin", "English"), ("th", "Thai")):
        occ = collections.Counter(); it_n = collections.Counter()
        for r in items:
            if r["script"] != lang: continue
            seen = set()
            for s, a, e in r["hits_core"]:
                key = s.lower(); occ[key] += 1; seen.add(key)
            for k in seen: it_n[k] += 1
        n_items = sum(r["script"] == lang for r in items); n_hit = sum(r["core"] for r in items if r["script"] == lang)
        L += [f"### {name}: the 40 most frequent matched tokens (core lists)\n",
              f"{n_hit} of {n_items} {name}-script items ({100 * n_hit / n_items:.1f}%) carry at least one match.\n",
              "| rank | token | occurrences | items |", "| --- | --- | --- | --- |"]
        for i, (k, n) in enumerate(occ.most_common(40), 1):
            L.append(f"| {i} | {esc(k)} | {n} | {it_n[k]} |")
        L.append("")
        extra = collections.Counter()
        core_keys = set(occ)
        for r in items:
            if r["script"] != lang: continue
            core_spans = {(a, e) for s, a, e in r["hits_core"]}
            for s, a, e in r["hits_broad"]:
                if (a, e) not in core_spans: extra[s.lower()] += 1
        L.append(f"Broad-only additions, {name} (top 20 occurrences): " + ", ".join(f"{esc(k)} {n}" for k, n in extra.most_common(20)) + ".\n")
    # Thai: tokens that contain a list word but do not count
    allspans = collections.Counter(norm(t) for r in items if r["spans"] for t, a, b in r["spans"] if norm(t))
    core_single = th_phrases(TH_CORE)[0]
    L += ["### Thai: tokens that contain a core list word and do not count\n",
          "Token matching skips these; a raw substring search would have counted them. Counts are occurrences in all Thai-script items.\n",
          "| list word | non-matching tokens containing it (count) |", "| --- | --- |"]
    for h in TH_AUDIT_HEADS:
        ex = [(t, n) for t, n in allspans.most_common() if h in t and t not in core_single]
        if ex: L.append(f"| {h} | " + ", ".join(f"{t} {n}" for t, n in ex[:18]) + (" …" if len(ex) > 18 else "") + " |")
    L.append("")
    # samples
    rnd = random.Random(20260919)
    for lang, name in (("latin", "English"), ("th", "Thai")):
        pool = [r for r in items if r["script"] == lang and r["core"]]
        L += [f"### {name}: 15 matched items, sampled at random (seed 20260919), match in bold\n"]
        for r in rnd.sample(pool, 15):
            L.append(f"- `{r['block_id']}` #{r['idx']}: {highlight(r['text'], r['hits_core'])}")
        L.append("")


def section_contrasts(items, M, B, seed, L):
    all_b = sorted(M)  # same order as analyze.py -> identical setups dict and resamples
    S = A.setups(M, all_b)
    res = {}
    L += ["## 2. Contrasts on the per-list other-mention rate\n",
          "Metric: for each generated list, the share of its items with at least one other-mention (all items, so every list enters). "
          "Contrasts, weighting and bootstrap are analyze.py's own functions, imported: list metrics averaged within setup "
          "(schema × cell × model × arm); language = Thai-language text minus English-language text of the same event, averaged over "
          "the two settings; setting = Thai event minus US event, averaged over the two languages; interaction = diagonal minus "
          f"off-diagonal; equal weight per schema; percentile bootstrap over lists within setups, {B} draws, seed {seed}, the same "
          "setup order as analyze.py, so the resampled lists are the ones behind results-llm.md. The *corrected CI* column is the same "
          "bootstrap with within-setup residuals inflated by √(k/(k−1)), as in paper/robustness_recompute.py (with three lists per "
          "setup the plain bootstrap understates the variance). The reading column applies analyze.py's rule (10-point band) only "
          "for comparison: that band was fixed for the other-benefit share, not for this rate.\n",
          "**What each contrast compares.** The setting contrast compares texts in the same language (English US event vs English Thai "
          "event; Thai US event vs Thai Thai event), so the same word list is on both sides. The interaction is the Thai-language "
          "setting difference minus the English-language setting difference: each half is within one language and list, so a list "
          "that finds more or fewer mentions overall cancels out; only a difference in how the two lists respond to a change would "
          "leak in. **The language contrast compares the English list with the Thai list, and is only as good as the lists' "
          "equivalence**: a Thai list that catches a larger or smaller share of the mentions that are there moves it directly. Treat "
          "the language contrast as a check on the lists as much as on the models.\n"]
    for v in ("core",):
        rows = A.contrast_rows(M, S, v); est = A.estimates(rows); ci = A.bootstrap(M, S, v, B, seed)
        M2 = rescaled(M, S, v); ci2 = A.bootstrap(M2, S, v, B, seed)
        res[v] = (rows, est, ci, ci2)
        for arm, title in (("imp", "Implicit arm (main)"), ("exp", "Explicit arm")):
            L += [f"### {title}, core lists\n", "| contrast | model | estimate (points) | 95% CI | corrected CI | reading (corrected) | schemas + / − / n | sign test p |",
                  "|" + " --- |" * 8]
            for kind in KINDS:
                for model in MODELS + ("both",):
                    e = est.get((arm, model, kind)); c = ci.get((arm, model, kind)); c2 = ci2.get((arm, model, kind))
                    pos, neg, n, p = A.sign(rows, arm, model, kind)
                    L.append(f"| {kind} | {model} | {A.pts(e)} | {A.ci_s(c)} | {A.ci_s(c2)} | {A.reading(e, c2)} | {pos} / {neg} / {n} | {p:.2f} |")
            L.append("")
        L += ["### Explicit minus implicit, core lists (points)\n", "| contrast | model | difference | 95% CI | corrected CI |", "|" + " --- |" * 5]
        for kind in KINDS:
            for model in MODELS + ("both",):
                k = ("rq2", model, kind)
                L.append(f"| {kind} | {model} | {A.pts(est.get(k))} | {A.ci_s(ci.get(k))} | {A.ci_s(ci2.get(k))} |")
        L.append("")
    # coder's other-benefit share, same code, for the side-by-side
    rows_o = A.contrast_rows(M, S, "owner_share"); est_o = A.estimates(rows_o); ci_o = A.bootstrap(M, S, "owner_share", B, seed)
    ci_o2 = A.bootstrap(rescaled(M, S, "owner_share"), S, "owner_share", B, seed)
    rows_c, est_c, ci_c, ci_c2 = res["core"]
    L += ["### Side by side, both models pooled: lexical other-mention rate vs the coder's other-benefit share\n",
          "The coder's numbers are recomputed here with the same functions and seed (they match results-llm.md).\n",
          "| arm | contrast | other-mention (lexical) | corrected CI | other-benefit share (coder) | corrected CI |", "|" + " --- |" * 6]
    for arm in ("imp", "exp", "rq2"):
        for kind in KINDS:
            k = (arm, "both", kind)
            L.append(f"| {dict(imp='implicit', exp='explicit', rq2='explicit − implicit')[arm]} | {kind} | {A.pts(est_c[k])} {A.ci_s(ci_c[k])} | {A.ci_s(ci_c2[k])} | "
                     f"{A.pts(est_o[k])} {A.ci_s(ci_o[k])} | {A.ci_s(ci_o2[k])} |")
    L.append("")
    # sensitivity: list variants
    L += ["### List variants, implicit arm, both models (bootstrap B/4 draws, seed + 2, as analyze.py's sensitivity rows)\n",
          "| variant | mean rate | language | setting | interaction |", "|" + " --- |" * 5]
    imp = [b for b in all_b if M[b]["arm"] == "imp"]
    for v in VARIANTS:
        r2 = A.contrast_rows(M, S, v); e2 = A.estimates(r2); c2 = A.bootstrap(rescaled(M, S, v), S, v, max(200, B // 4), seed + 2)
        mr = statistics.mean(M[b][v] for b in imp)
        L.append(f"| {VAR_LABEL[v]} | {100 * mr:.1f}% | " + " | ".join(f"{A.pts(e2[('imp', 'both', k)])} {A.ci_s(c2.get(('imp', 'both', k)))}" for k in KINDS) + " |")
    L.append("\n(Intervals in this table are the corrected bootstrap.)\n")
    return res, (est_o, ci_o, ci_o2)


def section_cells(items, M, L):
    L += ["### Per cell × model, implicit arm\n",
          "Other-mention = mean over lists of the share of items with a core-list match; *among linings* restricts the items to those the "
          "coder called linings (so it uses the coder's item type, not its beneficiary). Other-benefit share = the coder's, as in results-llm.md.\n",
          "| cell | model | lists | other-mention, all items (core) | + school | broad | other-mention among linings (core) | other-benefit share (coder) |",
          "|" + " --- |" * 8]
    for cell in CELLS:
        for model in MODELS + ("both",):
            bl = [m for m in M.values() if m["arm"] == "imp" and m["cell"] == cell and (model == "both" or m["model"] == model)]
            L.append(f"| {cell} | {model} | {len(bl)} | {p1(A.mean(m['core'] for m in bl))} | {p1(A.mean(m['school'] for m in bl))} | "
                     f"{p1(A.mean(m['broad'] for m in bl))} | {p1(A.mean(m['core_lin'] for m in bl))} | {p1(A.mean(m['owner_share'] for m in bl))} |")
    L.append("")
    # setup-level correlation
    S = collections.defaultdict(list)
    for m in M.values():
        if m["arm"] == "imp": S[(m["schema"], m["cell"], m["model"])].append(m)
    pts_ = [dict(x=A.mean(m["core"] for m in v), y=A.mean(m["owner_share"] for m in v), lang=k[1][-2:], dom=v[0]["domain"], sm=(k[0], k[2]))
            for k, v in S.items()]
    pts_ = [p for p in pts_ if p["y"] is not None]
    def corr(ps, center=None):
        if center:
            g = collections.defaultdict(list)
            for p in ps: g[p[center]].append(p)
            mx = {k: statistics.mean(p["x"] for p in v) for k, v in g.items()}; my = {k: statistics.mean(p["y"] for p in v) for k, v in g.items()}
            xs = [p["x"] - mx[p[center]] for p in ps]; ys = [p["y"] - my[p[center]] for p in ps]
        else:
            xs = [p["x"] for p in ps]; ys = [p["y"] for p in ps]
        return statistics.correlation(xs, ys)
    r_all = corr(pts_); r_en = corr([p for p in pts_ if p["lang"] == "en"]); r_th = corr([p for p in pts_ if p["lang"] == "th"])
    r_dom = corr(pts_, "dom"); r_sm = corr(pts_, "sm")
    L.append(f"Across the {len(pts_)} implicit-arm setups (schema × cell × model), the setup mean of the other-mention rate correlates with the "
             f"coder's other-benefit share at r = {r_all:.2f} (English cells {r_en:.2f}, Thai cells {r_th:.2f}). Much of that is the domain "
             f"(academic lists rarely name anyone and rarely benefit anyone else): within domain r = {r_dom:.2f}; within schema × model, the "
             f"variation the contrasts use, r = {r_sm:.2f}. Both measures are taken on the same lists, so some of the within-schema "
             "correlation is shared sampling of lists, not agreement about an effect.\n")
    return r_all, r_en, r_th, r_dom, r_sm


def section_crosstab(items, L):
    lin = [r for r in items if r["code"] and r["code"]["item_type"] == "lining"]
    L += ["## 3. Lexical flag against the coder's beneficiary\n",
          f"Items the coder called linings, both arms: {len(lin)}. Rows: the coder's beneficiary; columns: the lexical other-mention (core lists).\n",
          "| coder beneficiary | linings | lexical mention | no mention | % with mention |", "| --- | --- | --- | --- | --- |"]
    for o in ("self", "other", "group", "transcendent", "unspecified"):
        rs = [r for r in lin if r["code"]["owner"] == o]
        if not rs: continue
        k = sum(r["core"] for r in rs)
        L.append(f"| {o} | {len(rs)} | {k} | {len(rs) - k} | {100 * k / len(rs):.1f} |")
    L.append("")
    out = {}
    L += ["Agreement, treating *other or group* as the coder's positive and *self* as its negative (transcendent and unspecified left out):\n",
          "| subset | linings | agreement % | Cohen's κ | mention rate when coder says other/group | mention rate when coder says self |", "|" + " --- |" * 6]
    for name, f in (("all", lambda r: True), ("English cells", lambda r: r["lang"] == "en"), ("Thai cells", lambda r: r["lang"] == "th"),
                    ("implicit arm", lambda r: r["arm"] == "imp"), ("GPT lists", lambda r: r["model"] == "gpt"), ("DeepSeek lists", lambda r: r["model"] == "ds")):
        tab = collections.Counter()
        for r in lin:
            if r["code"]["owner"] not in ("self", "other", "group") or not f(r): continue
            tab[(int(r["code"]["owner"] in ("other", "group")), int(r["core"]))] += 1
        n = sum(tab.values()); agree = (tab[(0, 0)] + tab[(1, 1)]) / n
        sens = tab[(1, 1)] / (tab[(1, 0)] + tab[(1, 1)]); fpr = tab[(0, 1)] / (tab[(0, 0)] + tab[(0, 1)])
        out[name] = (n, agree, kappa(tab), sens, fpr)
        L.append(f"| {name} | {n} | {100 * agree:.1f} | {kappa(tab):.2f} | {100 * sens:.1f} | {100 * fpr:.1f} |")
    L += ["",
          "The two measures ask different questions. The codebook's rule is that the beneficiary is who gains, not who is mentioned "
          "(its own examples: \"you'll feel closer to them\" and \"your teammates respect you more\" are self), so a self-benefit lining "
          "often names others, and an other-benefit lining can name its beneficiary without a list word. Mentions among *self* linings "
          "are therefore expected and are not coder error. The coder never used the transcendent or unspecified beneficiary.\n"]
    return out


GENERIC = {"people", "people's", "people’s", "people'", "people’", "everyone", "everyone's", "everyone’s", "everybody", "everybody's",
           "someone", "someone's", "someone’s", "somebody", "somebody's", "others", "others'", "others’",
           "ทุกคน", "คนอื่น", "คนอื่นๆ", "ผู้อื่น", "ผู้คน", "หลายคน", "บางคน", "ใครคนหนึ่ง", "คนใดคนหนึ่ง", "ใครสักคน", "ใครบางคน"}


def section_valence(items, M, L):
    imp = [m for m in M.values() if m["arm"] == "imp"]
    L += ["## 4. Valence check (validity of the lexical measure)\n",
          "The coder's other-benefit share is about 15 points lower for negative events than for positive or neutral ones "
          "(results-llm.md: 22 / 7 / 23%). If the lexical rate tracks the same construct, negative events should show fewer "
          "other-mentions. Implicit arm; per-list rates averaged over lists.\n",
          "| valence | lists | other-mention, all lists | English cells | Thai cells | GPT | DeepSeek | broad lists | coder other-benefit share |",
          "|" + " --- |" * 9]
    for val in ("pos", "neg", "neu"):
        bl = [m for m in imp if m["valence"] == val]
        def g(f, key="core"): return p1(A.mean(m[key] for m in bl if f(m)))
        L.append(f"| {val} | {len(bl)} | {g(lambda m: True)} | {g(lambda m: m['cell'].endswith('en'))} | {g(lambda m: m['cell'].endswith('th'))} | "
                 f"{g(lambda m: m['model'] == 'gpt')} | {g(lambda m: m['model'] == 'ds')} | {g(lambda m: True, 'broad')} | {g(lambda m: True, 'owner_share')} |")
    L.append("")
    # paired by story family (each family holds one pos, one neg, one neu schema), bootstrap over the six families
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in imp:
        fam[m["family"]][m["valence"]].append(m)
    def fam_diff(metric, a, b, langf=lambda m: True):
        d = []
        for f in sorted(fam):
            xa = A.mean(m[metric] for m in fam[f][a] if langf(m)); xb = A.mean(m[metric] for m in fam[f][b] if langf(m))
            d.append(100 * (xa - xb))
        return d
    rnd = random.Random(20260919)
    L += ["Differences paired within story family (each of the six families has one positive, one negative and one neutral schema); "
          "95% percentile bootstrap over the six families (2,000 draws, seed 20260919). Six families make these intervals coarse.\n",
          "| difference | lexical, all | lexical, English cells | lexical, Thai cells | coder other-benefit share |", "|" + " --- |" * 5]
    vres = {}
    for a, b in (("pos", "neg"), ("neu", "neg"), ("pos", "neu")):
        cells = []
        for lab, metric, lf in (("all", "core", lambda m: True), ("en", "core", lambda m: m["cell"].endswith("en")),
                                ("th", "core", lambda m: m["cell"].endswith("th")), ("coder", "owner_share", lambda m: True)):
            d = fam_diff(metric, a, b, lf); draws = []
            for _ in range(2000):
                draws.append(statistics.mean(rnd.choice(d) for _ in d))
            draws.sort(); est = statistics.mean(d); lo, hi = draws[50], draws[1949]
            cells.append(f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}] ({sum(x > 0 for x in d)}/6 families > 0)")
            vres[(a, b, lab)] = (est, lo, hi)
        L.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    L.append("")
    # diagnostics: where the negative-event mentions come from
    impi = [r for r in items if r["arm"] == "imp"]
    L += ["**Why the lexical rate misses the valence effect (diagnostic, defined after seeing the table above).** Mention rate among the "
          "coder's linings by beneficiary, and the share of items whose only matches are generic people-words (*people*, *everyone*, "
          "*someone*, *others*, ทุกคน, คนอื่น, หลายคน, บางคน and the like), implicit arm:\n",
          "| valence | self linings | with a mention % | group / other linings | with a mention % | items with generic-only mentions % | top matched words |",
          "|" + " --- |" * 7]
    diag = {}
    for val in ("pos", "neg", "neu"):
        rs = [r for r in impi if r["valence"] == val]
        lin = [r for r in rs if r["code"] and r["code"]["item_type"] == "lining"]
        sl = [r for r in lin if r["code"]["owner"] == "self"]; ol = [r for r in lin if r["code"]["owner"] in ("other", "group")]
        gen_only = sum(bool(r["hits_core"]) and all(s.lower() in GENERIC for s, a, e in r["hits_core"]) for r in rs) / len(rs)
        top = collections.Counter(s.lower() for r in rs for s, a, e in r["hits_core"]).most_common(8)
        diag[val] = (sum(r["core"] for r in sl) / len(sl), sum(r["core"] for r in ol) / len(ol), gen_only)
        L.append(f"| {val} | {len(sl)} | {100 * diag[val][0]:.1f} | {len(ol)} | {100 * diag[val][1]:.1f} | {100 * gen_only:.1f} | "
                 + ", ".join(f"{esc(k)} {n}" for k, n in top) + " |")
    osh = {v: A.mean(m["owner_share"] for m in imp if m["valence"] == v) for v in ("pos", "neg", "neu")}
    L += ["",
          f"Negative events draw fewer other-benefit linings (the coder's {100 * osh['neg']:.0f}% against {100 * osh['pos']:.0f}% and "
          f"{100 * osh['neu']:.0f}%), but the self-benefit linings they do draw name other people more often ({100 * diag['neg'][0]:.0f}% against "
          f"{100 * diag['pos'][0]:.0f}% and {100 * diag['neu'][0]:.0f}%): people appear as comparison, company or support for the asker's own gain. "
          "The two shifts cancel in the item-level mention rate. Five negative-event self-benefit linings with a mention, sampled at random (seed 20260919):\n"]
    pool = [r for r in impi if r["valence"] == "neg" and r["core"] and r["code"] and r["code"]["item_type"] == "lining" and r["code"]["owner"] == "self"]
    for r in random.Random(20260919).sample(pool, 5):
        L.append(f"- `{r['block_id']}` #{r['idx']} (coder goal {r['code']['goal']}): {highlight(r['text'], r['hits_core'])}")
    L += ["",
          "By domain (implicit arm, core lists): " + "; ".join(
              f"{d} {p1(A.mean(m['core'] for m in imp if m['domain'] == d))}% (coder {p1(A.mean(m['owner_share'] for m in imp if m['domain'] == d))}%)"
              for d in ("social", "sport", "academic")) + ".\n"]
    return vres, diag


def section_religion(items, L):
    hits = [(r, h) for r in items for h in r["rel"]]
    unclassified = [h["hid"] for r, h in hits if h["hid"] not in CLASS]
    L += ["## 5. Religion, fate, acceptance and gratitude terms\n",
          "English (whole word, case-insensitive): " + "; ".join(f"{lab} (`{rx}`)" for lab, rx in EN_REL) + ". *thank* and *appreciate* are added as "
          "counterparts of ขอบคุณ and ซาบซึ้ง. Thai tokens (newmm, as above): " + ", ".join(TH_REL) + ". การยอมรับ, เป็นที่ยอมรับ, "
          "ความกตัญญู, ความเกรงใจ, ชะตากรรม (fate) and ซาบซึ้งใจ are added as compound tokens. Token matching keeps กรรม out of กิจกรรม "
          "(115 tokens), พิธีกรรม (17) and พฤติกรรม, and วัด out of จังหวัด (23); but newmm does split กรรมการ (judges) into กรรม + การ "
          "and วัดผล into วัด + ผล in some contexts, which is one reason every hit was read.\n",
          "Every hit was read in context and classified by hand: **G** = a genuine religious, fate, acceptance-as-coping, gratitude or "
          "kreng-jai framing (subtype given); **N** = another sense of the word (accepting an invitation, being accepted or recognised by "
          "others, วัด 'measure', กรรม from a split กรรมการ 'judges', the idiom \"future-you will thank you\"). Kreng-jai hits are G when "
          "the word carries its sense of deference to others' feelings; most of them present freedom from it as the upside.\n"]
    stale = [k for k in CLASS if k not in {h["hid"] for r, h in hits}]
    if unclassified or stale:
        L.append(f"**{len(unclassified)} hits are not classified; {len(stale)} classifications match no hit.**\n")
    # genuine per cell, both arms and implicit only
    L += ["### Genuine hits per cell\n", "| cell | items | genuine, both arms | per 100 items | genuine, implicit arm | acceptance | gratitude | kreng-jai | religion or fate |",
          "|" + " --- |" * 9]
    for cell in CELLS:
        g = [(r, h) for r, h in hits if r["cell"] == cell and CLASS.get(h["hid"], ("?",))[0] == "G"]
        st = collections.Counter(CLASS[h["hid"]][1] for r, h in g)
        n_it = sum(1 for r in items if r["cell"] == cell)
        L.append(f"| {cell} | {n_it} | {len(g)} | {100 * len(g) / n_it:.1f} | {sum(r['arm'] == 'imp' for r, h in g)} | {st['acceptance']} | "
                 f"{st['gratitude']} | {st['krengjai']} | {st['religion'] + st['fate']} |")
    gl = [(r, h) for r, h in hits if CLASS.get(h["hid"], ("?",))[0] == "G"]
    cod = collections.Counter((CLASS[h["hid"]][1], (r["code"] or {}).get("item_type", ""), (r["code"] or {}).get("goal", "")) for r, h in gl)
    L.append("\nHow the coder coded the items behind the genuine hits (subtype, item type, goal): "
             + "; ".join(f"{a} / {b}{(' / ' + c) if c else ''} {n}" for (a, b, c), n in cod.most_common()) + ".\n")
    # counts by cell x arm
    L += ["### Counts by cell and arm\n", "| cell | arm | items | hits | genuine | genuine by subtype |", "| --- | --- | --- | --- | --- | --- |"]
    gen_cell = collections.Counter()
    for cell in CELLS:
        for arm in ARMS:
            hs = [(r, h) for r, h in hits if r["cell"] == cell and r["arm"] == arm]
            g = [(r, h) for r, h in hs if CLASS.get(h["hid"], ("?",))[0] == "G"]
            st = collections.Counter(CLASS[h["hid"]][1] for r, h in g)
            gen_cell[(cell, arm)] = len(g)
            L.append(f"| {cell} | {arm} | {sum(1 for r in items if r['cell'] == cell and r['arm'] == arm)} | {len(hs)} | {len(g)} | "
                     + (", ".join(f"{k} {v}" for k, v in st.most_common()) or "–") + " |")
    tot_g = sum(gen_cell.values())
    L.append(f"\nAll cells: {len(hits)} hits, {tot_g} genuine, in {len(items)} items.\n")
    L += ["### By term\n", "| term | hits | genuine |", "| --- | --- | --- |"]
    by = collections.Counter(h["label"] for r, h in hits); byg = collections.Counter(h["label"] for r, h in hits if CLASS.get(h["hid"], ("?",))[0] == "G")
    for k, n in by.most_common():
        L.append(f"| {k} | {n} | {byg[k]} |")
    L += ["", "### Every hit in context\n",
          "Coder columns: the coder's item type, and for linings its goal and beneficiary, or for consolations its type.\n",
          "| # | list / item | cell | arm | term | context (±40 characters) | coder | class | note |", "|" + " --- |" * 9]
    for i, (r, h) in enumerate(sorted(hits, key=lambda x: (CELLS.index(x[0]["cell"]), x[0]["arm"], x[1]["label"], x[0]["block_id"], x[0]["idx"])), 1):
        c = r["code"] or {}
        cod = c.get("item_type", "")
        if cod == "lining": cod += f" / {c.get('goal', '')} / {c.get('owner', '')}"
        elif cod == "consolation": cod += f" / {c.get('consolation_type', '')}"
        cl = CLASS.get(h["hid"], ("?", "", "unclassified"))
        L.append(f"| {i} | `{r['block_id']}` #{r['idx']} | {r['cell']} | {r['arm']} | {esc(h['label'])} | {context(r['text'], h['start'], h['end'])} | {cod} | "
                 f"{cl[0]}{(' ' + cl[1]) if cl[0] == 'G' else ''} | {esc(cl[2])} |")
    L.append("")
    return gen_cell, len(hits), tot_g, unclassified


# ======================================================================== main
def main(B, seed, out):
    items, M = load()
    L = ["# Lexical convergence check: other-mentions and religion / fate terms in the raw items\n",
         f"Source: `gen/run/blocks.json`, {len(M)} lists, {len(items)} items, parsed items exactly as stored; no API calls. The lexical "
         "measures use no model judgment; the coder's codes (`llm/all/codes_sol_all.csv`) are read only for the side-by-side columns, "
         "section 3 and the valence comparison. Built by `python lexical_check.py`.\n",
         "**Method.** Each item is flagged if it names a person or group other than the asker (English: whole-word regex over listed word "
         "forms; Thai: pythainlp newmm tokens matched against a token list, never raw substrings). A list's other-mention rate is the share of "
         "its items flagged, over all items. The language, setting and interaction contrasts use analyze.py's own functions, weighting, "
         "bootstrap and seed. A mention is not a benefit: an item that names teammates can be a self-benefit lining in the codebook. The "
         "lexical rate is therefore a coarser, coder-free proxy that should move with the other-benefit share if the coder's null were an "
         "artefact of content it could not see, not an estimate of the same quantity. Religion, fate, acceptance and gratitude terms are "
         "counted the same way and every hit is read and classified by hand.\n"]
    summary_at = len(L)
    section_lists(items, L)
    res, coder = section_contrasts(items, M, B, seed, L)
    r_all, r_en, r_th, r_dom, r_sm = section_cells(items, M, L)
    xt = section_crosstab(items, L)
    vres, diag = section_valence(items, M, L)
    gen_cell, n_hits, n_gen, unclassified = section_religion(items, L)
    rows, est, ci, ci2 = res["core"]
    est_o, ci_o, ci_o2 = coder
    S = dict(est=est, ci2=ci2, est_o=est_o, ci_o2=ci_o2, xt=xt, vres=vres, diag=diag, n_hits=n_hits, n_gen=n_gen, items=items,
             r_all=r_all, r_dom=r_dom, r_sm=r_sm, M=M, B=B, seed=seed)
    L.insert(summary_at, what_this_shows(S))
    pathlib.Path(out).write_text("\n".join(L), encoding="utf-8")
    k = lambda kind: f"{kind} {A.pts(est[('imp', 'both', kind)])} {A.ci_s(ci[('imp', 'both', kind)])} (corrected {A.ci_s(ci2[('imp', 'both', kind)])})"
    print(f"wrote {out}; implicit, both models, core lists: " + "; ".join(k(x) for x in KINDS))
    print(f"religion terms: {n_hits} hits, {n_gen} genuine; unclassified {len(unclassified)}")
    if unclassified:
        for r in items:
            for h in r["rel"]:
                if h["hid"] in unclassified:
                    print(f"  {h['hid']} | {r['cell']} {r['arm']} | {h['label']} | {r['text'][max(0, h['start'] - 60):h['end'] + 60]}")


def what_this_shows(S):
    est, ci2, est_o, ci_o2, xt, vres = S["est"], S["ci2"], S["est_o"], S["ci_o2"], S["xt"], S["vres"]
    def c(e, cc, k): return f"{A.pts(e[k])} points {A.ci_s(cc[k])}"
    imp = lambda kind: ("imp", "both", kind)
    pn = lambda lab: vres[("pos", "neg", lab)]
    items = S["items"]
    gen = [(r, h) for r in items for h in r["rel"] if CLASS.get(h["hid"], ("?",))[0] == "G"]
    gcell = collections.Counter(r["cell"] for r, h in gen); gimp = collections.Counter(r["cell"] for r, h in gen if r["arm"] == "imp")
    gsub = collections.Counter(CLASS[h["hid"]][1] for r, h in gen)
    kj = [(r, h) for r, h in gen if CLASS[h["hid"]][1] == "krengjai"]
    acc = [(r, h) for r, h in gen if CLASS[h["hid"]][1] == "acceptance"]
    M = S["M"]
    imp_b = [b for b in sorted(M) if M[b]["arm"] == "imp"]
    Ssch = A.setups(M, sorted(M))
    e_school = A.estimates(A.contrast_rows(M, Ssch, "school"))[imp("language")]
    e_broad = A.estimates(A.contrast_rows(M, Ssch, "broad"))[imp("language")]
    lo_hi = [ci2[imp(k)] for k in KINDS]
    inside = all(-A.SESOI < lo and hi < A.SESOI for lo, hi in lo_hi)
    t = ["## What this shows\n",
         f"**The lexical measure fails its own validity check.** The coder's other-benefit share is {pn('coder')[0]:+.1f} points higher for "
         f"positive than for negative events (paired by story family, 95% CI [{pn('coder')[1]:+.1f}, {pn('coder')[2]:+.1f}]). The lexical "
         f"other-mention rate does not show this: positive minus negative is {pn('all')[0]:+.1f} points [{pn('all')[1]:+.1f}, {pn('all')[2]:+.1f}] "
         f"(English cells {pn('en')[0]:+.1f}, Thai cells {pn('th')[0]:+.1f}). Negative events draw fewer other-benefit linings, but the "
         f"self-benefit linings they draw name other people more often ({100 * S['diag']['neg'][0]:.0f}% against {100 * S['diag']['pos'][0]:.0f}% for "
         "positive events), as comparison, company or support. Across the 144 implicit-arm setups the two measures correlate at "
         f"r = {S['r_all']:.2f}, but only through domain (academic lists name few people and benefit few others): within domain r = "
         f"{S['r_dom']:.2f}, and within schema × model, the variation the contrasts use, r = {S['r_sm']:.2f}. Counting the people an item "
         "names measures how socially populated the lists are, not who the lining is for, so its contrasts cannot stand in for the coder's "
         "other-benefit contrasts or confirm them.\n",
         "**What it does show.** "
         f"(1) In the implicit arm the lexical rate barely moves with language ({c(est, ci2, imp('language'))}), setting "
         f"({c(est, ci2, imp('setting'))}) or their interaction ({c(est, ci2, imp('interaction'))}; both models pooled, corrected bootstrap"
         + (", all three intervals inside ±10 points" if inside else "") + "). "
         "Lists about Thai events are about as full of other people as lists about US events in the same language, so the coder was not "
         "facing Thai-setting lists with more social content that it then failed to credit. The language contrast here also depends on the two "
         f"word lists being equivalent, and it is sensitive to them: adding *school* / โรงเรียน moves it to {e_school:+.1f} points and the broad "
         f"lists to {e_broad:+.1f}. "
         f"(2) Item by item, the coder's other- or group-benefit linings almost always carry a lexical mention ({100 * xt['all'][3]:.0f}%; English "
         f"cells {100 * xt['English cells'][3]:.0f}%, Thai cells {100 * xt['Thai cells'][3]:.0f}%), and its self-benefit linings carry one at the "
         f"same rate in both languages ({100 * xt['English cells'][4]:.0f}% and {100 * xt['Thai cells'][4]:.0f}%; κ {xt['English cells'][2]:.2f} and "
         f"{xt['Thai cells'][2]:.2f}). Nothing here suggests the coder reads people-mentions differently in Thai and English, though the mention "
         "is a necessary rather than a sufficient sign of an other-benefit lining. "
         f"(3) The lexical rate does not reproduce the coder's explicit-arm rise in the setting contrast (explicit minus implicit: lexical "
         f"{c(est, ci2, ('rq2', 'both', 'setting'))}, coder {c(est_o, ci_o2, ('rq2', 'both', 'setting'))}); given the failed validity check, "
         "this neither supports nor undercuts that rise.\n",
         f"**Religion and fate.** Of {S['n_hits']} hits of the religion, fate, acceptance and gratitude terms in {len(items):,} items, none is a "
         "religious or fate framing: no karma, merit-making, temple, monk, prayer, destiny or \"meant to be\" appears in that sense. The "
         "codebook offers a transcendent goal, a transcendent beneficiary and a fate / meaning consolation, and the coder used none of them "
         "in 5,190 items; the word count finds nothing it should have used them for. "
         f"{S['n_gen']} hits are genuine: acceptance of an outcome, a feeling or a role, or letting go ({gsub['acceptance']}), gratitude ({gsub['gratitude']}) and "
         f"kreng-jai ({gsub['krengjai']}). By cell (both arms; implicit arm in brackets): "
         + ", ".join(f"{cl} {gcell[cl]} ({gimp[cl]})" for cl in CELLS)
         + f", each cell about 1,300 items. They are more frequent in the two Thai-language cells ({gcell['us_th'] + gcell['th_th']}) than in "
         f"the two English ones ({gcell['us_en'] + gcell['th_en']})"
         + (", and within each language somewhat more frequent for the Thai event" if gcell["th_th"] > gcell["us_th"] and gcell["th_en"] > gcell["us_en"] else "")
         + "; the counts are small and the language comparison carries the same word-list caveat as above. "
         f"All {len(kj)} kreng-jai hits are in Thai-language text, in both settings "
         f"(us_th {sum(r['cell'] == 'us_th' for r, h in kj)}, th_th {sum(r['cell'] == 'th_th' for r, h in kj)}), and "
         f"{sum(r['schema'] == 'S11' for r, h in kj)} of them come from one story (going to a film alone), where freedom from kreng-jai is the upside. "
         f"The coder coded {sum((r['code'] or {}).get('item_type') == 'lining' for r, h in acc)} of the {len(acc)} items behind the genuine "
         f"acceptance hits as linings ({sum((r['code'] or {}).get('goal') == 'growth' for r, h in acc)} with a growth goal) and "
         f"{sum((r['code'] or {}).get('item_type') == 'consolation' for r, h in acc)} as a consolation. The codebook "
         "codes a coping skill the event taught (\"you learned to handle disappointment\") as a growth lining and keeps the acceptance "
         "consolation for \"it is what it is; letting go\", so these items were coded where the codebook puts them, not missed.\n"]
    return "\n".join(t)


if __name__ == "__main__":
    a = sys.argv[1:]
    B = int(a[a.index("--boot") + 1]) if "--boot" in a else 2000
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 20260919
    out = a[a.index("--out") + 1] if "--out" in a else str(HERE / "results-lexical.md")
    main(B, seed, out)
