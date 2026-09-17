"""S7: auxiliary M2b on FUSED eojeol only (Hangul-only eojeol whose morphemes are
[non-particle]* + FUSED(VV/VA/XSV/XSA + EP...) + [particle/ending]+ ). Gold boundary = end of the fused
morpheme (했|다). Alignment = byte-cumulative token boundary at that position. 128K, P0/P1/P2G/P2R/P2Rp k=30,100."""
import sys, json, csv
from pathlib import Path
import regex as re
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, RESULTS, SPLIT_MARK, is_particle_tag
from pretok import get_condition, parse_cond
from train_tokenizers import load
from build_stem_table import stem_table_for
import stem_consistency as sc
H = re.compile(r"^\p{Hangul}+$"); HEAD = {"VV", "VA", "XSV", "XSA"}
def fused(tag):
    p = tag.split("+"); return len(p) >= 2 and p[0] in HEAD and all(x == "EP" for x in p[1:])
def gold(sent, morphs):
    pos = 0
    for idx, eoj in enumerate(sent.split()):
        st = sent.index(eoj, pos); en = st + len(eoj); pos = en
        if not H.match(eoj): continue
        ms = [m for m in morphs if m[2] >= st and m[3] <= en]
        if not ms or ms[0][2] != st or ms[-1][3] != en or not all(ms[i][3] == ms[i+1][2] for i in range(len(ms)-1)): continue
        fi = next((j for j, m in enumerate(ms) if fused(m[1])), None)
        if fi is None or fi == len(ms) - 1: continue
        if not all(not is_particle_tag(t) and "+" not in t for _, t, _, _ in ms[:fi]): continue
        if not all(is_particle_tag(t) for _, t, _, _ in ms[fi+1:]): continue
        yield idx, eoj, sent[st:ms[fi][3]]
def run(name, vocab=128000):
    tok = load(name, vocab); st = stem_table_for(name)
    cond = get_condition(name, stem_table=st) if st is not None else get_condition(name)
    cache = {}; n = hit = 0
    with HELDOUT_TXT.open(encoding="utf-8") as ft, HELDOUT_MECAB.open(encoding="utf-8") as fm:
        for sent, line in zip(ft, fm):
            sent = sent.rstrip("\n"); morphs = json.loads(line)["m"]
            marked = None
            for idx, eoj, stem in gold(sent, morphs):
                if marked is None: marked = cond.mark(sent, morphs).split()
                me = marked[idx]
                if me not in cache:
                    e = tok.encode(" " + me, add_special_tokens=False)
                    cache[me] = sc.StemScorer.stem_ids(tuple(e.ids), tuple(e.tokens), len((" " + stem).encode()))[0]
                n += 1; hit += cache[me]
    return n, hit / n
rows = []
for name in ["P0", "P1", "P2G-30", "P2R-30", "P2Rp-30", "P2G-100", "P2R-100", "P2Rp-100"]:
    n, r = run(name); c, k, v = parse_cond(name)
    rows.append({"cond": c, "k": k, "variant": v, "vocab": 128000, "n_fused_eojeol": n, "m2b_fused": round(r, 4)})
    print(rows[-1], flush=True)
with (RESULTS / "stage2" / "s7_m2b_fused.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
