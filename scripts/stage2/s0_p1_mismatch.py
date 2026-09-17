"""S0: why is P1 M2a < 100%? Extract every non-modal stem occurrence and classify.
Per-occurrence marking (no plain-eojeol cache) so the analysis is exact."""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, SPLIT_MARK, RESULTS
from pretok import get_condition
from train_tokenizers import load
import stem_consistency as sc

cond_name = sys.argv[1] if len(sys.argv) > 1 else "P1"
tok = load(cond_name, 128000); cond = get_condition(cond_name)
enc_cache = {}
def encode(marked):
    if marked not in enc_cache:
        e = tok.encode(" " + marked, add_special_tokens=False)
        enc_cache[marked] = (tuple(e.ids), tuple(e.tokens))
    return enc_cache[marked]

stem_occ = defaultdict(Counter)      # stem -> Counter[(eoj, marked_eoj)]
eoj_markings = defaultdict(set)      # eoj -> set(marked)
with HELDOUT_TXT.open(encoding="utf-8") as ft, HELDOUT_MECAB.open(encoding="utf-8") as fm:
    for sent, line in zip(ft, fm):
        sent = sent.rstrip("\n"); morphs = json.loads(line)["m"]
        marked = cond.mark(sent, morphs).split()
        for idx, eoj, stem, suf in sc.gold_pairs_idx(sent, morphs):
            me = marked[idx]
            stem_occ[stem][(eoj, me)] += 1
            eoj_markings[eoj].add(me)

def stem_info(eoj, me, stem):
    ids, toks = encode(me)
    aligned, sids = sc.StemScorer.stem_ids(ids, toks, len((" " + stem).encode()))
    # marked stem part: prefix of marked eojeol covering len(stem) plain chars
    n = 0; i = 0
    while n < len(stem):
        if me[i] != SPLIT_MARK: n += 1
        i += 1
    return aligned, sids, me[:i].replace(SPLIT_MARK, "|")

total = mism = 0; n_stems = 0
cls = Counter(); examples = defaultdict(list)
for stem, forms in stem_occ.items():
    tot = sum(forms.values())
    if len({e for e, _ in forms}) < 2 or tot < 5: continue
    n_stems += 1; total += tot
    info = {key: stem_info(key[0], key[1], stem) for key in forms}
    seqs = Counter()
    for key, c in forms.items():
        al, sids, _ = info[key]
        if al: seqs[sids] += c
    mode = seqs.most_common(1)[0][0] if seqs else None
    mode_pat = Counter()
    for key, c in forms.items():
        if info[key][0] and info[key][1] == mode: mode_pat[info[key][2]] += c
    mode_pat = mode_pat.most_common(1)[0][0] if mode_pat else None
    for key, c in forms.items():
        al, sids, pat = info[key]
        if al and sids == mode: continue
        mism += c
        if not al: k = "b_unaligned"
        elif pat != mode_pat: k = "a_gold_segmentation_differs"
        else: k = "c_other"
        cls[k] += c
        if len(examples[k]) < 8:
            examples[k].append(f"stem={stem} form={key[0]} marked={key[1].replace(SPLIT_MARK,'|')} stem_pat={pat} mode_pat={mode_pat} ids={sids} mode={mode} n={c}")

multi = {e: ms for e, ms in eoj_markings.items() if len(ms) > 1}
out = [f"# S0 — {cond_name} 128K M2a 불일치 분석", "",
       f"- 대상 어간 {n_stems:,}, 출현 {total:,}, 불일치 출현 {mism:,} ({mism/total:.4%}) → M2a(per-occurrence) = {1-mism/total:.4f}",
       f"- 분류: " + ", ".join(f"{k}: {v:,}" for k, v in sorted(cls.items())),
       f"- 같은 어절 문자열이 문맥에 따라 다르게 마킹된 어절형: {len(multi):,} / {len(eoj_markings):,}", ""]
for k in sorted(examples):
    out.append(f"## {k} ({cls[k]:,})"); out += [f"- {e}" for e in examples[k]]; out.append("")
out.append("## 같은 어절, 다른 마킹 예시"); out += [f"- {e}: {sorted(m.replace(SPLIT_MARK,'|') for m in ms)}" for e, ms in list(multi.items())[:8]]
(RESULTS / "stage2" / f"s0_{cond_name.lower()}_mismatch.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
