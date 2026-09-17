"""P2R+ stem table: the ORIGINAL stem table plus fused past-tense stems.

Single rule change (post-hoc, after the S6 error analysis): in the TRAIN Mecab tagging, a Hangul-only
eojeol whose morphemes are  [non-particle morphemes]* + FUSED + [particle/ending morphemes]*  where
FUSED has a compound tag whose first part is VV/VA/XSV/XSA and whose remaining parts are all EP
(했 = VV+EP, 됐 = VV+EP, 공부했 -> 공부 + 했(XSV+EP)), registers the surface string up to and
including FUSED as a stem candidate (했, 말했, 공부했). Counts are added to the original table.
Rules (a)(b) (>=1 syllable, >=5 occurrences) are unchanged. TRAIN corpus only.
Output: data/stem_table_plus.tsv, results/stage2/s6_stem_table_plus_stats.json
"""
import sys, json
from collections import Counter
from pathlib import Path
import regex as re

sys.path.insert(0, str(Path(__file__).parent))
from config import TRAIN_TXT, TRAIN_MECAB, DATA, RESULTS, is_particle_tag
from build_stem_table import load_stem_table

HANGUL_ONLY = re.compile(r"^\p{Hangul}+$")
FUSED_HEAD = {"VV", "VA", "XSV", "XSA"}


def is_fused_ep(tag):
    parts = tag.split("+")
    return len(parts) >= 2 and parts[0] in FUSED_HEAD and all(p == "EP" for p in parts[1:])


def main():
    cnt = Counter()
    n_eoj = 0
    with TRAIN_TXT.open(encoding="utf-8") as ft, TRAIN_MECAB.open(encoding="utf-8") as fm:
        for i, (sent, line) in enumerate(zip(ft, fm)):
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"]
            pos = 0
            for eoj in sent.split():
                st = sent.index(eoj, pos); en = st + len(eoj); pos = en
                if not HANGUL_ONLY.match(eoj):
                    continue
                ms = [m for m in morphs if m[2] >= st and m[3] <= en]
                if not ms or ms[0][2] != st or ms[-1][3] != en:
                    continue
                if not all(ms[j][3] == ms[j + 1][2] for j in range(len(ms) - 1)):
                    continue
                fi = next((j for j, m in enumerate(ms) if is_fused_ep(m[1])), None)
                if fi is None:
                    continue
                before_ok = all(not is_particle_tag(t) and "+" not in t for _, t, _, _ in ms[:fi])
                after_ok = all(is_particle_tag(t) for _, t, _, _ in ms[fi + 1:])
                if before_ok and after_ok:
                    cnt[sent[st:ms[fi][3]]] += 1
                    n_eoj += 1
            if (i + 1) % 250_000 == 0:
                print(i + 1, flush=True)
    base = load_stem_table()
    plus = Counter(base)
    plus.update(cnt)
    out = DATA / "stem_table_plus.tsv"
    with out.open("w", encoding="utf-8") as f:
        for s, c in plus.most_common():
            f.write(f"{s}\t{c}\n")
    stats = {"fused_eojeol_in_train": n_eoj, "fused_stem_types": len(cnt), "fused_stem_types_ge5": sum(1 for c in cnt.values() if c >= 5),
             "new_types_not_in_base": sum(1 for s in cnt if s not in base), "table_size_base": len(base), "table_size_plus": len(plus),
             "top20": cnt.most_common(20)}
    (RESULTS / "stage2").mkdir(exist_ok=True)
    (RESULTS / "stage2" / "s6_stem_table_plus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
