"""4.4 Stem table for P2R mitigation rule (b), built from the TRAIN corpus only.

stem_table[s] = (# eojeol where Mecab gold stem == s, with any particle/ending)
              + (# Hangul-only eojeol equal to s with no particle/ending morpheme, i.e. standalone)
Same gold-stem definition as stem_consistency.gold_pairs. Output: data/stem_table.tsv
"""
import sys, json
from collections import Counter
from pathlib import Path
import regex as re

sys.path.insert(0, str(Path(__file__).parent))
from config import TRAIN_TXT, TRAIN_MECAB, DATA, STEM_TABLE, is_particle_tag
from stem_consistency import gold_pairs

HANGUL_ONLY = re.compile(r"^\p{Hangul}+$")


def main():
    cnt = Counter()
    with TRAIN_TXT.open(encoding="utf-8") as ft, TRAIN_MECAB.open(encoding="utf-8") as fm:
        for i, (sent, line) in enumerate(zip(ft, fm)):
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"]
            for eoj, stem, suf in gold_pairs(sent, morphs):
                cnt[stem] += 1
            # standalone: Hangul-only eojeol whose morphemes contain no particle/ending tag
            pos = 0
            for eoj in sent.split():
                st = sent.index(eoj, pos); en = st + len(eoj); pos = en
                if not HANGUL_ONLY.match(eoj):
                    continue
                ms = [m for m in morphs if m[2] >= st and m[3] <= en]
                if ms and not any(is_particle_tag(t) or "+" in t for _, t, _, _ in ms):
                    cnt[eoj] += 1
            if (i + 1) % 200_000 == 0:
                print(i + 1, flush=True)
    out = STEM_TABLE
    with out.open("w", encoding="utf-8") as f:
        for s, c in cnt.most_common():
            f.write(f"{s}\t{c}\n")
    print(f"stems={len(cnt):,}  >=5: {sum(1 for c in cnt.values() if c >= 5):,}  -> {out}")


def stem_table_for(cond_name):
    """Which stem table a P2R-family condition uses (None for others)."""
    if cond_name.startswith("P2Rp-"):
        return load_stem_table(DATA / "stem_table_plus.tsv")
    if cond_name.startswith("P2R-"):
        return load_stem_table()
    return None


def load_stem_table(path=STEM_TABLE):
    d = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            s, c = line.rstrip("\n").split("\t")
            d[s] = int(c)
    return d


if __name__ == "__main__":
    main()
