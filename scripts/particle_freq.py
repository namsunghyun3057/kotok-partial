"""4.3 Particle/ending surface-form frequency table and top-k lists (TRAIN corpus only).

Counting rule: a morpheme is counted iff every '+'-joined tag is in TAG_SET_T
AND it is not the first morpheme of its eojeol (i.e., it attaches to something).
The second condition excludes eojeol-initial items that Mecab tags as particles/endings
(typos, fragments), which are irrelevant for suffix splitting.
"""
import sys, json, csv
from collections import Counter
from pathlib import Path
import regex as re

sys.path.insert(0, str(Path(__file__).parent))
from config import TRAIN_TXT, TRAIN_MECAB, PARTICLE_FREQ, DATA, SEED_DIR, K_LIST, is_particle_tag

HANGUL_ONLY = re.compile(r"^\p{Hangul}+$")

def main():
    cnt = Counter()          # surface -> count (all tags pooled)
    cnt_tag = Counter()      # (surface, tag) -> count
    n_morph = n_eojeol = 0
    with TRAIN_TXT.open(encoding="utf-8") as ft, TRAIN_MECAB.open(encoding="utf-8") as fm:
        for sent, line in zip(ft, fm):
            sent = sent.rstrip("\n")
            ms = json.loads(line)["m"]
            n_morph += len(ms)
            for surf, pos, st, en in ms:
                # eojeol-initial if preceded by whitespace/start
                initial = st == 0 or sent[st - 1].isspace()
                if is_particle_tag(pos) and not initial:
                    cnt[surf] += 1
                    cnt_tag[(surf, pos)] += 1
            n_eojeol += len(sent.split())
    print(f"morphemes={n_morph:,} eojeol={n_eojeol:,} particle/ending tokens={sum(cnt.values()):,} unique={len(cnt):,}")

    # Exclusion rule (recorded in README): keep Hangul-only surface forms.
    rows = []
    total = sum(cnt.values())
    cum = 0
    rank = 0
    excluded = []
    for surf, c in cnt.most_common():
        if not HANGUL_ONLY.match(surf):
            excluded.append((surf, c))
            continue
        rank += 1
        cum += c
        tags = "|".join(f"{t}:{n}" for (s, t), n in cnt_tag.most_common() if s == surf) if rank <= 300 else ""
        rows.append({"rank": rank, "surface": surf, "count": c, "share": c / total, "cum_share": cum / total, "tags": tags})
    PARTICLE_FREQ.parent.mkdir(parents=True, exist_ok=True)
    with PARTICLE_FREQ.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for k in K_LIST:
        (SEED_DIR / f"particles_k{k}.txt").write_text("\n".join(r["surface"] for r in rows[:k]) + "\n", encoding="utf-8")
        print(f"k={k}: cum coverage {rows[k-1]['cum_share']:.4f}")
    print("excluded non-Hangul (top 10):", excluded[:10])
    print("\nTop 30:")
    for r in rows[:30]:
        print(f"{r['rank']:>3} {r['surface']:<6} {r['count']:>9,} {r['cum_share']:.4f}  {r['tags']}")

if __name__ == "__main__":
    main()
