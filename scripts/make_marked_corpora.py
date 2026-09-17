"""4.4/4.5 Write marked training corpora data/train_1M.{cond}.txt for every non-P0 condition.

P1, P2G-k use the Mecab cache; P2R-k uses only the k-list + stem table (rule b).
Also logs the number of inserted marks per condition (results/mark_stats.json).
Usage: python scripts/make_marked_corpora.py [cond ...]   (default: all 7)
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import TRAIN_TXT, TRAIN_MECAB, DATA, SEED_DIR, RESULTS_SEED, SPLIT_MARK, K_LIST
from pretok import get_condition
from build_stem_table import load_stem_table

ALL = ["P1"] + [f"P2G-{k}" for k in K_LIST] + [f"P2R-{k}" for k in K_LIST]


def main(conds):
    from build_stem_table import stem_table_for
    objs = []
    for c in conds:
        st = stem_table_for(c)
        objs.append(get_condition(c, stem_table=st) if st is not None else get_condition(c))
    outs = [(SEED_DIR / f"train_1M.{c}.txt").open("w", encoding="utf-8") for c in conds]
    marks = [0] * len(conds)
    n_eoj = 0
    t0 = time.time()
    with TRAIN_TXT.open(encoding="utf-8") as ft, TRAIN_MECAB.open(encoding="utf-8") as fm:
        for i, (sent, line) in enumerate(zip(ft, fm)):
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"]
            n_eoj += len(sent.split())
            for j, (o, f) in enumerate(zip(objs, outs)):
                m = o.mark(sent, morphs)
                marks[j] += m.count(SPLIT_MARK)
                f.write(m + "\n")
            if (i + 1) % 200_000 == 0:
                print(i + 1, f"{time.time()-t0:.0f}s", flush=True)
    for f in outs:
        f.close()
    stats = {c: {"marks": m, "marks_per_eojeol": m / n_eoj} for c, m in zip(conds, marks)}
    RESULTS_SEED.mkdir(parents=True, exist_ok=True)
    p = RESULTS_SEED / "mark_stats.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(stats)
    p.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ALL)
