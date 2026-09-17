"""S5: out-of-domain evaluation on Korean Wikipedia (10k sentences), 24 tokenizers.
Same preprocessing as prepare_corpus (NFC, 10-300 chars, Hangul>=0.5, dedup); doc-level shuffle with SEED.
Outputs: data/ood_wiki_10k.txt(+.mecab.jsonl), results/stage2/s5_ood.csv"""
import sys, json, csv, random, hashlib
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEED, DATA, RESULTS
from prepare_corpus import sentences
import tag_corpus
from metrics import fertility_and_m3, load, get_condition, parse_cond
import stem_consistency as sc
from build_stem_table import load_stem_table

TXT = DATA / "ood_wiki_10k.txt"
MEC = DATA / "ood_wiki_10k.mecab.jsonl"
N = 10_000
CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
COLS = ["cond", "k", "variant", "vocab", "eval_set", "fertility", "m2a", "m2b", "m3", "n_words", "n_tokens", "split_rate", "m3_nonascii"]


def build():
    f = next((DATA / "wiki_raw" / "20231101.ko").glob("*.parquet"))
    pf = pq.ParquetFile(f)
    rng = random.Random(SEED)
    rgs = list(range(pf.num_row_groups)); rng.shuffle(rgs)
    seen = set(); out = []; ndoc = 0
    for rg in rgs:
        docs = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        rng.shuffle(docs)
        for d in docs:
            added = 0
            for s in sentences(d):
                h = hashlib.md5(s.encode()).digest()
                if h in seen:
                    continue
                seen.add(h); out.append(s); added += 1
                if len(out) >= N:
                    break
            ndoc += bool(added)
            if len(out) >= N:
                break
        if len(out) >= N:
            break
    TXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wiki OOD: {len(out)} sentences from {ndoc} docs, avg chars {sum(map(len, out))/len(out):.1f}, avg eojeol {sum(len(s.split()) for s in out)/len(out):.2f}")


def main():
    if not TXT.exists():
        build()
    if not MEC.exists():
        tag_corpus.main(TXT, MEC, 8)
    st = load_stem_table()
    rows = []
    for vocab in (32000, 64000, 128000):
        for c in CONDS:
            tok = load(c, vocab)
            cond = get_condition(c, stem_table=st) if c.startswith("P2R-") else get_condition(c)
            r1 = fertility_and_m3(tok, cond, txt=TXT, mecab=MEC)
            r2 = sc.run(tok, cond, heldout_txt=TXT, heldout_mecab=MEC)
            cc, k, v = parse_cond(c)
            rows.append({"cond": cc, "k": k, "variant": v, "vocab": vocab, "eval_set": "wiki_ood_10k",
                         "fertility": round(r1["fertility"], 4), "m2a": round(r2["m2a"], 4), "m2b": round(r2["m2b"], 4),
                         "m3": round(r1["m3"], 5), "n_words": r1["n_words"], "n_tokens": r1["n_tokens"],
                         "split_rate": round(r1["split_rate"], 5), "m3_nonascii": round(r1["m3_nonascii"], 5)})
            print(rows[-1], f"n_stems={r2['n_stems']} n_eligible={r2['n_eligible_eojeol']}", flush=True)
            with (RESULTS / "stage2" / "s5_ood.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
