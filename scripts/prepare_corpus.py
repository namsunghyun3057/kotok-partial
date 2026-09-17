"""4.2 Build train_1M.txt / heldout_100k.txt from HPLT v2.0 cleaned kor_Hang.

- Deterministic doc-level shuffle with fixed SEED: (shard, row_group) order is shuffled,
  then docs inside each 1000-doc row group are shuffled. Row groups are streamed one at a time.
- Split at DOCUMENT level: held-out (100k sentences) is filled first, then train (1M).
  A document contributes to exactly one split.
- Sentence split: newline, then (?<=[.!?])\s+ . NFC normalization only.
- Filters: 10 <= len <= 300 chars, Hangul ratio >= 0.5 over letters, exact dedup (md5).
"""
import sys, unicodedata, random, json, hashlib, time
from pathlib import Path
import pyarrow.parquet as pq
import regex as re

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, DATA, TRAIN_TXT, HELDOUT_TXT, N_TRAIN, N_HELDOUT

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
HANGUL = re.compile(r"[\p{Hangul}]")
LETTER = re.compile(r"\p{L}")
NL = "\n"

def sentences(doc: str):
    for para in doc.split(NL):
        para = para.strip()
        if not para:
            continue
        for s in SENT_SPLIT.split(para):
            s = unicodedata.normalize("NFC", s.strip())
            if not (10 <= len(s) <= 300):
                continue
            n_let = len(LETTER.findall(s))
            if n_let == 0 or len(HANGUL.findall(s)) / n_let < 0.5:
                continue
            yield s

def main():
    files = sorted((DATA / "hplt_raw" / "kor_Hang").glob("*.parquet"))
    print("shards:", [f.name for f in files])
    rng = random.Random(SEED)
    units = [(f, i) for f in files for i in range(pq.ParquetFile(f).num_row_groups)]
    rng.shuffle(units)
    n_docs_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print("docs available:", n_docs_total, "row groups:", len(units))

    def doc_stream():
        for f, i in units:
            batch = pq.ParquetFile(f).read_row_group(i, columns=["text"]).column("text").to_pylist()
            rng.shuffle(batch)
            yield from batch
    docs = doc_stream()

    seen = set()
    t0 = time.time()
    stats = {"seed": SEED, "shards": [f.name for f in files], "n_docs_total": n_docs_total, "n_docs_used": {}}
    for name, path, target in (("heldout", HELDOUT_TXT, N_HELDOUT), ("train", TRAIN_TXT, N_TRAIN)):
        n = chars = eoj = ndoc = 0
        with path.open("w", encoding="utf-8") as f:
            while n < target:
                d = next(docs)
                added = 0
                for s in sentences(d):
                    h = hashlib.md5(s.encode()).digest()
                    if h in seen:
                        continue
                    seen.add(h)
                    f.write(s + NL)
                    n += 1; chars += len(s); eoj += len(s.split()); added += 1
                    if n >= target:
                        break
                ndoc += bool(added)
        stats["n_docs_used"][name] = ndoc
        stats[name] = {"n_sent": n, "avg_chars": chars / n, "avg_eojeol": eoj / n}
        print(name, stats[name], f"{time.time()-t0:.0f}s", flush=True)
    stats["elapsed_s"] = time.time() - t0
    (DATA / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
