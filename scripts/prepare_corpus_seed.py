"""S9: resample the TRAIN corpus with a new document-shuffle seed, keeping held-out fixed.

1. Replay the base shuffle (seed 20260907) exactly as prepare_corpus.py did, to recover the set of
   documents consumed while filling held-out (md5 of doc text) and verify the regenerated held-out
   equals data/heldout_100k.txt line by line.
2. Shuffle (shard,row_group) order and within-row-group docs with the NEW seed, skip held-out docs,
   dedup sentences against held-out sentences and within train, take 1,000,000 sentences.
Also writes the held-out sentence -> document index map (data/heldout_docids.txt) for the
cluster bootstrap (S8).
Usage: python scripts/prepare_corpus_seed.py 20260908
"""
import sys, random, hashlib, json, time
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_SEED, DATA, HELDOUT_TXT, N_TRAIN, N_HELDOUT
from prepare_corpus import sentences

FILES = sorted((DATA / "hplt_raw" / "kor_Hang").glob("*.parquet"))


def doc_stream(seed):
    rng = random.Random(seed)
    units = [(f, i) for f in FILES for i in range(pq.ParquetFile(f).num_row_groups)]
    rng.shuffle(units)
    for f, i in units:
        batch = pq.ParquetFile(f).read_row_group(i, columns=["text"]).column("text").to_pylist()
        rng.shuffle(batch)
        yield from batch


def replay_heldout():
    """Return (heldout_doc_md5s, heldout_sentence_md5s, sentence->docidx list)."""
    docs = doc_stream(BASE_SEED)
    seen = set(); out = []; docids = []; doc_md5 = []
    n = 0; d_idx = 0
    while n < N_HELDOUT:
        d = next(docs)
        h = hashlib.md5(d.encode()).digest()
        doc_md5.append(h)
        added = 0
        for s in sentences(d):
            sh = hashlib.md5(s.encode()).digest()
            if sh in seen:
                continue
            seen.add(sh); out.append(s); docids.append(d_idx); added += 1; n += 1
            if n >= N_HELDOUT:
                break
        d_idx += 1
    ref = HELDOUT_TXT.read_text(encoding="utf-8").splitlines()
    assert out == ref, "replayed held-out differs from data/heldout_100k.txt"
    (DATA / "heldout_docids.txt").write_text("\n".join(map(str, docids)) + "\n", encoding="utf-8")
    print(f"held-out replay OK: {len(out)} sentences, {d_idx} docs consumed, {len(set(docids))} contributing docs")
    return set(doc_md5), seen


def build(seed, heldout_docs, heldout_sents):
    out_dir = DATA / f"seed_{seed}"; out_dir.mkdir(exist_ok=True)
    seen = set(heldout_sents); n = 0; ndoc = 0; chars = eoj = 0; t0 = time.time()
    with (out_dir / "train_1M.txt").open("w", encoding="utf-8") as f:
        for d in doc_stream(seed):
            if hashlib.md5(d.encode()).digest() in heldout_docs:
                continue
            added = 0
            for s in sentences(d):
                sh = hashlib.md5(s.encode()).digest()
                if sh in seen:
                    continue
                seen.add(sh); f.write(s + "\n"); n += 1; chars += len(s); eoj += len(s.split()); added += 1
                if n >= N_TRAIN:
                    break
            ndoc += bool(added)
            if n >= N_TRAIN:
                break
    stats = {"seed": seed, "n_sent": n, "n_docs_used": ndoc, "avg_chars": chars / n, "avg_eojeol": eoj / n,
             "heldout_docs_excluded": len(heldout_docs), "elapsed_s": time.time() - t0}
    (out_dir / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]]
    hd, hs = replay_heldout()
    for sd in seeds:
        build(sd, hd, hs)
