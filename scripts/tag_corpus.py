"""4.3 Tag a sentence file with Mecab-ko and cache as JSONL.

Each line: {"m": [[surface, pos, start, end], ...]}  (line i == sentence i of input)
start/end are CHARACTER offsets into the NFC sentence (from python-mecab-ko span).
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).parent))

_tagger = None
def _init():
    global _tagger
    import mecab
    _tagger = mecab.MeCab()

def _tag(sent: str):
    out = []
    for m in _tagger.parse(sent):
        out.append([m.surface, m.feature.pos, m.span.start, m.span.end])
    return json.dumps({"m": out}, ensure_ascii=False)

def main(src, dst, workers=12):
    src, dst = Path(src), Path(dst)
    sents = src.read_text(encoding="utf-8").splitlines()
    print(f"{src.name}: {len(sents)} sentences")
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool, dst.open("w", encoding="utf-8") as f:
        for i, line in enumerate(pool.imap(_tag, sents, chunksize=2000)):
            f.write(line + "\n")
            if (i + 1) % 100_000 == 0:
                print(f"  {i+1} done, {time.time()-t0:.0f}s", flush=True)
    print(f"done {dst} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
