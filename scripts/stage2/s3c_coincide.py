"""S3 (c): r_coincide = share of pre-tokenization mark positions (held-out) at which the P0 tokenizer
(same vocab) ALREADY has a token boundary. Uses only: the condition's marking rule, P0 tokenizer.
Never loads a P2/P1 tokenizer (asserted). Cached in results/stage2/s3_coincide.json."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, RESULTS, STAGE2, SPLIT_MARK
from pretok import get_condition
import train_tokenizers
from build_stem_table import load_stem_table
import stem_consistency as sc

CACHE = STAGE2 / "s3_coincide.json"


def load_p0(vocab):
    return train_tokenizers.load("P0", vocab)


def r_coincide(cond_name, vocab, tok0=None):
    assert cond_name != "P0"
    tok0 = tok0 or load_p0(vocab)
    kw = {"stem_table": load_stem_table()} if cond_name.startswith("P2R-") else {}
    cond = get_condition(cond_name, **kw)
    n_marks = n_hit = 0
    with HELDOUT_TXT.open(encoding="utf-8") as ft, HELDOUT_MECAB.open(encoding="utf-8") as fm:
        for sent, line in zip(ft, fm):
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"] if cond.needs_mecab else None
            marked = cond.mark(sent, morphs)
            if SPLIT_MARK not in marked:
                continue
            # byte offsets of marks in the UNMARKED sentence
            mark_bytes = set(); b = 0
            for ch in marked:
                if ch == SPLIT_MARK:
                    mark_bytes.add(b)
                else:
                    b += len(ch.encode("utf-8"))
            # P0 token boundary byte offsets
            bounds = set(); pos = 0
            for t in tok0.encode(sent, add_special_tokens=False).tokens:
                pos += len(sc.token_bytes(t)); bounds.add(pos)
            n_marks += len(mark_bytes)
            n_hit += len(mark_bytes & bounds)
    return n_hit / n_marks, n_marks


def main():
    conds = ["P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for vocab in (32000, 64000, 128000):
        tok0 = load_p0(vocab)
        for c in conds:
            key = f"{c}_{vocab}"
            if key in out:
                continue
            r, n = r_coincide(c, vocab, tok0)
            out[key] = {"r_coincide": r, "n_marks": n}
            print(key, f"r_coincide={r:.4f} n_marks={n}", flush=True)
            CACHE.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
