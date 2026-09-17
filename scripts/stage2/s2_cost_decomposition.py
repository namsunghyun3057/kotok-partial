"""S2: decompose delta-fertility (P2/P1 vs P0) into (i) forced +1 per split point and
(ii) fragmentation excess, per eojeol, using SENTENCE-level encodings (so totals match
metrics.csv fertility exactly).

Tokens are attributed to eojeol by byte offsets (byte-cumulative over token bytes; marks
and spaces are pre-token boundaries, so no token crosses an eojeol boundary).
For a split eojeol, the stem part = text before the FIRST mark; stem_single_token_rate =
share of split eojeol whose stem part is exactly one token.

Output: results/stage2/s2_cost_decomposition.csv
  cond,k,variant,vocab,split_rate,delta_fert,from_splits,from_fragmentation,frag_share,
  stem_single_token_rate,stem_tokens_mean,n_split_eojeol,n_words
"""
import sys, json, csv, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, RESULTS, STAGE2, SPLIT_MARK
from pretok import get_condition, parse_cond
from train_tokenizers import load
from build_stem_table import load_stem_table
import stem_consistency as sc

OUT = STAGE2 / "s2_cost_decomposition.csv"
COLS = ["cond", "k", "variant", "vocab", "split_rate", "delta_fert", "from_splits", "from_fragmentation", "frag_share",
        "stem_single_token_rate", "stem_tokens_mean", "n_split_eojeol", "n_words"]


def eojeol_byte_spans(sent: str):
    """Byte spans [start, end) of each eojeol chunk incl. its leading whitespace (GPT-2 attaches the space)."""
    spans, b = [], 0
    i = 0
    n = len(sent)
    while i < n:
        j = i
        while j < n and sent[j].isspace():
            j += 1
        k = j
        while k < n and not sent[k].isspace():
            k += 1
        if k == j:
            break
        chunk_bytes = len(sent[i:k].encode("utf-8"))
        spans.append((b, b + chunk_bytes))
        b += chunk_bytes
        i = k
    return spans


def per_eojeol_counts(enc_tokens, spans):
    """Token counts per eojeol chunk, from token byte lengths."""
    counts = [0] * len(spans)
    pos = 0
    e = 0
    for t in enc_tokens:
        while e < len(spans) - 1 and pos >= spans[e][1]:
            e += 1
        counts[e] += 1
        pos += len(sc.token_bytes(t))
    return counts


def stem_token_count(enc_tokens, spans, e_idx, stem_end_byte):
    """Number of tokens of eojeol e_idx that start before stem_end_byte (absolute byte offset)."""
    pos = 0
    n = 0
    for t in enc_tokens:
        if spans[e_idx][0] <= pos < stem_end_byte:
            n += 1
        pos += len(sc.token_bytes(t))
        if pos >= spans[e_idx][1]:
            break
    return n


def run(cond_name, vocab, tok0, limit=None):
    tok = load(cond_name, vocab)
    from build_stem_table import stem_table_for
    _st = stem_table_for(cond_name)
    kw = {"stem_table": _st} if _st is not None else {}
    cond = get_condition(cond_name, **kw)
    n_words = n_split = 0
    sum_delta = sum_splits = 0
    stem_single = 0
    stem_tok_total = 0
    with HELDOUT_TXT.open(encoding="utf-8") as ft, HELDOUT_MECAB.open(encoding="utf-8") as fm:
        for i, (sent, line) in enumerate(zip(ft, fm)):
            if limit and i >= limit:
                break
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"] if cond.needs_mecab else None
            marked = cond.mark(sent, morphs)
            spans = eojeol_byte_spans(sent)
            n_words += len(spans)
            e0 = tok0.encode(sent, add_special_tokens=False).tokens
            e2 = tok.encode(marked, add_special_tokens=False).tokens
            c0 = per_eojeol_counts(e0, spans)
            c2 = per_eojeol_counts(e2, spans)
            sum_delta += sum(c2) - sum(c0)
            marked_eojs = marked.split()
            plain_eojs = sent.split()
            assert len(marked_eojs) == len(spans) == len(plain_eojs)
            for e_idx, me in enumerate(marked_eojs):
                k = me.count(SPLIT_MARK)
                if k == 0:
                    continue
                n_split += 1
                sum_splits += k
                stem = me.split(SPLIT_MARK, 1)[0]
                # absolute byte offset of the stem end: eojeol chunk start + leading whitespace + stem bytes
                lead = spans[e_idx][1] - spans[e_idx][0] - len(plain_eojs[e_idx].encode("utf-8"))
                stem_end = spans[e_idx][0] + lead + len(stem.encode("utf-8"))
                st = stem_token_count(e2, spans, e_idx, stem_end)
                stem_tok_total += st
                stem_single += (st == 1)
    c, k, v = parse_cond(cond_name)
    delta = sum_delta / n_words
    fs = sum_splits / n_words
    ff = delta - fs
    return {"cond": c, "k": k, "variant": v, "vocab": vocab, "split_rate": round(n_split / n_words, 5),
            "delta_fert": round(delta, 4), "from_splits": round(fs, 4), "from_fragmentation": round(ff, 4),
            "frag_share": round(ff / delta, 4) if delta else 0.0,
            "stem_single_token_rate": round(stem_single / n_split, 4) if n_split else float("nan"),
            "stem_tokens_mean": round(stem_tok_total / n_split, 4) if n_split else float("nan"),
            "n_split_eojeol": n_split, "n_words": n_words}


def main(vocabs=(32000, 64000, 128000), conds=("P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"), limit=None):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for vocab in vocabs:
        tok0 = load("P0", vocab)
        for c in conds:
            t0 = time.time()
            r = run(c, vocab, tok0, limit=limit)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), f"{time.time()-t0:.0f}s", flush=True)
            with OUT.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        tok0 = load("P0", 128000)
        print(run("P2G-30", 128000, tok0, limit=5000))
    else:
        main()
