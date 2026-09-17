"""4.6 M1 fertility, M3 byte-fallback ratio, plus M2 via stem_consistency; writes results/metrics.csv.

metrics.csv columns: cond, k, variant, vocab, eval_set, fertility, m2a, m2b, m3, n_words, n_tokens
Usage: python scripts/metrics.py P0 128000 [P1 128000 ...]
"""
import sys, json, csv, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, RESULTS, SPLIT_MARK
from pretok import get_condition, parse_cond
from train_tokenizers import load
import stem_consistency as sc

COLS = ["cond", "k", "variant", "vocab", "eval_set", "fertility", "m2a", "m2b", "m3", "n_words", "n_tokens"]


def fertility_and_m3(tokenizer, cond, txt=HELDOUT_TXT, mecab=HELDOUT_MECAB, limit=None, batch=5000):
    n_words = n_tokens = n_byte = n_byte_nonascii = n_split_eoj = 0
    buf = []
    def flush():
        nonlocal n_tokens, n_byte, n_byte_nonascii
        for enc in tokenizer.encode_batch(buf, add_special_tokens=False):
            n_tokens += len(enc.ids)
            n_byte += sum(1 for i in enc.ids if i < 256)
            n_byte_nonascii += sum(1 for i, t in zip(enc.ids, enc.tokens) if i < 256 and sc.token_bytes(t)[0] >= 128)
        buf.clear()
    with txt.open(encoding="utf-8") as ft, mecab.open(encoding="utf-8") as fm:
        for i, (sent, line) in enumerate(zip(ft, fm)):
            if limit and i >= limit:
                break
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"] if cond.needs_mecab else None
            n_words += len(sent.split())
            marked = cond.mark(sent, morphs)
            n_split_eoj += sum(1 for e in marked.split() if SPLIT_MARK in e)
            buf.append(marked)
            if len(buf) >= batch:
                flush()
    if buf:
        flush()
    return {"fertility": n_tokens / n_words, "m3": n_byte / n_tokens, "m3_nonascii": n_byte_nonascii / n_tokens, "split_rate": n_split_eoj / n_words, "n_words": n_words, "n_tokens": n_tokens}


def evaluate(cond_name, vocab, limit=None, **cond_kw):
    tok = load(cond_name, vocab)
    if cond_name.startswith("P2R") and "stem_table" not in cond_kw:
        from build_stem_table import stem_table_for
        st = stem_table_for(cond_name)
        if st is not None:
            cond_kw["stem_table"] = st
    cond = get_condition(cond_name, **cond_kw)
    t0 = time.time()
    r1 = fertility_and_m3(tok, cond, limit=limit)
    r2 = sc.run(tok, cond, limit=limit)
    r2.pop("per_stem")
    c, k, v = parse_cond(cond_name)
    row = {"cond": c, "k": k, "variant": v, "vocab": vocab, "eval_set": "hplt_heldout_100k",
           "fertility": round(r1["fertility"], 4), "m2a": round(r2["m2a"], 4), "m2b": round(r2["m2b"], 4),
           "m3": round(r1["m3"], 5), "n_words": r1["n_words"], "n_tokens": r1["n_tokens"]}
    extra = {"cond": c, "k": k, "variant": v, "vocab": vocab, "m3_nonascii": round(r1["m3_nonascii"], 5), "split_rate": round(r1["split_rate"], 5),
             **{k_: v_ for k_, v_ in r2.items() if k_ not in row and k_ != "offset_mismatch"}}
    print(json.dumps({**row, **extra, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
    return row, extra


def append_rows(rows, path=RESULTS / "metrics.csv", cols=COLS):
    RESULTS.mkdir(exist_ok=True)
    existing = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    key = lambda r: (r["cond"], str(r["k"]), r["variant"], str(r["vocab"]), r.get("eval_set", ""))
    new_keys = {key(r) for r in rows}
    merged = [r for r in existing if key(r) not in new_keys] + rows
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(merged)


if __name__ == "__main__":
    args = sys.argv[1:]
    rows, extras = [], []
    for i in range(0, len(args), 2):
        r, e = evaluate(args[i], int(args[i + 1]))
        rows.append(r); extras.append(e)
    append_rows(rows)
    append_rows(extras, path=RESULTS / "metrics_extra.csv", cols=list(extras[0].keys()))
    print("wrote", RESULTS / "metrics.csv")
