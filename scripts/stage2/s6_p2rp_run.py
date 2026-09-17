"""S6: P2R+ (fused-stem table) evaluation — metrics (HPLT held-out), S2 cost decomposition, wiki OOD,
and error-type decomposition (under-split / over-split / boundary mismatch) of P2R and P2R+ vs P2G.
Usage: python scripts/stage2/s6_p2rp_run.py [P2Rp-30 P2Rp-100 ...]  (tokenizers must exist)
Outputs: results/stage2/s6_p2rp_metrics.csv, s6_p2rp_cost.csv, s6_p2rp_ood.csv, s6_error_types.csv
"""
import sys, csv, json
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, RESULTS, DATA, SPLIT_MARK
from metrics import evaluate, fertility_and_m3, load, get_condition, parse_cond
from build_stem_table import stem_table_for
import stem_consistency as sc
from s2_cost_decomposition import run as s2_run, COLS as S2COLS

OUT = RESULTS / "stage2"
WIKI_TXT = DATA / "ood_wiki_10k.txt"; WIKI_MEC = DATA / "ood_wiki_10k.mecab.jsonl"
MCOLS = ["cond", "k", "variant", "vocab", "eval_set", "fertility", "m2a", "m2b", "m3", "n_words", "n_tokens", "split_rate", "m3_nonascii"]


def write(rows, path, cols):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def cond_obj(name):
    st = stem_table_for(name)
    return get_condition(name, stem_table=st) if st is not None else get_condition(name)


def ood_row(name, vocab):
    tok = load(name, vocab); cond = cond_obj(name)
    r1 = fertility_and_m3(tok, cond, txt=WIKI_TXT, mecab=WIKI_MEC)
    r2 = sc.run(tok, cond, heldout_txt=WIKI_TXT, heldout_mecab=WIKI_MEC)
    c, k, v = parse_cond(name)
    return {"cond": c, "k": k, "variant": v, "vocab": vocab, "eval_set": "wiki_ood_10k",
            "fertility": round(r1["fertility"], 4), "m2a": round(r2["m2a"], 4), "m2b": round(r2["m2b"], 4),
            "m3": round(r1["m3"], 5), "n_words": r1["n_words"], "n_tokens": r1["n_tokens"],
            "split_rate": round(r1["split_rate"], 5), "m3_nonascii": round(r1["m3_nonascii"], 5)}


def error_types(ref_name, test_name, txt, mecab):
    """Compare marks of test vs ref (P2G) per eojeol. Returns counts + top examples."""
    ref = cond_obj(ref_name); test = cond_obj(test_name)
    under, over, diff = Counter(), Counter(), Counter(); n = 0
    with txt.open(encoding="utf-8") as ft, mecab.open(encoding="utf-8") as fm:
        for sent, line in zip(ft, fm):
            sent = sent.rstrip("\n"); m = json.loads(line)["m"]
            for eg, er in zip(ref.mark(sent, m).split(), test.mark(sent, m).split()):
                n += 1
                if eg == er:
                    continue
                if SPLIT_MARK in eg and SPLIT_MARK not in er:
                    under[eg.replace(SPLIT_MARK, "")] += 1
                elif SPLIT_MARK in er and SPLIT_MARK not in eg:
                    over[er.replace(SPLIT_MARK, "|")] += 1
                else:
                    diff[f"{eg.replace(SPLIT_MARK, '|')}→{er.replace(SPLIT_MARK, '|')}"] += 1
    return {"n_eojeol": n, "under": sum(under.values()), "over": sum(over.values()), "diff": sum(diff.values()),
            "under_top": under.most_common(8), "over_top": over.most_common(8), "diff_top": diff.most_common(6)}


def main(conds):
    vocab = 128000
    # metrics (HPLT)
    rows = []
    for c in conds:
        r, e = evaluate(c, vocab)
        rows.append({**r, "split_rate": e["split_rate"], "m3_nonascii": e["m3_nonascii"]})
    write(rows, OUT / "s6_p2rp_metrics.csv", MCOLS)
    # S2
    tok0 = load("P0", vocab)
    s2rows = [s2_run(c, vocab, tok0) for c in conds]
    write(s2rows, OUT / "s6_p2rp_cost.csv", S2COLS)
    for r in s2rows: print(r, flush=True)
    # OOD
    orows = [ood_row(c, vocab) for c in conds]
    write(orows, OUT / "s6_p2rp_ood.csv", MCOLS)
    for r in orows: print(r, flush=True)
    # error types: P2R-k and P2Rp-k vs P2G-k, HPLT & wiki
    erows = []; ex = ["# S6 오류 유형 분해 (기준 = P2G-k 마킹, 어절 단위)", ""]
    ks = sorted({parse_cond(c)[1] for c in conds})
    for k in ks:
        for test in (f"P2R-{k}", f"P2Rp-{k}"):
            for setname, txt, mec in (("hplt_heldout_100k", HELDOUT_TXT, HELDOUT_MECAB), ("wiki_ood_10k", WIKI_TXT, WIKI_MEC)):
                e = error_types(f"P2G-{k}", test, txt, mec)
                erows.append({"test": test, "ref": f"P2G-{k}", "eval_set": setname, "n_eojeol": e["n_eojeol"],
                              "under": e["under"], "under_rate": round(e["under"] / e["n_eojeol"], 4),
                              "over": e["over"], "over_rate": round(e["over"] / e["n_eojeol"], 4),
                              "diff": e["diff"], "diff_rate": round(e["diff"] / e["n_eojeol"], 4)})
                ex.append(f"## {test} vs P2G-{k} @ {setname}: under {e['under']:,} over {e['over']:,} diff {e['diff']:,} / {e['n_eojeol']:,}")
                ex.append("- under: " + ", ".join(f"{w}({c})" for w, c in e["under_top"]))
                ex.append("- over: " + ", ".join(f"{w}({c})" for w, c in e["over_top"]))
                ex.append("- diff: " + ", ".join(f"{w}({c})" for w, c in e["diff_top"])); ex.append("")
                print(erows[-1], flush=True)
    write(erows, OUT / "s6_error_types.csv", list(erows[0].keys()))
    (OUT / "s6_error_types.md").write_text("\n".join(ex), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["P2Rp-30", "P2Rp-100"])
