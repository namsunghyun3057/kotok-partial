"""S8: document-level cluster bootstrap (4,000 resamples) of fertility, M2a, M2b for the 8 conditions
at 128K (base-seed tokenizers), plus paired CIs for P2G-k minus P2R-k.

Per-document sufficient statistics are computed once; M2a uses the full-sample stem filter
(>=2 forms, >=5 occurrences) and full-sample modal token sequence per stem (plug-in approximation),
so a resample's M2a = sum(matches)/sum(occurrences) over its documents.
Outputs: results/stage2/s8_bootstrap.csv, s8_bootstrap_diff.csv, s8_doc_stats.npz
"""
import sys, json, csv, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, DATA, RESULTS, SPLIT_MARK
from pretok import get_condition, parse_cond
from train_tokenizers import load
from build_stem_table import stem_table_for
import stem_consistency as sc

CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
VOCAB = 128000; B = 4000; BOOT_SEED = 20260907
OUT = RESULTS / "stage2"


def doc_stats(cond_name):
    tok = load(cond_name, VOCAB); st = stem_table_for(cond_name)
    cond = get_condition(cond_name, stem_table=st) if st is not None else get_condition(cond_name)
    docids = [int(x) for x in (DATA / "heldout_docids.txt").read_text().split()]
    D = max(docids) + 1
    words = np.zeros(D); toks = np.zeros(D); elig = np.zeros(D); aligned = np.zeros(D)
    enc_cache = {}
    occ = []  # (doc, stem, eoj, aligned, sids)
    stem_forms = defaultdict(set)
    with HELDOUT_TXT.open(encoding="utf-8") as ft, HELDOUT_MECAB.open(encoding="utf-8") as fm:
        for i, (sent, line) in enumerate(zip(ft, fm)):
            d = docids[i]; sent = sent.rstrip("\n"); morphs = json.loads(line)["m"]
            marked = cond.mark(sent, morphs)
            words[d] += len(sent.split()); toks[d] += len(tok.encode(marked, add_special_tokens=False).ids)
            meojs = marked.split()
            for idx, eoj, stem, suf in sc.gold_pairs_idx(sent, morphs):
                me = meojs[idx]; key = (stem, me)
                if key not in enc_cache:
                    e = tok.encode(" " + me, add_special_tokens=False)
                    enc_cache[key] = sc.StemScorer.stem_ids(tuple(e.ids), tuple(e.tokens), len((" " + stem).encode()))
                al, sids = enc_cache[key]
                elig[d] += 1; aligned[d] += al
                occ.append((d, stem, al, sids)); stem_forms[stem].add(eoj)
    # full-sample filter + mode
    cnt = Counter(o[1] for o in occ)
    keep = {s for s, c in cnt.items() if c >= 5 and len(stem_forms[s]) >= 2}
    seqs = defaultdict(Counter)
    for d, s, al, sids in occ:
        if s in keep and al: seqs[s][sids] += 1
    mode = {s: c.most_common(1)[0][0] for s, c in seqs.items()}
    m2a_occ = np.zeros(D); m2a_match = np.zeros(D)
    for d, s, al, sids in occ:
        if s in keep:
            m2a_occ[d] += 1; m2a_match[d] += (al and sids == mode.get(s))
    return np.stack([words, toks, elig, aligned, m2a_occ, m2a_match], axis=1)


def main():
    OUT.mkdir(exist_ok=True)
    stats = {}
    for c in CONDS:
        t0 = time.time(); stats[c] = doc_stats(c); print(c, "docs", stats[c].shape[0], f"{time.time()-t0:.0f}s", flush=True)
    np.savez(OUT / "s8_doc_stats.npz", **{c: stats[c] for c in CONDS})
    D = stats["P0"].shape[0]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, D, size=(B, D))
    W = np.zeros((B, D))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=D)
    def metrics(S):
        agg = W @ S  # B x 6
        return np.stack([agg[:, 1] / agg[:, 0], agg[:, 5] / agg[:, 4], agg[:, 3] / agg[:, 2]], axis=1)  # fert, m2a, m2b
    def point(S):
        a = S.sum(0); return np.array([a[1] / a[0], a[5] / a[4], a[3] / a[2]])
    rows = []; boots = {}
    for c in CONDS:
        M = metrics(stats[c]); P = point(stats[c]); boots[c] = M
        for j, name in enumerate(["fertility", "m2a", "m2b"]):
            lo, hi = np.percentile(M[:, j], [2.5, 97.5])
            rows.append({"cond": c, "vocab": VOCAB, "metric": name, "point": round(float(P[j]), 5), "ci_lo": round(float(lo), 5), "ci_hi": round(float(hi), 5),
                         "boot_sd": round(float(M[:, j].std()), 5), "B": B, "n_docs": D})
            print(rows[-1], flush=True)
    with (OUT / "s8_bootstrap.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    drows = []
    for k in (10, 30, 100):
        g, r = f"P2G-{k}", f"P2R-{k}"
        Pg, Pr = point(stats[g]), point(stats[r])
        for j, name in enumerate(["fertility", "m2a", "m2b"]):
            diff = boots[g][:, j] - boots[r][:, j]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            drows.append({"k": k, "metric": name, "P2G_minus_P2R": round(float(Pg[j] - Pr[j]), 5), "ci_lo": round(float(lo), 5), "ci_hi": round(float(hi), 5),
                          "excludes_zero": bool(lo > 0 or hi < 0)})
            print(drows[-1], flush=True)
    with (OUT / "s8_bootstrap_diff.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0].keys())); w.writeheader(); w.writerows(drows)


if __name__ == "__main__":
    main()
