"""S9: full pipeline for one resampled training seed. Run with env KOTOK_SEED=<seed>.
Steps (skip if output exists): tag train -> k-lists -> stem table -> 7 marked corpora -> 24 tokenizers
-> metrics 24 (HPLT held-out) -> S2 cost decomposition (21) -> r_coincide (21) + fertility predictors.
Outputs under results/stage2/seeds/<seed>/ : metrics_all_vocab.csv, s2_cost_decomposition.csv,
s3_coincide.json, s3_prediction_seed.csv
"""
import sys, os, csv, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
import config
from config import SEED, IS_BASE, SEED_DIR, TRAIN_TXT, TRAIN_MECAB, STEM_TABLE, PARTICLE_FREQ, TOK_DIR, STAGE2, K_LIST
assert not IS_BASE, "set KOTOK_SEED to a non-base seed"
import tag_corpus, particle_freq, build_stem_table, make_marked_corpora, train_tokenizers
from metrics import evaluate, COLS
from s2_cost_decomposition import run as s2_run, COLS as S2COLS
from s3c_coincide import r_coincide
import s3_prediction

CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
VOCABS = (32000, 64000, 128000)
STAGE2.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(f"[seed {SEED} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def write(rows, path, cols):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main(vocabs=VOCABS):
    assert TRAIN_TXT.exists(), TRAIN_TXT
    if not TRAIN_MECAB.exists():
        log("tagging"); tag_corpus.main(TRAIN_TXT, TRAIN_MECAB, 6)
    if not (SEED_DIR / "particles_k100.txt").exists():
        log("k-lists"); particle_freq.main()
    if not STEM_TABLE.exists():
        log("stem table"); build_stem_table.main()
    if not (SEED_DIR / "train_1M.P2R-100.txt").exists():
        log("marked corpora"); make_marked_corpora.main(make_marked_corpora.ALL)
    for v in vocabs:
        for c in CONDS:
            if not (TOK_DIR / f"{c}_{v}" / "tokenizer.json").exists():
                log("train", c, v); train_tokenizers.train(c, v)
    mpath = STAGE2 / "metrics_all_vocab.csv"
    if not mpath.exists():
        rows = []
        for v in vocabs:
            for c in CONDS:
                r, e = evaluate(c, v)
                rows.append({**r, "split_rate": e["split_rate"], "m3_nonascii": e["m3_nonascii"]})
                write(rows, mpath, COLS + ["split_rate", "m3_nonascii"])
        log("metrics done")
    spath = STAGE2 / "s2_cost_decomposition.csv"
    if not spath.exists():
        rows = []
        for v in vocabs:
            tok0 = train_tokenizers.load("P0", v)
            for c in CONDS[1:]:
                rows.append(s2_run(c, v, tok0)); write(rows, spath, S2COLS)
        log("S2 done")
    cpath = STAGE2 / "s3_coincide.json"
    rc = json.loads(cpath.read_text()) if cpath.exists() else {}
    for v in vocabs:
        tok0 = train_tokenizers.load("P0", v)
        for c in CONDS[1:]:
            if f"{c}_{v}" not in rc:
                r, n = r_coincide(c, v, tok0); rc[f"{c}_{v}"] = {"r_coincide": r, "n_marks": n}
                cpath.write_text(json.dumps(rc, indent=1))
    log("coincide done")
    # fertility predictors: naive and (c); splits_pred from this seed's table / rule marks
    cov, cum = {}, {}
    for r in csv.DictReader(PARTICLE_FREQ.open(encoding="utf-8")):
        rk = int(r["rank"]); cov[rk] = float(r["cum_share"]); cum[rk] = cum.get(rk - 1, 0) + int(r["count"])
    n_eoj = sum(len(l.split()) for l in TRAIN_TXT.open(encoding="utf-8"))
    splits_t = {k: cum[k] / n_eoj for k in K_LIST}
    splits_r = s3_prediction.rule_marks_per_eojeol()
    met = {(r["cond"], int(r["k"]), r["variant"], int(r["vocab"])): float(r["fertility"]) for r in csv.DictReader(mpath.open(encoding="utf-8"))}
    prows = []
    for v in vocabs:
        f0 = met[("P0", 0, "-", v)]
        for var in ("G", "R"):
            for k in K_LIST:
                sp = splits_t[k] if var == "G" else splits_r[k]
                obs = met[("P2", k, var, v)]; r = rc[f"P2{var}-{k}_{v}"]["r_coincide"]
                naive = f0 + sp; c = f0 + sp * (1 - r)
                prows.append({"cond": f"P2{var}-{k}", "vocab": v, "fert_obs": obs, "splits_pred": round(sp, 4), "r_coincide": round(r, 4),
                              "fert_pred_naive": round(naive, 4), "relerr_naive": round((naive - obs) / obs, 4),
                              "fert_pred_c": round(c, 4), "relerr_c": round((c - obs) / obs, 4), "coverage_k": round(cov[k], 4)})
    write(prows, STAGE2 / "s3_prediction_seed.csv", list(prows[0].keys()))
    log("prediction done; all finished")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reduced":
        main(vocabs=(128000,))
    else:
        main()
