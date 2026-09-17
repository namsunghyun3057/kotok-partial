"""S10: vocabulary-size extension (base seed 20260907): train 8 conditions x {16K, 256K}, compute metrics
and cost decomposition, then regress relative delta-fertility and phi on log2(V) over 5 vocab sizes
(16K..256K). Existing result files are read only; new outputs go to results/stage2/s10_*.

Outputs: results/stage2/s10_vocab_ext.csv   (all 5 vocabs x 8 conds: fertility, m2a, m2b, split_rate,
                                             m3_nonascii, delta_fert, rel_delta, m, phi, stem_single)
         results/stage2/s10_regression.csv  (per condition: slope/intercept/R2/residuals for rel_delta and phi)
         results/stage2/s10_summary.md
"""
import sys, csv, json, time, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS, TOK_DIR, IS_BASE
assert IS_BASE
import train_tokenizers
from metrics import evaluate
from s2_cost_decomposition import run as s2_run

S2 = RESULTS / "stage2"
CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
NEW_VOCABS = (16000, 256000)
ALL_VOCABS = (16000, 32000, 64000, 128000, 256000)
log = lambda *a: print(f"[s10 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def cname(r):
    return r["cond"] if r["cond"] in ("P0", "P1") else f"P2{r['variant']}-{r['k']}"


def main():
    # 1. train missing tokenizers
    for v in NEW_VOCABS:
        for c in CONDS:
            if not (TOK_DIR / f"{c}_{v}" / "tokenizer.json").exists():
                log("train", c, v); train_tokenizers.train(c, v)
    # 2. metrics + S2 for new vocabs (cache in s10_raw.json)
    cache_p = S2 / "s10_raw.json"
    raw = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    for v in NEW_VOCABS:
        tok0 = None
        for c in CONDS:
            key = f"{c}_{v}"
            if key in raw and "fertility" in raw[key] and (c == "P0" or "from_splits" in raw[key]):
                continue
            r, e = evaluate(c, v)
            row = {"fertility": r["fertility"], "m2a": r["m2a"], "m2b": r["m2b"], "split_rate": e["split_rate"], "m3_nonascii": e["m3_nonascii"]}
            if c != "P0":
                tok0 = tok0 or train_tokenizers.load("P0", v)
                s2 = s2_run(c, v, tok0)
                row.update({"delta_fert": s2["delta_fert"], "from_splits": s2["from_splits"], "from_fragmentation": s2["from_fragmentation"],
                            "stem_single_token_rate": s2["stem_single_token_rate"]})
            raw[key] = row
            cache_p.write_text(json.dumps(raw, indent=1), encoding="utf-8")
            log("done", key, row)
    # 3. assemble 5-vocab table from existing files + new
    met = {(cname(r), int(r["vocab"])): r for r in csv.DictReader((S2 / "metrics_all_vocab.csv").open(encoding="utf-8"))}
    s2old = {(cname(r), int(r["vocab"])): r for r in csv.DictReader((S2 / "s2_cost_decomposition.csv").open(encoding="utf-8"))}
    rows = []
    for v in ALL_VOCABS:
        f0 = float(met[("P0", v)]["fertility"]) if v in (32000, 64000, 128000) else raw[f"P0_{v}"]["fertility"]
        for c in CONDS:
            if v in (32000, 64000, 128000):
                m = met[(c, v)]; base = {"fertility": float(m["fertility"]), "m2a": float(m["m2a"]), "m2b": float(m["m2b"]),
                                       "split_rate": float(m["split_rate"]), "m3_nonascii": float(m["m3_nonascii"])}
                if c != "P0":
                    s = s2old[(c, v)]
                    base.update({"delta_fert": float(s["delta_fert"]), "from_splits": float(s["from_splits"]),
                                 "from_fragmentation": float(s["from_fragmentation"]), "stem_single_token_rate": float(s["stem_single_token_rate"])})
            else:
                base = raw[f"{c}_{v}"]
            row = {"cond": c, "vocab": v, "log2V": round(math.log2(v), 4), **{k: round(float(x), 5) for k, x in base.items()}}
            if c != "P0":
                row["rel_delta"] = round(base["delta_fert"] / f0, 5)
                row["phi"] = round(base["from_fragmentation"] / base["from_splits"], 5)
            rows.append(row)
    cols = ["cond", "vocab", "log2V", "fertility", "m2a", "m2b", "split_rate", "m3_nonascii", "delta_fert", "rel_delta", "from_splits", "from_fragmentation", "phi", "stem_single_token_rate"]
    with (S2 / "s10_vocab_ext.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # 4. regressions on log2(V)
    reg = []; md = ["# S10. 어휘 크기 확장 (16K–256K, 원 시드)", "",
                    "## 조건별 log2(V) 선형 회귀 (5점)", "", "| 조건 | 지표 | 기울기/두 배 | 절편 | R² | 최대 |잔차| 어휘 | 잔차(16K,32K,64K,128K,256K) |", "|---|---|---|---|---|---|---|"]
    x = np.array([math.log2(v) for v in ALL_VOCABS])
    for c in CONDS[1:]:
        for m in ("rel_delta", "phi"):
            y = np.array([next(r[m] for r in rows if r["cond"] == c and r["vocab"] == v) for v in ALL_VOCABS])
            A = np.vstack([x, np.ones_like(x)]).T
            (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
            yhat = slope * x + icpt; res = y - yhat
            r2 = 1 - ((res ** 2).sum() / ((y - y.mean()) ** 2).sum())
            worst = ALL_VOCABS[int(np.argmax(np.abs(res)))]
            reg.append({"cond": c, "metric": m, "slope_per_doubling": round(float(slope), 5), "intercept": round(float(icpt), 5), "r2": round(float(r2), 4),
                        "worst_vocab": worst, **{f"res_{v//1000}K": round(float(e), 5) for v, e in zip(ALL_VOCABS, res)}})
            scale = 100 if m == "rel_delta" else 1
            md.append(f"| {c} | {'Δfert/P0 (%)' if m=='rel_delta' else 'φ'} | {slope*scale:+.3f} | {icpt*scale:.3f} | {r2:.4f} | {worst//1000}K | " +
                      ", ".join(f"{e*scale:+.3f}" for e in res) + " |")
    with (S2 / "s10_regression.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg[0].keys())); w.writeheader(); w.writerows(reg)
    md += ["", "## 5개 어휘의 fertility / Δ(%) / φ / M2a", "", "| 조건 | " + " | ".join(f"{v//1000}K" for v in ALL_VOCABS) + " |", "|---|" + "---|" * 5]
    for c in CONDS:
        cells = []
        for v in ALL_VOCABS:
            r = next(r for r in rows if r["cond"] == c and r["vocab"] == v)
            cells.append(f"{r['fertility']:.3f}" + (f" / {r['rel_delta']*100:+.1f}% / φ{r['phi']:+.2f}" if c != "P0" else "") + f" / M2a {r['m2a']*100:.1f}")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    (S2 / "s10_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
