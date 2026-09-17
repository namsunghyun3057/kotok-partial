"""S10 aggregate: merge 5 vocabs x 8 conds x 3 seeds into results/stage2/s10_vocab_ext.csv (seed column),
regress rel_delta and phi on log2(V) per seed and on the 3-seed mean -> s10_regression.csv, s10_summary.md."""
import sys, csv, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS
S2 = RESULTS / "stage2"
SEEDS = [20260907, 20260908, 20260909]
CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
VOCABS = (16000, 32000, 64000, 128000, 256000)


def cname(r):
    return r["cond"] if r["cond"] in ("P0", "P1") else f"P2{r['variant']}-{r['k']}"


def load_seed(seed):
    base = S2 if seed == SEEDS[0] else S2 / "seeds" / str(seed)
    raw = json.loads((base / "s10_raw.json").read_text(encoding="utf-8")) if (base / "s10_raw.json").exists() else {}
    met = {(cname(r), int(r["vocab"])): r for r in csv.DictReader((base / "metrics_all_vocab.csv").open(encoding="utf-8"))}
    s2 = {(cname(r), int(r["vocab"])): r for r in csv.DictReader((base / "s2_cost_decomposition.csv").open(encoding="utf-8"))}
    out = {}
    for v in VOCABS:
        for c in CONDS:
            if v in (32000, 64000, 128000):
                m = met[(c, v)]; d = {"fertility": float(m["fertility"]), "m2a": float(m["m2a"]), "m2b": float(m["m2b"]),
                                     "split_rate": float(m["split_rate"]), "m3_nonascii": float(m["m3_nonascii"])}
                if c != "P0":
                    s = s2[(c, v)]; d.update({"delta_fert": float(s["delta_fert"]), "from_splits": float(s["from_splits"]),
                                              "from_fragmentation": float(s["from_fragmentation"]), "stem_single_token_rate": float(s["stem_single_token_rate"])})
            else:
                key = f"{c}_{v}"
                if key not in raw: continue
                d = dict(raw[key])
            out[(c, v)] = d
    for (c, v), d in out.items():
        if c != "P0" and (("P0", v) in out):
            d["rel_delta"] = d["delta_fert"] / out[("P0", v)]["fertility"]
            d["phi"] = d["from_fragmentation"] / d["from_splits"]
    return out


def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (a * x + b); r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2, res


def main():
    data = {s: load_seed(s) for s in SEEDS}
    cols = ["seed", "cond", "vocab", "log2V", "fertility", "m2a", "m2b", "split_rate", "m3_nonascii", "delta_fert", "rel_delta", "from_splits", "from_fragmentation", "phi", "stem_single_token_rate"]
    rows = []
    for s in SEEDS:
        for v in VOCABS:
            for c in CONDS:
                if (c, v) not in data[s]: continue
                d = data[s][(c, v)]
                rows.append({"seed": s, "cond": c, "vocab": v, "log2V": round(math.log2(v), 4), **{k: round(d[k], 5) for k in cols[4:] if k in d}})
    with (S2 / "s10_vocab_ext.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    x = np.array([math.log2(v) for v in VOCABS])
    reg = []; md = ["# S10. 어휘 크기 확장 16K–256K × 3시드", "", "## (1) log2(V) 회귀: 시드별 기울기(두 배당)·R²와 3시드 평균", "",
                    "| 조건 | 지표 | 시드 07 기울기 (R²) | 시드 08 | 시드 09 | 3시드 평균 기울기 (R²) |", "|---|---|---|---|---|---|"]
    complete = [s for s in SEEDS if all((c, v) in data[s] for c in CONDS for v in VOCABS)]
    for c in CONDS[1:]:
        for m in ("rel_delta", "phi"):
            scale = 100 if m == "rel_delta" else 1
            cells = []; ys = []
            for s in SEEDS:
                if s in complete:
                    y = np.array([data[s][(c, v)][m] for v in VOCABS]); a, b, r2, res = fit(x, y); ys.append(y)
                    cells.append(f"{a*scale:+.3f} ({r2:.4f})")
                    reg.append({"seed": s, "cond": c, "metric": m, "slope_per_doubling": round(float(a), 5), "intercept": round(float(b), 5), "r2": round(float(r2), 4),
                                **{f"res_{v//1000}K": round(float(e), 5) for v, e in zip(VOCABS, res)}})
                else:
                    cells.append("–")
            if ys:
                ym = np.mean(ys, axis=0); a, b, r2, res = fit(x, ym)
                reg.append({"seed": "mean", "cond": c, "metric": m, "slope_per_doubling": round(float(a), 5), "intercept": round(float(b), 5), "r2": round(float(r2), 4),
                            **{f"res_{v//1000}K": round(float(e), 5) for v, e in zip(VOCABS, res)}})
                cells.append(f"{a*scale:+.3f} ({r2:.4f})")
            md.append(f"| {c} | {'Δfert/P0 (%)' if m=='rel_delta' else 'φ'} | " + " | ".join(cells) + " |")
    with (S2 / "s10_regression.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg[0].keys())); w.writeheader(); w.writerows(reg)
    md += ["", f"완결 시드: {complete}", "", "## (2) 시드 간 범위(최대−최소): fertility / M2a(점)", "",
           "| 조건 | " + " | ".join(f"{v//1000}K" for v in VOCABS) + " |", "|---|" + "---|" * len(VOCABS)]
    for c in CONDS:
        cells = []
        for v in VOCABS:
            fs = [data[s][(c, v)]["fertility"] for s in SEEDS if (c, v) in data[s]]
            ms = [data[s][(c, v)]["m2a"] * 100 for s in SEEDS if (c, v) in data[s]]
            cells.append(f"{max(fs)-min(fs):.4f} / {max(ms)-min(ms):.2f} (n={len(fs)})" if fs else "–")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    # summary of ranges by vocab group
    def grp(vs, m, scale=1):
        vals = []
        for v in vs:
            for c in CONDS:
                xs_ = [data[s][(c, v)][m] * scale for s in SEEDS if (c, v) in data[s]]
                if len(xs_) == 3: vals.append(max(xs_) - min(xs_))
        return (max(vals) if vals else float("nan")), (np.mean(vals) if vals else float("nan"))
    for label, vs in (("16K·256K", (16000, 256000)), ("32K~128K", (32000, 64000, 128000))):
        fmax, fmean = grp(vs, "fertility"); mmax, mmean = grp(vs, "m2a", 100)
        md.append(f"- {label}: fertility 범위 최대 {fmax:.4f} (평균 {fmean:.4f}); M2a 범위 최대 {mmax:.2f}점 (평균 {mmean:.2f}점)")
    (S2 / "s10_summary.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))


if __name__ == "__main__":
    main()
