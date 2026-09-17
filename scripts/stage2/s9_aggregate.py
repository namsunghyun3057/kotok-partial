"""S9 aggregate: 3-seed mean and range (min-max) per condition x vocab for fertility, M2a, M2b, phi,
plus checks: (1) P2G-P2R M2a/M2b gap range across seeds, (2) sign of phi, (3) direction of predictor (c).
Outputs: results/stage2/s9_seed_summary.csv, s9_seed_summary.md"""
import sys, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS
S2 = RESULTS / "stage2"
SEEDS = [20260907, 20260908, 20260909]
CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]


def paths(seed):
    base = S2 if seed == 20260907 else S2 / "seeds" / str(seed)
    return {"met": base / "metrics_all_vocab.csv", "s2": base / "s2_cost_decomposition.csv",
            "pred": (S2 / "s3_prediction.csv") if seed == 20260907 else base / "s3_prediction_seed.csv"}


def name(r):
    return r["cond"] if r["cond"] in ("P0", "P1") else f"P2{r['variant']}-{r['k']}"


def load(seed):
    p = paths(seed); out = {}
    if not p["met"].exists():
        return None
    for r in csv.DictReader(p["met"].open(encoding="utf-8")):
        out.setdefault((name(r), int(r["vocab"])), {}).update(fertility=float(r["fertility"]), m2a=float(r["m2a"]), m2b=float(r["m2b"]))
    if p["s2"].exists():
        for r in csv.DictReader(p["s2"].open(encoding="utf-8")):
            out.setdefault((name(r), int(r["vocab"])), {})["phi"] = float(r["from_fragmentation"]) / float(r["from_splits"])
    if p["pred"].exists():
        for r in csv.DictReader(p["pred"].open(encoding="utf-8")):
            key = (r["cond"], int(r["vocab"]))
            rel_c = float(r["fert_relerr_c"]) if "fert_relerr_c" in r else float(r["relerr_c"])
            out.setdefault(key, {})["relerr_c"] = rel_c
    return out


def main():
    data = {s: load(s) for s in SEEDS}
    avail = [s for s in SEEDS if data[s]]
    rows = []; md = [f"# S9. 학습 코퍼스 재표집 (시드 {', '.join(map(str, avail))})", ""]
    for v in (32000, 64000, 128000):
        for c in CONDS:
            row = {"cond": c, "vocab": v, "n_seeds": 0}
            for m in ("fertility", "m2a", "m2b", "phi", "relerr_c"):
                vals = [data[s][(c, v)][m] for s in avail if (c, v) in data[s] and m in data[s][(c, v)]]
                if vals:
                    row[f"{m}_mean"] = round(sum(vals) / len(vals), 5); row[f"{m}_min"] = round(min(vals), 5); row[f"{m}_max"] = round(max(vals), 5)
                    row["n_seeds"] = max(row["n_seeds"], len(vals))
            rows.append(row)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cond", "vocab", "n_seeds"), k))
    with (S2 / "s9_seed_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # table: 128K mean and range
    md += ["## 128K: 조건별 3시드 평균 [최소–최대]", "", "| 조건 | fertility | M2a (%) | M2b (%) | φ |", "|---|---|---|---|---|"]
    def fmt(r, m, scale=1, nd=3):
        if f"{m}_mean" not in r: return "–"
        return f"{r[f'{m}_mean']*scale:.{nd}f} [{r[f'{m}_min']*scale:.{nd}f}–{r[f'{m}_max']*scale:.{nd}f}]"
    for r in rows:
        if r["vocab"] == 128000:
            md.append(f"| {r['cond']} | {fmt(r,'fertility')} | {fmt(r,'m2a',100,1)} | {fmt(r,'m2b',100,1)} | {fmt(r,'phi',1,2)} |")
    # (1) gap ranges
    md += ["", "## (1) P2G − P2R 격차의 시드 간 범위 (점)", "", "| 어휘 | k | M2a 격차 [범위] | M2b 격차 [범위] | fertility 차(P2R−P2G) [범위] |", "|---|---|---|---|---|"]
    for v in (32000, 64000, 128000):
        for k in (10, 30, 100):
            g = [data[s].get((f"P2G-{k}", v)) for s in avail]; rr = [data[s].get((f"P2R-{k}", v)) for s in avail]
            pairs = [(a, b) for a, b in zip(g, rr) if a and b and "m2a" in a and "m2a" in b]
            if not pairs: continue
            da = [(a["m2a"] - b["m2a"]) * 100 for a, b in pairs]; db = [(a["m2b"] - b["m2b"]) * 100 for a, b in pairs]; df = [b["fertility"] - a["fertility"] for a, b in pairs]
            md.append(f"| {v//1000}K | {k} | {sum(da)/len(da):.2f} [{min(da):.2f}–{max(da):.2f}] | {sum(db)/len(db):.2f} [{min(db):.2f}–{max(db):.2f}] | {sum(df)/len(df):+.4f} [{min(df):+.4f}–{max(df):+.4f}] |")
    # (2) phi sign
    neg = tot = 0; pos_cases = []
    for s in avail:
        for (c, v), d in data[s].items():
            if "phi" in d:
                tot += 1; neg += d["phi"] < 0
                if d["phi"] >= 0: pos_cases.append((s, c, v, d["phi"]))
    md += ["", f"## (2) φ 부호: {tot}개 (조건×어휘×시드) 중 음수 {neg}개" + ("" if not pos_cases else f"; 비음수: {pos_cases}")]
    # (3) predictor (c) direction
    over = tot2 = 0; under_cases = []
    for s in avail:
        for (c, v), d in data[s].items():
            if "relerr_c" in d:
                tot2 += 1; over += d["relerr_c"] > 0
                if d["relerr_c"] <= 0: under_cases.append((s, c, v, d["relerr_c"]))
    md += [f"## (3) 예측기 (c) 방향: {tot2}개 점 중 과대 예측 {over}개" + ("" if not under_cases else f"; 과소/동일: {under_cases}")]
    rel = [d["relerr_c"] for s in avail for d in data[s].values() if "relerr_c" in d]
    if rel: md.append(f"- (c) 상대 오차 범위: {min(rel):+.3%} ~ {max(rel):+.3%}")
    (S2 / "s9_seed_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
