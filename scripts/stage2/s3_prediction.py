"""S3: predict P2-k metrics from the TRAIN frequency table (+ P0 tokenizer) and compare with observed.

Pred 1  M2a_pred(k)      = cumulative frequency coverage of the top-k particle/ending list
                           (results/particle_freq.csv, train corpus only -> no leakage into held-out).
Pred 2  fert_pred(k, V)  = fert_P0(V) + splits_pred(k) * (1 + phi)
        splits_pred(k)   = expected marks per eojeol. Two table-only estimators:
            (t) sum_{i<=k} count_i / n_eojeol_train   (pure frequency table; = marks/eojeol for P2G if every
                listed particle token gets a mark, i.e. ignores eojeol-initial and interior-only effects)
            (r) marks/eojeol of the analyzer-free P2R rule applied to TRAIN (no tokenizer, no held-out)
        phi = fragmentation coefficient = from_fragmentation / from_splits (S2):
            (a) fixed, from 128K P2G-30 only;  (b) per-condition measured (upper bound of the formula).
Judgement rule (written before looking): relative error <= 5% => "predictable from the frequency table".
Outputs: results/stage2/s3_prediction.csv, figures/fig3_pred_vs_obs.png
"""
import sys, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS, FIGURES, PARTICLE_FREQ, DATA, TRAIN_TXT, K_LIST, SPLIT_MARK, STAGE2
from pretok import get_condition
from build_stem_table import load_stem_table

for f in ["Malgun Gothic", "NanumGothic"]:
    if any(ft.name == f for ft in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

OUT = RESULTS / "stage2" / "s3_prediction.csv"


def load_obs():
    rows = list(csv.DictReader((RESULTS / "stage2" / "metrics_all_vocab.csv").open(encoding="utf-8")))
    return {(r["cond"], int(r["k"]), r["variant"], int(r["vocab"])): {k: float(v) for k, v in r.items() if k in ("fertility", "m2a", "m2b", "split_rate")} for r in rows}


def load_s2():
    rows = list(csv.DictReader((RESULTS / "stage2" / "s2_cost_decomposition.csv").open(encoding="utf-8")))
    return {(r["cond"], int(r["k"]), r["variant"], int(r["vocab"])): {k: float(v) for k, v in r.items() if k not in ("cond", "variant")} for r in rows}


def table_predictors():
    cov, cum_count = {}, {}
    for r in csv.DictReader(PARTICLE_FREQ.open(encoding="utf-8")):
        rk = int(r["rank"])
        cov[rk] = float(r["cum_share"])
        cum_count[rk] = (cum_count.get(rk - 1, 0) + int(r["count"]))
    stats = json.loads((DATA / "corpus_stats.json").read_text(encoding="utf-8"))
    n_eoj_train = stats["train"]["n_sent"] * stats["train"]["avg_eojeol"]
    return cov, {k: cum_count[k] / n_eoj_train for k in K_LIST}


def rule_marks_per_eojeol():
    """(r) estimator: apply the analyzer-free P2R rule to TRAIN, count marks per eojeol. Cached."""
    cache = STAGE2 / "s3_rule_marks_train.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    st = load_stem_table()
    conds = {k: get_condition(f"P2R-{k}", stem_table=st) for k in K_LIST}
    marks = {k: 0 for k in K_LIST}; n_eoj = 0
    with TRAIN_TXT.open(encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n"); n_eoj += len(s.split())
            for k, c in conds.items():
                marks[k] += c.mark(s).count(SPLIT_MARK)
    out = {k: marks[k] / n_eoj for k in K_LIST}
    cache.write_text(json.dumps(out))
    return out


def load_coincide():
    d = json.loads((RESULTS / "stage2" / "s3_coincide.json").read_text())
    return {k: v["r_coincide"] for k, v in d.items()}


def main():
    obs = load_obs(); s2 = load_s2(); rc = load_coincide()
    first_cov = {int(k): v for k, v in json.loads((RESULTS / "stage2" / "s3_first_suffix_coverage.json").read_text()).items()}
    cov, splits_t = table_predictors()
    splits_r = rule_marks_per_eojeol()
    phi_fixed = s2[("P2", 30, "G", 128000)]["from_fragmentation"] / s2[("P2", 30, "G", 128000)]["from_splits"]
    rows = []
    for (c, k, v, V), o in sorted(obs.items()):
        if c != "P2":
            continue
        f0 = obs[("P0", 0, "-", V)]["fertility"]
        phi_cond = s2[(c, k, v, V)]["from_fragmentation"] / s2[(c, k, v, V)]["from_splits"]
        sp = splits_t[k] if v == "G" else splits_r[k]
        r = rc[f"P2{v}-{k}_{V}"]
        fc = first_cov[k]
        m2a_p0 = obs[("P0", 0, "-", V)]["m2a"]
        preds = {
            "m2a_pred": cov[k],
            "m2a_pred_first": fc,                                   # first-suffix coverage (train Mecab cache only)
            "m2a_pred_first_p0": fc + (1 - fc) * m2a_p0,            # + accidental alignment at P0 rate (P0 tokenizer)
            "fert_pred_naive": f0 + sp,                    # phi = 0: +1 token per mark
            "fert_pred_a": f0 + sp * (1 + phi_fixed),
            "fert_pred_b": f0 + sp * (1 + phi_cond),
            "fert_pred_c": f0 + sp * (1 - r),
        }
        row = {"cond": f"P2{v}-{k}", "k": k, "variant": v, "vocab": V, "fert_obs": o["fertility"], "m2a_obs": o["m2a"],
               "splits_pred": round(sp, 4), "splits_obs": s2[(c, k, v, V)]["from_splits"],
               "phi_fixed": round(phi_fixed, 4), "phi_cond": round(phi_cond, 4), "r_coincide": round(r, 4)}
        for name, val in preds.items():
            row[name] = round(val, 4)
            base = o["m2a"] if name.startswith("m2a") else o["fertility"]
            row[name.replace("pred", "relerr")] = round((val - base) / base, 4)
        rows.append(row)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def summ(name, sel):
        col = name.replace("pred", "relerr")
        rel = [r[col] for r in sel]
        obs_col = "m2a_obs" if name.startswith("m2a") else "fert_obs"
        mae = sum(abs(r[name] - r[obs_col]) for r in sel) / len(sel)
        return {"MAE": mae, "mean_rel": sum(map(abs, rel)) / len(rel), "max_rel": max(map(abs, rel)),
                "signed_mean_rel": sum(rel) / len(rel), "pass5": all(abs(x) <= 0.05 for x in rel), "n": len(sel)}
    lines = ["# S3 예측 vs 관측 (18개 점)", "", f"phi_fixed(128K P2G-30) = {phi_fixed:+.4f}", "",
             "| 예측 | 부분집합 | MAE | 평균 상대오차(절대) | 최대 상대오차(절대) | 부호 평균(+ = 과대) | 5% 통과 |", "|---|---|---|---|---|---|---|"]
    subsets = [("all", rows), ("P2G", [r for r in rows if r["variant"] == "G"]), ("P2R", [r for r in rows if r["variant"] == "R"])]
    subsets += [(f"{V//1000}K", [r for r in rows if r["vocab"] == V]) for V in (32000, 64000, 128000)]
    for name in ("m2a_pred", "m2a_pred_first", "m2a_pred_first_p0", "fert_pred_naive", "fert_pred_a", "fert_pred_b", "fert_pred_c"):
        for label, sel in subsets:
            m = summ(name, sel)
            lines.append(f"| {name} | {label} | {m['MAE']:.4f} | {m['mean_rel']:.2%} | {m['max_rel']:.2%} | {m['signed_mean_rel']:+.2%} | {'PASS' if m['pass5'] else 'FAIL'} |")
    lines += ["", "## 점별 상대 오차 (fertility; + = 과대 예측)", "",
              "| 조건 | 어휘 | 관측 | naive | (a) | (b) | (c) | M2a 관측 | 커버리지 | 오차 | 첫접사 커버리지 | 오차 | 첫접사+P0 | 오차 | r_coincide |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["vocab"], r["variant"], r["k"])):
        lines.append(f"| {r['cond']} | {r['vocab']//1000}K | {r['fert_obs']:.3f} | {r['fert_relerr_naive']:+.1%} | {r['fert_relerr_a']:+.1%} | {r['fert_relerr_b']:+.1%} | {r['fert_relerr_c']:+.1%} | {r['m2a_obs']:.3f} | {r['m2a_pred']:.3f} | {r['m2a_relerr']:+.1%} | {r['m2a_pred_first']:.3f} | {r['m2a_relerr_first']:+.1%} | {r['m2a_pred_first_p0']:.3f} | {r['m2a_relerr_first_p0']:+.1%} | {r['r_coincide']:.3f} |")
    (RESULTS / "stage2" / "s3_prediction.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    fig3(rows, "fert_pred_c", "fig3_pred_vs_obs", "Fertility (예측 c: 1 - r_coincide)")
    fig3(rows, "fert_pred_a", "fig3_appendix_pred_a", "Fertility (예측 a: φ 고정, 부록)")
    fig3(rows, "fert_pred_b", "fig3_appendix_pred_b", "Fertility (예측 b: 조건별 φ, 부록)")


def fig3(rows, px_fert, fname, title_fert):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    colors = {32000: "tab:green", 64000: "tab:orange", 128000: "tab:purple"}
    for ax, (px, py, title) in zip(axes, ((px_fert, "fert_obs", title_fert), ("m2a_pred_first_p0", "m2a_obs", "어간 일관성 M2a (예측 = 첫 접사 커버리지 + P0 우연 정렬)"))):
        for r in rows:
            ax.scatter(r[px], r[py], color=colors[r["vocab"]], marker="o" if r["variant"] == "G" else "s", s=40, alpha=0.85)
            if r["vocab"] == 128000:
                ax.annotate(r["cond"], (r[px], r[py]), textcoords="offset points", xytext=(4, 2), fontsize=7)
        lo = min(min(r[px] for r in rows), min(r[py] for r in rows)); hi = max(max(r[px] for r in rows), max(r[py] for r in rows))
        pad = (hi - lo) * 0.05
        xs = [lo - pad, hi + pad]
        ax.plot(xs, xs, color="gray", ls="--", lw=1)
        ax.fill_between(xs, [x * 0.95 for x in xs], [x * 1.05 for x in xs], color="gray", alpha=0.12, lw=0)
        ax.set_xlabel("예측"); ax.set_ylabel("관측"); ax.set_title(title, fontsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", ls="", label=f"{v//1000}K") for v, c in colors.items()]
    handles += [Line2D([], [], color="gray", marker="o", ls="", label="P2G"), Line2D([], [], color="gray", marker="s", ls="", label="P2R")]
    axes[0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{fname}.png", dpi=200); fig.savefig(FIGURES / f"{fname}.pdf")
    plt.close(fig)
    print(fname, "saved")


if __name__ == "__main__":
    main()
