"""S6 side-by-side table: P2R vs P2R+ vs P2G (128K) on HPLT held-out and wiki OOD,
with cost decomposition and error types. -> results/stage2/s6_p2r_vs_p2rp.md"""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS
S = RESULTS / "stage2"


def rd(name):
    return list(csv.DictReader((S / name).open(encoding="utf-8")))


def key(r):
    return (r["cond"], int(r["k"]), r["variant"], int(r["vocab"]))


met = {key(r): r for r in rd("metrics_all_vocab.csv")}; met.update({key(r): r for r in rd("s6_p2rp_metrics.csv")})
ood = {key(r): r for r in rd("s5_ood.csv")}; ood.update({key(r): r for r in rd("s6_p2rp_ood.csv")})
cost = {key(r): r for r in rd("s2_cost_decomposition.csv")}; cost.update({key(r): r for r in rd("s6_p2rp_cost.csv")})
err = {(r["test"], r["eval_set"]): r for r in rd("s6_error_types.csv")}
V = 128000
lines = ["# S6. P2R vs P2R+ vs P2G (128K) — 어간 테이블에 융합 과거형 어간을 등재한 사후 실험", ""]
for k in (30, 100):
    lines += [f"## k = {k}", "",
              "| 지표 | P2G (상한) | P2R | P2R+ | P2R+ − P2R |", "|---|---|---|---|---|"]
    def row(label, get, fmt, signed=True):
        g, r, p = get(("P2", k, "G", V)), get(("P2", k, "R", V)), get(("P2", k, "Rp", V))
        d = p - r
        lines.append(f"| {label} | {fmt(g)} | {fmt(r)} | {fmt(p)} | {('+' if d >= 0 else '')}{fmt(d)} |")
    pct = lambda x: f"{x*100:.1f}%"; f3 = lambda x: f"{x:.3f}"
    for setname, src in (("HPLT", met), ("위키", ood)):
        row(f"fertility ({setname})", lambda kk: float(src[kk]["fertility"]), f3)
        row(f"M2a ({setname})", lambda kk: float(src[kk]["m2a"]), pct)
        row(f"M2b ({setname})", lambda kk: float(src[kk]["m2b"]), pct)
        row(f"분리 어절 비율 ({setname})", lambda kk: float(src[kk]["split_rate"]), pct)
    row("Δfert vs P0 (HPLT)", lambda kk: float(cost[kk]["delta_fert"]), f3)
    row("마크/어절 (HPLT)", lambda kk: float(cost[kk]["from_splits"]), f3)
    row("φ (HPLT)", lambda kk: float(cost[kk]["from_fragmentation"]) / float(cost[kk]["from_splits"]), lambda x: f"{x:+.2f}")
    row("어간 1토큰 비율 (HPLT)", lambda kk: float(cost[kk]["stem_single_token_rate"]), pct)
    lines += ["", "| 오류 유형 (기준 P2G) | P2R HPLT | P2R+ HPLT | P2R 위키 | P2R+ 위키 |", "|---|---|---|---|---|"]
    for lab, col in (("미분할", "under_rate"), ("과분할", "over_rate"), ("경계 불일치", "diff_rate")):
        cells = [f"{float(err[(t, s)][col])*100:.2f}%" for s in ("hplt_heldout_100k", "wiki_ood_10k") for t in (f"P2R-{k}", f"P2Rp-{k}")]
        lines.append(f"| {lab} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")
    # gap to P2G
    for setname, src in (("HPLT", met), ("위키", ood)):
        g = src[("P2", k, "G", V)]
        gr = [(float(g[m]) - float(src[("P2", k, v, V)][m])) * 100 for v in ("R", "Rp") for m in ("m2a", "m2b")]
        lines.append(f"- P2G 대비 격차 ({setname}): M2a P2R {gr[0]:.1f}점 → P2R+ {gr[2]:.1f}점; M2b P2R {gr[1]:.1f}점 → P2R+ {gr[3]:.1f}점")
    lines.append("")
(S / "s6_p2r_vs_p2rp.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
