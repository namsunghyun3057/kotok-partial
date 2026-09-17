"""Table 3: cost decomposition with phi by vocab. -> results/stage2/table3_cost_decomposition.md"""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS

rows = list(csv.DictReader((RESULTS / "stage2" / "s2_cost_decomposition.csv").open(encoding="utf-8")))
R = {(r["cond"], int(r["k"]), r["variant"], int(r["vocab"])): r for r in rows}
V = (32000, 64000, 128000)


def name(c, k, v):
    return c if c != "P2" else f"P2{v}-{k}"


lines = ["# 표 3. 분리 비용 분해 (Δfertility = 마크율 × (1+φ); φ = 파편화 성분 / 강제 분리 성분)", "",
         "| 조건 | 분리 어절 비율 | 마크/어절 | Δfert 32K | Δfert 64K | Δfert 128K | φ 32K | φ 64K | φ 128K | 어간 1토큰 128K |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for key in [("P2", 10, "G"), ("P2", 30, "G"), ("P2", 100, "G"), ("P2", 10, "R"), ("P2", 30, "R"), ("P2", 100, "R"), ("P1", 0, "-")]:
    rs = [R[(*key, v)] for v in V]
    phi = [float(r["from_fragmentation"]) / float(r["from_splits"]) for r in rs]
    cells = [name(*key), f"{float(rs[0]['split_rate']):.1%}", f"{float(rs[0]['from_splits']):.3f}"]
    cells += [f"+{float(r['delta_fert']):.3f}" for r in rs]
    cells += [f"{x:+.2f}" for x in phi]
    cells += [f"{float(rs[2]['stem_single_token_rate']):.1%}"]
    lines.append("| " + " | ".join(cells) + " |")
out = RESULTS / "stage2" / "table3_cost_decomposition.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
