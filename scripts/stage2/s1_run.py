"""S1: metrics for 8 conditions x {32K, 64K, 128K} -> results/stage2/metrics_all_vocab.csv
and P2Rnb-{10,30,100} (rule b off, 128K) -> results/stage2/metrics_ruleb_off.csv.
Existing results/metrics.csv is not touched. Prints the delta-fertility table."""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS
from metrics import evaluate, COLS

OUT = RESULTS / "stage2"
COLS2 = COLS + ["split_rate", "m3_nonascii"]
CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]


def write(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS2)
        w.writeheader(); w.writerows(rows)


def merge(row, extra):
    return {**row, "split_rate": extra["split_rate"], "m3_nonascii": extra["m3_nonascii"]}


def main(which="all"):
    OUT.mkdir(exist_ok=True)
    if which in ("all", "vocab"):
        rows = []
        for v in (32000, 64000, 128000):
            for c in CONDS:
                r, e = evaluate(c, v)
                rows.append(merge(r, e))
                write(rows, OUT / "metrics_all_vocab.csv")
        delta_table(rows)
    if which in ("all", "ruleb"):
        rows = []
        for c in ("P2Rnb-10", "P2Rnb-30", "P2Rnb-100"):
            r, e = evaluate(c, 128000)
            rows.append(merge(r, e))
        write(rows, OUT / "metrics_ruleb_off.csv")


def delta_table(rows):
    R = {(r["cond"], int(r["k"]), r["variant"], int(r["vocab"])): r for r in rows}
    print("\nΔfertility = fertility(P2-k) - fertility(P0), by vocab")
    print(f"{'cond':10s} " + " ".join(f"{v//1000:>3d}K(P0={R[('P0',0,'-',v)]['fertility']:.3f})" for v in (32000, 64000, 128000)))
    for v_ in ("G", "R"):
        for k in (10, 30, 100):
            cells = []
            for v in (32000, 64000, 128000):
                d = R[("P2", k, v_, v)]["fertility"] - R[("P0", 0, "-", v)]["fertility"]
                cells.append(f"{d:+.3f} ({d / R[('P0', 0, '-', v)]['fertility']:+.1%})")
            print(f"P2{v_}-{k:<6d} " + "  ".join(cells))
    cells = [f"{R[('P1',0,'-',v)]['fertility'] - R[('P0',0,'-',v)]['fertility']:+.3f}" for v in (32000, 64000, 128000)]
    print(f"{'P1':10s} " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
