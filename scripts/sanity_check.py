"""4.9 Sanity checks (run BEFORE interpreting results).
Usage: python scripts/sanity_check.py [csv_path]   (default results/metrics.csv; stage2 CSV has m3_nonascii)"""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS

NUM = ("fertility", "m2a", "m2b", "m3", "m3_nonascii", "split_rate")


def load(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    return {(r["cond"], int(r["k"]), r["variant"], int(r["vocab"])): {k: (float(v) if k in NUM else v) for k, v in r.items()} for r in rows}


def main(path=RESULTS / "metrics.csv"):
    R = load(path)
    vocabs = sorted({key[3] for key in R})
    results = []
    for v in vocabs:
        p0, p1 = R[("P0", 0, "-", v)], R[("P1", 0, "-", v)]
        f0 = p0["fertility"]
        tag = f"[{v//1000}K]"
        if v == 128000:
            results.append((f"{tag} P0 fertility same order as Thunder-Tok BPE 128K HPLT (~1.70)", 1.2 <= f0 <= 2.2, f"{f0:.3f}"))
        results.append((f"{tag} P1 fertility clearly > P0", p1["fertility"] > f0 * 1.05, f"{p1['fertility']:.3f} vs {f0:.3f}"))
        for var in ("G", "R"):
            ks = [10, 30, 100]
            fs = [R[("P2", k, var, v)]["fertility"] for k in ks]
            m2 = [R[("P2", k, var, v)]["m2a"] for k in ks]
            results.append((f"{tag} P2{var}: fertility monotone in k", all(a <= b for a, b in zip(fs, fs[1:])), " ".join(f"{x:.3f}" for x in fs)))
            results.append((f"{tag} P2{var}: fertility between P0 and P1", all(f0 <= x <= p1["fertility"] for x in fs), ""))
            results.append((f"{tag} P2{var}: M2a monotone in k (else suspect mis-segmentation)", all(a <= b for a, b in zip(m2, m2[1:])), " ".join(f"{x:.3f}" for x in m2)))
        results.append((f"{tag} P1 M2b highest (structural)", p1["m2b"] >= max(r["m2b"] for key, r in R.items() if key[3] == v) - 1e-9, f"{p1['m2b']:.3f}"))
    # across vocab sizes
    if len(vocabs) >= 2:
        for key in sorted({(c, k, var) for c, k, var, _ in R}):
            for col in ("m3_nonascii",):  # ID<256 M3 dropped (definitional artefact; footnote in paper)
                if col not in R[(*key, vocabs[0])]:
                    continue
                vals = [R[(*key, v)][col] for v in vocabs]
                name = key[0] if key[0] != "P2" else f"P2{key[2]}-{key[1]}"
                results.append((f"{name}: {col} decreases with vocab size", all(a > b for a, b in zip(vals, vals[1:])), " ".join(f"{x:.5f}" for x in vals)))
            vals = [R[(*key, v)]["fertility"] for v in vocabs]
            name = key[0] if key[0] != "P2" else f"P2{key[2]}-{key[1]}"
            results.append((f"{name}: fertility decreases with vocab size", all(a > b for a, b in zip(vals, vals[1:])), " ".join(f"{x:.3f}" for x in vals)))
    else:
        results.append(("M3 decreases with vocab size", None, f"vocabs={vocabs}"))
    ok_all = True
    n_pass = n_fail = 0
    for name, ok, detail in results:
        flag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        if ok is False:
            ok_all = False; n_fail += 1
        elif ok:
            n_pass += 1
        print(f"[{flag}] {name}  {detail}")
    print(f"{n_pass} PASS, {n_fail} FAIL -> " + ("ALL PASS" if ok_all else "SOME CHECKS FAILED"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RESULTS / "metrics.csv")
