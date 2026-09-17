"""S11-B: stem table rebuilt from the wiki 10k sentences (rule of 3.3, k-lists unchanged = HPLT),
P2R-30/P2R-100 x 128K retrained on the HPLT train corpus marked with the wiki table (condition P2Rw-k),
evaluated on wiki: M2a, M2b, split rate; gap to P2G compared with the HPLT-table P2R.
Output: data/stem_table_wiki.tsv, results/stage2/s11_wiki_table.md (+csv)"""
import sys, json, csv
from collections import Counter
from pathlib import Path
import regex as re
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, RESULTS, is_particle_tag
from stem_consistency import gold_pairs
import stem_consistency as sc
from build_stem_table import load_stem_table
import make_marked_corpora, train_tokenizers
from metrics import fertility_and_m3
from pretok import get_condition, P2R

S2 = RESULTS / "stage2"; V = 128000
WTXT, WMEC = DATA / "ood_wiki_10k.txt", DATA / "ood_wiki_10k.mecab.jsonl"
WTAB = DATA / "stem_table_wiki.tsv"
H = re.compile(r"^\p{Hangul}+$")

if not WTAB.exists():
    cnt = Counter()
    with WTXT.open(encoding="utf-8") as ft, WMEC.open(encoding="utf-8") as fm:
        for sent, line in zip(ft, fm):
            sent = sent.rstrip("\n"); morphs = json.loads(line)["m"]
            for eoj, stem, suf in gold_pairs(sent, morphs): cnt[stem] += 1
            pos = 0
            for eoj in sent.split():
                st = sent.index(eoj, pos); en = st + len(eoj); pos = en
                if not H.match(eoj): continue
                ms = [m for m in morphs if m[2] >= st and m[3] <= en]
                if ms and not any(is_particle_tag(t) or "+" in t for _, t, _, _ in ms): cnt[eoj] += 1
    with WTAB.open("w", encoding="utf-8") as f:
        for s, c in cnt.most_common(): f.write(f"{s}\t{c}\n")
wtab = load_stem_table(WTAB)
stats = {"wiki_stem_types": len(wtab), "wiki_stem_types_ge5": sum(1 for c in wtab.values() if c >= 5)}
print(stats, flush=True)

def cond_w(k):
    c = P2R(k, stem_table=wtab); c.name = f"P2Rw-{k}"; return c

# mark + train
conds = {k: cond_w(k) for k in (30, 100)}
for k, c in conds.items():
    out = DATA / f"train_1M.P2Rw-{k}.txt"
    if not out.exists():
        from config import TRAIN_TXT, SPLIT_MARK
        with TRAIN_TXT.open(encoding="utf-8") as ft, out.open("w", encoding="utf-8") as fo:
            for line in ft: fo.write(c.mark(line.rstrip("\n")) + "\n")
        print("marked", out.name, flush=True)
    if not (train_tokenizers.TOK_DIR / f"P2Rw-{k}_{V}" / "tokenizer.json").exists():
        train_tokenizers.train(f"P2Rw-{k}", V); print("trained", k, flush=True)

# evaluate on wiki
ood = {(r["cond"], r["k"], r["variant"]): r for r in csv.DictReader((S2 / "s5_ood.csv").open(encoding="utf-8")) if r["vocab"] == "128000"}
rows = []; md = ["# S11-B. 위키 어간 테이블로 재학습한 P2R (128K, 위키 10,000문장 평가)", "",
                 f"위키 어간 테이블: {stats['wiki_stem_types']:,}종(5회 이상 {stats['wiki_stem_types_ge5']:,}종) vs HPLT 테이블 370,878종(76,535종). k 목록은 HPLT 그대로. 테이블은 평가 문장 자체에서 만들어져 in-sample임.", "",
                 "| k | 조건 | fertility | M2a | M2b | 분리 어절 비율 | P2G 대비 M2a 격차 | M2b 격차 |", "|---|---|---|---|---|---|---|---|"]
for k, c in conds.items():
    tok = train_tokenizers.load(f"P2Rw-{k}", V)
    r1 = fertility_and_m3(tok, c, txt=WTXT, mecab=WMEC); r2 = sc.run(tok, c, heldout_txt=WTXT, heldout_mecab=WMEC)
    g = ood[("P2", str(k), "G")]; rr = ood[("P2", str(k), "R")]
    for name, f, a, b, sr in ((f"P2G-{k}", float(g["fertility"]), float(g["m2a"]), float(g["m2b"]), float(g["split_rate"])),
                              (f"P2R-{k} (HPLT 테이블)", float(rr["fertility"]), float(rr["m2a"]), float(rr["m2b"]), float(rr["split_rate"])),
                              (f"P2Rw-{k} (위키 테이블)", r1["fertility"], r2["m2a"], r2["m2b"], r1["split_rate"])):
        ga = (float(g["m2a"]) - a) * 100; gb = (float(g["m2b"]) - b) * 100
        rows.append({"k": k, "cond": name, "fertility": round(f, 4), "m2a": round(a, 4), "m2b": round(b, 4), "split_rate": round(sr, 4), "gap_m2a": round(ga, 2), "gap_m2b": round(gb, 2)})
        md.append(f"| {k} | {name} | {f:.3f} | {a*100:.1f} | {b*100:.1f} | {sr*100:.1f}% | {ga:.1f} | {gb:.1f} |")
with (S2 / "s11_wiki_table.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(S2 / "s11_wiki_table.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
