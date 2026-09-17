"""S11-A: direct measurement of vocabulary reallocation, P0 128K vs P2G-30 128K.
(1) P0 combo tokens (stem+particle) absent from the P2G-30 vocab.
(2) P2G-30 new tokens (absent from P0) classified stem_only / suffix_seq / combo / fused / other.
(3) held-out: among eojeol split by P2G-30, those encoded with FEWER tokens than under P0;
    total saved tokens per eojeol (all words), and the subset whose P2G-30 encoding uses >=1 new token.
Compared with table-3 components for P2G-30 128K: fragmentation m*phi, coincidence m*r_co, residual.
Output: results/stage2/s11_realloc.md (+ .json)"""
import sys, json, csv
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, RESULTS, SPLIT_MARK
from pretok import get_condition
from train_tokenizers import load
import stem_consistency as sc
from s2_cost_decomposition import eojeol_byte_spans, per_eojeol_counts
from s4_vocab_analysis import hangul_tokens, classify

S2 = RESULTS / "stage2"
V = 128000
tok0, tokg = load("P0", V), load("P2G-30", V)
h0, hg = hangul_tokens(tok0), hangul_tokens(tokg)
w0 = {w for w, _ in h0.values()}; wg = {w for w, _ in hg.values()}
# (1)
p0_combo = [w for w, _ in h0.values() if classify(w)[0] == "combo"]
combo_dropped = sum(1 for w in p0_combo if w not in wg)
# (2)
new_words = [w for w in wg if w not in w0]
new_cls = Counter(classify(w)[0] for w in new_words)
new_ids = {i for i, (w, sp) in hg.items() if w not in w0}
# (3)
cond = get_condition("P2G-30")
n_words = n_split = n_saved_eoj = n_saved_eoj_new = 0
saved = saved_new = 0
with HELDOUT_TXT.open(encoding="utf-8") as ft, HELDOUT_MECAB.open(encoding="utf-8") as fm:
    for sent, line in zip(ft, fm):
        sent = sent.rstrip("\n"); morphs = json.loads(line)["m"]
        marked = cond.mark(sent, morphs); spans = eojeol_byte_spans(sent); n_words += len(spans)
        e0 = tok0.encode(sent, add_special_tokens=False); eg = tokg.encode(marked, add_special_tokens=False)
        c0 = per_eojeol_counts(e0.tokens, spans); cg = per_eojeol_counts(eg.tokens, spans)
        # ids per eojeol for P2G
        ids_by_eoj = []; pos = 0; e = 0; cur = []
        for t, i in zip(eg.tokens, eg.ids):
            while e < len(spans) - 1 and pos >= spans[e][1]:
                ids_by_eoj.append(cur); cur = []; e += 1
            cur.append(i); pos += len(sc.token_bytes(t))
        ids_by_eoj.append(cur)
        while len(ids_by_eoj) < len(spans): ids_by_eoj.append([])
        for k, me in enumerate(marked.split()):
            if SPLIT_MARK not in me: continue
            n_split += 1
            if cg[k] < c0[k]:
                n_saved_eoj += 1; saved += c0[k] - cg[k]
                if any(i in new_ids for i in ids_by_eoj[k]):
                    n_saved_eoj_new += 1; saved_new += c0[k] - cg[k]
# table-3 components
s2 = {(r["cond"], r["k"], r["variant"], r["vocab"]): r for r in csv.DictReader((S2 / "s2_cost_decomposition.csv").open(encoding="utf-8"))}[("P2", "30", "G", "128000")]
m = float(s2["from_splits"]); frag = float(s2["from_fragmentation"]); rco = json.loads((S2 / "s3_coincide.json").read_text())["P2G-30_128000"]["r_coincide"]
coinc = m * rco; resid = -frag - coinc
out = {"p0_hangul_tokens": len(h0), "p0_combo_tokens": len(p0_combo), "p0_combo_dropped_in_p2g30": combo_dropped,
       "p2g30_new_tokens": len(new_words), "p2g30_new_by_class": dict(new_cls),
       "heldout_words": n_words, "split_eojeol": n_split, "split_eojeol_fewer_tokens_than_P0": n_saved_eoj, "saved_tokens_total": saved,
       "saved_per_word": saved / n_words, "subset_using_new_tokens": n_saved_eoj_new, "saved_new_total": saved_new, "saved_new_per_word": saved_new / n_words,
       "table3_m": m, "table3_m_phi": frag, "coincidence_m_rco": coinc, "residual_realloc_estimate": resid,
       "ratio_saved_to_residual": (saved / n_words) / resid, "ratio_saved_new_to_residual": (saved_new / n_words) / resid}
(S2 / "s11_realloc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
md = ["# S11-A. 어휘 재배분 직접 측정 (P0 128K vs P2G-30 128K)", "",
      f"(1) P0 한글 토큰 {len(h0):,} 중 조합형 {len(p0_combo):,}개; 그중 P2G-30 어휘에 없는 것 {combo_dropped:,}개 ({combo_dropped/len(p0_combo):.1%}).",
      f"(2) P2G-30 신규 토큰(P0에 없음) {len(new_words):,}개: " + ", ".join(f"{k} {v:,}" for k, v in new_cls.most_common()) + ".",
      f"(3) held-out {n_words:,}어절 중 P2G-30이 분리한 어절 {n_split:,}개; 그중 P0보다 토큰이 줄어든 어절 {n_saved_eoj:,}개({n_saved_eoj/n_split:.1%}), 절약 토큰 총 {saved:,}개 = 어절당 {saved/n_words:.4f}. 신규 토큰을 1개 이상 쓰는 부분집합: {n_saved_eoj_new:,}개, 절약 {saved_new:,}개 = 어절당 {saved_new/n_words:.4f}.", "",
      "| 성분 (어절당 토큰) | 값 |", "|---|---|",
      f"| 마크율 m | {m:.4f} |", f"| 파편화 성분 m·φ (표 3) | {frag:+.4f} |", f"| 경계 겹침 성분 m·r_co (r_co={rco:.3f}) | −{coinc:.4f} |",
      f"| 잔여분 = |m·φ| − m·r_co (어휘 재배분 추정) | {resid:.4f} |",
      f"| (3) 직접 측정: 분리 어절의 P0 대비 절약 | {saved/n_words:.4f} ({(saved/n_words)/resid:.0%} of 잔여분) |",
      f"| (3′) 신규 토큰 사용 어절만 | {saved_new/n_words:.4f} ({(saved_new/n_words)/resid:.0%} of 잔여분) |"]
(S2 / "s11_realloc.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
