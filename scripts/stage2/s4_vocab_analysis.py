"""S4: what fertility cannot see — vocabulary composition (128K).

(1) For P0 / P2G-30 / P2R-30: classify every Hangul-only token (leading space stripped) with Mecab:
      combo      = >=1 non-particle morpheme(s) then >=1 particle/ending morpheme (학교에서, 학교를)
      stem_only  = no particle/ending morpheme (학교, 대한민국)
      suffix_seq = all morphemes particle/ending (에서는, 습니다)
      fused      = contains a '+' compound tag (샀, 했다)
    combo tokens grouped by stem -> #variants per stem; top-20 stems.
(2) Tokens new in P2G-30 / P2R-30 vs P0: class distribution + top-100 by token id (merge order ~ frequency).
(3) Contamination: Hangul tokens in P2R-k but not in P2G-k (k=10,30,100), classified:
      known_stem      = in stem_table (>=5) ; suffix_seq ; single_morph (Mecab gives exactly one morpheme)
      unknown_fragment = none of the above  <- candidate mis-segmentation debris (예: 사과 -> 사)
    Control: same for P2G-k-only tokens (vocabulary noise baseline).
Outputs: results/stage2/s4_vocab_composition.csv, s4_vocab_contamination.md, s4_new_tokens_top100.md
"""
import sys, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import regex as re
import mecab
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS, DATA, K_LIST, is_particle_tag
from train_tokenizers import load
from build_stem_table import load_stem_table

OUT = RESULTS / "stage2"
HANGUL_ONLY = re.compile(r"^\p{Hangul}+$")
tagger = mecab.MeCab()
_cache = {}


def analyze(word):
    if word not in _cache:
        _cache[word] = [(m.surface, m.feature.pos) for m in tagger.parse(word)]
    return _cache[word]


def classify(word):
    ms = analyze(word)
    if any("+" in t for _, t in ms):
        return "fused", None
    flags = [is_particle_tag(t) for _, t in ms]
    if not any(flags):
        return "stem_only", None
    if all(flags):
        return "suffix_seq", None
    first = flags.index(True)
    if all(flags[first:]):
        return "combo", "".join(s for s, _ in ms[:first])
    return "other", None


def hangul_tokens(tok):
    """id -> word (Hangul-only after stripping the leading space marker Ġ)."""
    out = {}
    for t, i in tok.get_vocab().items():
        s = tok.decode([i])
        w = s[1:] if s.startswith(" ") else s
        if w and HANGUL_ONLY.match(w) and (len(s) == len(w) or s.startswith(" ")):
            out[i] = (w, s.startswith(" "))
    return out


def composition(name, toks):
    cls = Counter(); stems = defaultdict(set)
    for i, (w, sp) in toks.items():
        c, stem = classify(w)
        cls[c] += 1
        if c == "combo":
            stems[stem].add(w)
    n = len(toks)
    rows = [{"cond": name, "n_hangul_tokens": n, **{f"n_{k}": cls[k] for k in ("combo", "stem_only", "suffix_seq", "fused", "other")},
             "combo_share": round(cls["combo"] / n, 4), "n_combo_stems": len(stems),
             "variants_per_stem_mean": round(sum(map(len, stems.values())) / max(1, len(stems)), 3)}]
    top = sorted(stems.items(), key=lambda kv: -len(kv[1]))[:20]
    return rows, top, cls


def main(vocab=128000):
    OUT.mkdir(exist_ok=True)
    st = load_stem_table()
    toks = {c: hangul_tokens(load(c, vocab)) for c in ["P0", "P2G-30", "P2R-30"]}
    comp_rows, md = [], [f"# S4 어휘 분석 (128K)", ""]
    for c in ["P0", "P2G-30", "P2R-30"]:
        rows, top, cls = composition(c, toks[c])
        comp_rows += rows
        md.append(f"## {c}: 한글 토큰 {rows[0]['n_hangul_tokens']:,} — 조합형 {cls['combo']:,} ({rows[0]['combo_share']:.1%}), 어간형 {cls['stem_only']:,}, 접사열 {cls['suffix_seq']:,}, 융합 {cls['fused']:,}")
        md.append("| 어간 | 변형 수 | 예 |"); md.append("|---|---|---|")
        for stem, vs in top:
            md.append(f"| {stem} | {len(vs)} | {', '.join(sorted(vs, key=len)[:8])} |")
        md.append("")
    with (OUT / "s4_vocab_composition.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys())); w.writeheader(); w.writerows(comp_rows)

    # (2) new tokens vs P0
    p0_words = {w for w, _ in toks["P0"].values()}
    new_md = ["# S4 신규 진입 토큰 (P0 128K 어휘에 없는 한글 토큰, id 오름차순 = 병합 순서)", ""]
    for c in ["P2G-30", "P2R-30"]:
        new = sorted((i, w) for i, (w, sp) in toks[c].items() if w not in p0_words)
        cls = Counter(classify(w)[0] for _, w in new)
        lens = Counter(len(w) for _, w in new)
        new_md.append(f"## {c}: 신규 {len(new):,}개 — " + ", ".join(f"{k} {v:,}" for k, v in cls.most_common()) +
                      f"; 길이 분포 " + ", ".join(f"{L}자:{n}" for L, n in sorted(lens.items())[:8]))
        new_md.append(", ".join(f"{w}[{classify(w)[0][:2]}]" for _, w in new[:100])); new_md.append("")
        comp_rows_new = {"cond": c, "n_new": len(new), **{f"new_{k}": v for k, v in cls.items()}}
        md.append(f"- {c} 신규 토큰 {len(new):,}: " + ", ".join(f"{k} {v:,}" for k, v in cls.most_common()))
    (OUT / "s4_new_tokens_top100.md").write_text("\n".join(new_md), encoding="utf-8")

    # (3) contamination
    cont = ["# S4 어휘 오염: P2R-k에만 있고 P2G-k에는 없는 한글 토큰 (128K)", "",
            "분류(Mecab 태깅 기반): known_stem = 학습 코퍼스 골드 어간·단독 어절 ≥5회; combo_unsplit = 실질형태소+조사·어미 조합(한쪽 조건이 자르지 않아 남은 조각); fused = 융합형(VV+EP 등); suffix_seq = 조사·어미열; stem_frag = 어간형으로 분석되지만 학습 코퍼스 골드 어간·단독 어절로 5회 미만(오분할 파편 후보); 대조군 = P2G-k에만 있는 토큰.", "",
            "| k | 방향 | 고유 토큰 | known_stem | combo_unsplit(어간+조사 미분리) | fused | suffix_seq | stem_frag(어간형 미등재 파편) | other |", "|---|---|---|---|---|---|---|---|---|"]
    examples = []
    for k in K_LIST:
        tg = hangul_tokens(load(f"P2G-{k}", vocab)); tr = hangul_tokens(load(f"P2R-{k}", vocab))
        wg = {w for w, _ in tg.values()}; wr = {w for w, _ in tr.values()}
        for label, only in (("P2R only", wr - wg), ("P2G only (대조)", wg - wr)):
            cls = Counter(); ex = defaultdict(list)
            for w in sorted(only, key=len):
                c0 = classify(w)[0]
                if st.get(w, 0) >= 5:
                    c = "known_stem"
                elif c0 == "combo":
                    c = "combo_unsplit"          # stem+particle piece that P2R failed to split (rule b) / P2G split
                elif c0 == "fused":
                    c = "fused"
                elif c0 == "suffix_seq":
                    c = "suffix_seq"
                elif c0 == "stem_only":
                    c = "stem_frag"             # Mecab-analysable stem-like piece not attested as a stem (over-split debris candidate)
                else:
                    c = "other"
                cls[c] += 1
                if len(ex[c]) < 25:
                    ex[c].append(w)
            n = len(only)
            cont.append(f"| {k} | {label} | {n:,} | {cls['known_stem']:,} | {cls['combo_unsplit']:,} ({cls['combo_unsplit']/n:.1%}) | {cls['fused']:,} | {cls['suffix_seq']:,} | {cls['stem_frag']:,} ({cls['stem_frag']/n:.1%}) | {cls['other']:,} |")
            for c in ("combo_unsplit", "stem_frag", "fused"):
                examples.append(f"- k={k} {label} {c} 예: " + ", ".join(ex[c]))
    cont += ["", "## 예시"] + examples
    (OUT / "s4_vocab_contamination.md").write_text("\n".join(cont), encoding="utf-8")
    (OUT / "s4_vocab_composition.md").write_text("\n".join(md), encoding="utf-8")
    md.append(""); md += cont
    print("\n".join(md))


if __name__ == "__main__":
    main()
