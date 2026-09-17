"""Qualitative segmentation table: results/qualitative.md (conditions x example eojeol groups)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS, K_LIST
from pretok import get_condition
from train_tokenizers import load
from build_stem_table import load_stem_table
import mecab

CONDS = ["P0", "P1"] + [f"P2G-{k}" for k in K_LIST] + [f"P2R-{k}" for k in K_LIST]
GROUPS = [
    # (1) regular noun + particles: the intended case
    ["학교", "학교에서", "학교에서는", "학교에서만큼은"],
    # (2) verb stem + endings
    ["먹다", "먹었다", "먹는", "먹었겠지만"],
    # (3) P2R under-split: fused past-tense stems (했=하+았) are absent from the stem table
    ["했다", "말했다", "밝혔다", "위해서는"],
    # (4) P2R over-split: high-frequency adverbs/conjunctions ending in a listed form
    ["또는", "그리고", "많이", "바로"],
    # (5) boundary disagreement (것으|로 vs 것|으로) and the classic over-split 사|과
    ["것으로", "있다고", "사과", "사과를"],
]


def main(vocab=128000):
    tagger = mecab.MeCab()
    st = load_stem_table()
    conds = {c: get_condition(c, stem_table=st) if c.startswith("P2R") else get_condition(c) for c in CONDS}
    toks = {c: load(c, vocab) for c in CONDS}
    lines = [f"# 정성 예시 (어휘 {vocab}) — 문장 중간 위치(앞 공백 포함) 기준 토큰 분절, `|` = 토큰 경계, `‸` = 사전 토큰화 마크", ""]
    for group in GROUPS:
        lines.append("| 조건 | " + " | ".join(group) + " |")
        lines.append("|---|" + "---|" * len(group))
        for c in CONDS:
            cells = []
            for w in group:
                morphs = [[m.surface, m.feature.pos, m.span.start, m.span.end] for m in tagger.parse(w)]
                marked = conds[c].mark(w, morphs)
                enc = toks[c].encode(" " + marked, add_special_tokens=False)
                pieces = [toks[c].decode([i]).replace(" ", "␣") for i in enc.ids]
                cells.append("|".join(pieces) + f" ({len(enc.ids)})")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "qualitative.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 128000)
