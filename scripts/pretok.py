"""4.4 Pre-tokenization conditions.

Design (fixed 2026-09-07, see README): every condition is realised by inserting a
private-use char SPLIT_MARK (U+E000) at extra split points *inside* eojeol. The HF
tokenizer then uses  Split(SPLIT_MARK, removed) -> ByteLevel(GPT-2 regex)  for all
conditions, so the regex and BPE algorithm are identical; only the marks differ.

Common interface:
    cond.mark(sentence, morphs) -> marked sentence     (morphs = Mecab cache row, may be None for P0/P2R)
    cond.split(sentence, morphs) -> list[str]          (debug view: pieces)

morphs: list of [surface, pos, start, end] with CHARACTER offsets into `sentence`.
"""
import sys
from pathlib import Path
import regex as re

sys.path.insert(0, str(Path(__file__).parent))
from config import SPLIT_MARK, TAG_SET_T, DATA, SEED_DIR, is_particle_tag

HANGUL_RUN = re.compile(r"\p{Hangul}+")


def _insert(sentence: str, cuts):
    """Insert SPLIT_MARK before each char offset in `cuts`."""
    if not cuts:
        return sentence
    out, prev = [], 0
    for c in sorted(set(cuts)):
        out.append(sentence[prev:c])
        out.append(SPLIT_MARK)
        prev = c
    out.append(sentence[prev:])
    return "".join(out)


def _interior(sentence, pos):
    """A cut is valid only strictly inside a non-space run (never at eojeol edges)."""
    return 0 < pos < len(sentence) and not sentence[pos - 1].isspace() and not sentence[pos].isspace()


class P0:
    name = "P0"
    needs_mecab = False

    def cuts(self, sentence, morphs):
        return []

    def mark(self, sentence, morphs=None):
        return _insert(sentence, self.cuts(sentence, morphs))

    def split(self, sentence, morphs=None):
        return self.mark(sentence, morphs).replace(SPLIT_MARK, " | ").split()


class P1(P0):
    """Full separation: cut at every Mecab morpheme boundary inside an eojeol."""
    name = "P1"
    needs_mecab = True

    def cuts(self, sentence, morphs):
        return [st for _, _, st, _ in morphs if _interior(sentence, st)]


class P2G(P0):
    """Analyzer-guided partial separation: cut before a morpheme iff it is a
    particle/ending (all '+'-joined tags in T) AND its surface is in the top-k list."""
    needs_mecab = True

    def __init__(self, k):
        self.k = k
        self.name = f"P2G-{k}"
        self.plist = set((SEED_DIR / f"particles_k{k}.txt").read_text(encoding="utf-8").split())

    def cuts(self, sentence, morphs):
        return [st for surf, pos, st, _ in morphs
                if is_particle_tag(pos) and surf in self.plist and _interior(sentence, st)]


class P2R(P0):
    """Rule-based partial separation (no analyzer at inference).

    For each Hangul run of length n, consider every stem length j (1 <= j < n) such
    that run[j:] can be tiled by surface forms from the top-k list (DP, right to left).
    Choose the LONGEST such stem (= least splitting) that passes
      (a) stem >= 1 syllable  (j >= 1), and
      (b) stem observed in TRAIN as a Mecab gold stem or standalone eojeol >= min_f times
          (stem table; built once with the analyzer, analyzer-free at inference).
    Then cut at j and at the greedy longest-first tiling boundaries inside run[j:].
    With use_rule_b=False the SHORTEST stem is chosen (maximal peeling, naive baseline)."""
    needs_mecab = False

    def __init__(self, k, stem_table=None, min_f=5, use_rule_b=True):
        self.k = k
        self.name = f"P2R-{k}"
        plist = (SEED_DIR / f"particles_k{k}.txt").read_text(encoding="utf-8").split()
        self.plist_set = set(plist)
        self.maxlen = max(map(len, plist))
        self.min_f = min_f
        self.use_rule_b = use_rule_b
        if use_rule_b and stem_table is None:
            raise ValueError("P2R with rule (b) needs stem_table (data/stem_table.tsv)")
        self.stem_table = stem_table
        self._cache = {}

    def _run_cuts(self, run):
        if run in self._cache:
            return self._cache[run]
        n = len(run)
        tile = [False] * (n + 1)      # tile[j]: run[j:] is a concatenation of list items
        tile[n] = True
        for j in range(n - 1, -1, -1):
            tile[j] = any(tile[j + L] for L in range(1, min(self.maxlen, n - j) + 1) if run[j:j + L] in self.plist_set)
        cands = [j for j in range(1, n) if tile[j]]
        if self.use_rule_b:
            cands = [j for j in cands if self.stem_table.get(run[:j], 0) >= self.min_f]
            j = max(cands) if cands else None
        else:
            j = min(cands) if cands else None
        cuts = []
        if j is not None:
            pos = j
            while pos < n:
                cuts.append(pos)
                for L in range(min(self.maxlen, n - pos), 0, -1):
                    if run[pos:pos + L] in self.plist_set and tile[pos + L]:
                        pos += L
                        break
        if len(self._cache) > 500_000:
            self._cache.clear()
        self._cache[run] = cuts
        return cuts

    def cuts(self, sentence, morphs=None):
        out = []
        for m in HANGUL_RUN.finditer(sentence):
            out.extend(m.start() + c for c in self._run_cuts(m.group()))
        return out


def get_condition(name: str, **kw):
    if name == "P0":
        return P0()
    if name == "P1":
        return P1()
    fam, _, k = name.partition("-")
    if fam == "P2G":
        return P2G(int(k))
    if fam == "P2R":
        return P2R(int(k), **kw)
    if fam == "P2Rp":  # P2R+ : same rule, stem table with fused past-tense stems (post-hoc, see build_stem_table_plus.py)
        c = P2R(int(k), **kw)
        c.name = f"P2Rp-{k}"
        return c
    if fam == "P2Rnb":  # rule (b) off (appendix)
        kw = {k_: v_ for k_, v_ in kw.items() if k_ != "stem_table"}
        c = P2R(int(k), stem_table=None, use_rule_b=False, **kw)
        c.name = f"P2Rnb-{k}"
        return c
    raise ValueError(name)


def parse_cond(name: str):
    """'P2G-30' -> ('P2', 30, 'G'); 'P2Rnb-30' -> ('P2', 30, 'Rnb'); 'P0' -> ('P0', 0, '-')."""
    if name in ("P0", "P1"):
        return name, 0, "-"
    fam, _, k = name.partition("-")
    return "P2", int(k), fam[2:]
