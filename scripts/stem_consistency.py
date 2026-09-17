"""4.6 M2 stem consistency (M2a) and boundary alignment (M2b) on the held-out set.

Gold (from Mecab cache): for each eojeol consisting only of Hangul, the gold
stem/suffix boundary is the start of the FIRST morpheme whose tag is entirely in
TAG_SET_T (particle/ending). Requirements: stem non-empty, suffix non-empty,
no compound-tag morpheme fusing stem+ending (e.g. 'VV+EC') before the boundary.

Tokenization unit: " " + eojeol  (as it appears after a space inside a sentence;
the GPT-2 regex makes eojeol tokenization context-free apart from the leading space).
The stem char range therefore is [0, 1+len(stem)) including the leading space.

Boundary alignment: byte-cumulative. Each token is mapped back to its UTF-8 bytes
via the ByteLevel alphabet; the boundary is aligned iff some token ends exactly at
len((" "+stem).encode()). This is independent of HF offset semantics and treats
any token straddling the boundary (including partial-character byte tokens) as
misaligned. HF char offsets are cross-checked on a sample (see --check).

M2a: over stems with >=2 distinct eojeol forms and >=5 occurrences, the share of
occurrences whose stem token-id sequence equals the stem's modal sequence;
misaligned occurrences never match. Weighted by occurrences.
M2b: share of eligible eojeol occurrences (all, not only filtered stems) whose gold
boundary coincides with a token boundary.
"""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
import regex as re
from tokenizers import Tokenizer, pre_tokenizers

sys.path.insert(0, str(Path(__file__).parent))
from config import HELDOUT_TXT, HELDOUT_MECAB, SPLIT_MARK, is_particle_tag

HANGUL_ONLY = re.compile(r"^\p{Hangul}+$")

# ByteLevel unicode-char -> byte mapping (inverse of the GPT-2 bytes_to_unicode table)
_alphabet = pre_tokenizers.ByteLevel.alphabet()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
U2B = _bytes_to_unicode()
assert set(U2B) == set(_alphabet), "ByteLevel alphabet mismatch"


def token_bytes(tok_str: str) -> bytes:
    return bytes(U2B[ch] for ch in tok_str)


def gold_pairs(sentence: str, morphs):
    """Yield (eojeol, stem, suffix) for eligible eojeol in the sentence."""
    for _, eoj, stem, suf in gold_pairs_idx(sentence, morphs):
        yield eoj, stem, suf


def gold_pairs_idx(sentence: str, morphs):
    """Yield (eojeol_index, eojeol, stem, suffix). Index = position in sentence.split()."""
    pos = 0
    for idx, eoj in enumerate(sentence.split()):
        st = sentence.index(eoj, pos)
        en = st + len(eoj)
        pos = en
        if not HANGUL_ONLY.match(eoj):
            continue
        ms = [m for m in morphs if m[2] >= st and m[3] <= en]
        if not ms or ms[0][2] != st or ms[-1][3] != en:
            continue
        # contiguity check
        ok = all(ms[i][3] == ms[i + 1][2] for i in range(len(ms) - 1))
        if not ok:
            continue
        b = None
        fused = False
        for surf, tag, s, e in ms:
            if is_particle_tag(tag):
                b = s
                break
            if "+" in tag and any(t in {"EP", "EF", "EC", "ETN", "ETM"} or t.startswith("J") for t in tag.split("+")):
                fused = True  # e.g. VV+EC: stem and ending fused in one surface form
                break
        if fused or b is None or b == st or b == en:
            continue
        yield idx, eoj, sentence[st:b], sentence[b:en]


class StemScorer:
    def __init__(self, tokenizer: Tokenizer, cond, morph_lookup=None):
        """cond: pretok condition object (mark(sentence, morphs)).
        For P1/P2G the marks for an eojeol depend on its Mecab analysis, so the
        marked form is produced from the sentence-level analysis by the caller."""
        self.tok = tokenizer
        self.cond = cond
        self.cache = {}  # marked " "+eojeol -> (ids, tok_strs)

    def encode_eojeol(self, marked_eojeol: str):
        key = marked_eojeol
        if key not in self.cache:
            enc = self.tok.encode(" " + marked_eojeol, add_special_tokens=False)
            self.cache[key] = (tuple(enc.ids), tuple(enc.tokens))
        return self.cache[key]

    @staticmethod
    def stem_ids(ids, toks, stem_nbytes):
        """Return (aligned: bool, stem_id_seq or None)."""
        cum = 0
        for i, t in enumerate(toks):
            cum += len(token_bytes(t))
            if cum == stem_nbytes:
                return True, ids[: i + 1]
            if cum > stem_nbytes:
                return False, None
        return False, None


def run(tokenizer: Tokenizer, cond, heldout_txt=HELDOUT_TXT, heldout_mecab=HELDOUT_MECAB, limit=None,
        min_forms=2, min_occ=5, check_offsets=False):
    scorer = StemScorer(tokenizer, cond)
    # stem -> Counter(marked_form -> count); marked_form -> (aligned, stem_ids); stem -> set(plain forms)
    stem_forms = defaultdict(Counter)
    stem_plain = defaultdict(set)
    form_result = {}
    n_eligible = n_aligned = 0
    n_sent = 0
    offset_mismatch = 0
    with heldout_txt.open(encoding="utf-8") as ft, heldout_mecab.open(encoding="utf-8") as fm:
        for sent, line in zip(ft, fm):
            sent = sent.rstrip("\n")
            morphs = json.loads(line)["m"]
            n_sent += 1
            if limit and n_sent > limit:
                break
            # marks never touch whitespace, so split() of the marked sentence aligns 1:1 by position
            marked_eojs = cond.mark(sent, morphs).split()
            for idx, eoj, stem, suf in gold_pairs_idx(sent, morphs):
                me = marked_eojs[idx]
                assert me.replace(SPLIT_MARK, "") == eoj
                key = (stem, me)
                stem_forms[stem][me] += 1
                stem_plain[stem].add(eoj)
                n_eligible += 1
                if key not in form_result:
                    ids, toks = scorer.encode_eojeol(me)
                    nb = len((" " + stem).encode("utf-8"))
                    aligned, sids = StemScorer.stem_ids(ids, toks, nb)
                    form_result[key] = (aligned, sids)
                    if check_offsets:
                        enc = tokenizer.encode(" " + eoj, add_special_tokens=False)
                        b_char = 1 + len(stem)
                        hf_aligned = any(e == b_char for _, e in enc.offsets) and not any(s < b_char < e for s, e in enc.offsets)
                        if hf_aligned != aligned:
                            offset_mismatch += 1
                n_aligned += form_result[key][0]
    # M2a
    num = den = 0
    n_stems = 0
    per_stem = []
    for stem, forms in stem_forms.items():
        total = sum(forms.values())
        if len(stem_plain[stem]) < min_forms or total < min_occ:
            continue
        n_stems += 1
        seqs = Counter()
        for me, c in forms.items():
            aligned, sids = form_result[(stem, me)]
            if aligned:
                seqs[sids] += c
        top = seqs.most_common(1)[0][1] if seqs else 0
        num += top
        den += total
        per_stem.append((stem, total, len(stem_plain[stem]), top / total))
    m2a = num / den if den else float("nan")
    m2b = n_aligned / n_eligible if n_eligible else float("nan")
    return {"m2a": m2a, "m2b": m2b, "n_stems": n_stems, "n_stem_occ": den, "n_eligible_eojeol": n_eligible,
            "n_unique_forms": len({me for _, me in form_result}), "offset_mismatch": offset_mismatch if check_offsets else None,
            "per_stem": per_stem}


if __name__ == "__main__":
    from pretok import get_condition
    from train_tokenizers import load
    cond_name = sys.argv[1] if len(sys.argv) > 1 else "P0"
    vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 128000
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tok = load(cond_name, vocab)
    cond = get_condition(cond_name)
    r = run(tok, cond, limit=limit, check_offsets=(cond_name == "P0"))
    ps = r.pop("per_stem")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    ps.sort(key=lambda x: -x[1])
    print("top stems (stem, occ, forms, consistency):")
    for row in ps[:15]:
        print(row)
