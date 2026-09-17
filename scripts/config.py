"""Project-wide constants. Single source of truth for seeds, paths, tag sets."""
import os
from pathlib import Path

BASE_SEED = 20260907
# Per-seed workspace for the resampling study (S9): KOTOK_SEED=20260908 etc. Held-out is shared.
SEED = int(os.environ.get("KOTOK_SEED", str(BASE_SEED)))
IS_BASE = SEED == BASE_SEED

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED_DIR = DATA if IS_BASE else DATA / f"seed_{SEED}"          # train-side artifacts
TOK_DIR = ROOT / "tokenizers" if IS_BASE else ROOT / "tokenizers" / f"seed_{SEED}"
STAGE2 = RESULTS / "stage2" if IS_BASE else RESULTS / "stage2" / "seeds" / str(SEED)
RESULTS_SEED = RESULTS if IS_BASE else STAGE2

TRAIN_TXT = SEED_DIR / "train_1M.txt"
HELDOUT_TXT = DATA / "heldout_100k.txt"
TRAIN_MECAB = SEED_DIR / "train_1M.mecab.jsonl"
HELDOUT_MECAB = DATA / "heldout_100k.mecab.jsonl"
PARTICLE_FREQ = RESULTS / "particle_freq.csv" if IS_BASE else STAGE2 / "particle_freq.csv"
STEM_TABLE = SEED_DIR / "stem_table.tsv"

N_TRAIN = 1_000_000
N_HELDOUT = 100_000

# Tag set T: all particles (J*) + all endings (E*). Used for the k-list.
# Mecab-ko-dic emits compound tags like "EP+EF" (세요) or "VV+EC" (해).
# A morpheme counts as particle/ending only if EVERY '+'-joined tag is in T.
TAG_SET_T = {
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
}

# Stem tags for M2 (content morphemes). A stem = maximal run of these
# at the start of an eojeol, followed by >=1 morpheme from TAG_SET_T.
STEM_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "XR", "XSN", "XSV", "XSA", "VCP", "VCN", "MAG"}

K_LIST = [10, 30, 100]
VOCAB_SIZES = [128_000]  # 32K/64K only if time permits

# Private-use char inserted at split points for the offline pretokenization.
SPLIT_MARK = "\ue000"


def is_particle_tag(tag: str) -> bool:
    return all(t in TAG_SET_T for t in tag.split("+"))

# S0 (results/stage2/s0_p1_mismatch.md): with P1 every stem is its own pre-token chunk, so the
# only M2a mismatches are Mecab segmenting the same stem surface differently across contexts.
# => the practical ceiling of M2a is the gold analyzer's self-consistency, not 100%.
GOLD_CEILING_M2A = 0.9965
