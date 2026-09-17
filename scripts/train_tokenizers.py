"""4.5 Train byte-level BPE tokenizers, one per (condition, vocab).

Pipeline (identical for all conditions):
  marked corpus file -> Split(SPLIT_MARK, removed) -> ByteLevel(add_prefix_space=False, use_regex=True)
  -> BPE(byte alphabet, no special tokens => byte tokens are ids 0..255)
Saved to tokenizers/{cond}_{vocab}/tokenizer.json (fully serialisable; no custom python objects).
Usage: python scripts/train_tokenizers.py P0 128000
"""
import sys, json, time
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

sys.path.insert(0, str(Path(__file__).parent))
from config import SPLIT_MARK, TOK_DIR, DATA, SEED_DIR, TRAIN_TXT, RESULTS, RESULTS_SEED


def marked_corpus_path(cond_name: str) -> Path:
    return TRAIN_TXT if cond_name == "P0" else SEED_DIR / f"train_1M.{cond_name}.txt"


def build_tokenizer():
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(SPLIT_MARK, behavior="removed"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def load(cond_name: str, vocab_size: int) -> Tokenizer:
    return Tokenizer.from_file(str(TOK_DIR / f"{cond_name}_{vocab_size}" / "tokenizer.json"))


def train(cond_name: str, vocab_size: int):
    corpus = marked_corpus_path(cond_name)
    assert corpus.exists(), corpus
    out_dir = TOK_DIR / f"{cond_name}_{vocab_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t0 = time.time()
    tok.train([str(corpus)], trainer)
    elapsed = time.time() - t0
    tok.save(str(out_dir / "tokenizer.json"))
    vocab = tok.get_vocab()
    byte_tokens = sum(1 for t in vocab if len(t) == 1)
    dec = decoders.ByteLevel()
    n_hangul = 0
    for t in vocab:
        try:
            s = dec.decode([t])
        except Exception:
            s = ""
        if any("가" <= ch <= "힣" for ch in s):
            n_hangul += 1
    info = {"cond": cond_name, "vocab_size": vocab_size, "actual_vocab": len(vocab),
            "byte_tokens": byte_tokens, "hangul_tokens": n_hangul, "train_seconds": round(elapsed, 1),
            "corpus": corpus.name, "min_frequency": 2}
    (out_dir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS_SEED.mkdir(parents=True, exist_ok=True)
    with (RESULTS_SEED / "train_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    print(json.dumps(info, ensure_ascii=False))
    return tok


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]))
