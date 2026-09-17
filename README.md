# kotok-partial

Official code and artifacts for the HCLT 2026 paper

**조사를 몇 개 떼어야 하는가? 한국어 서브워드 토큰화에서 부분 분리의 토큰 비용과 어간 일관성**
*How Many Particles Should Be Split Off? Token Cost and Stem Consistency of Partial Pre-tokenization for Korean Subword Tokenization*

Sunghyun Nam, Jimin Lee, Jungmin Cha, Jaehyung Seo — Konkuk University

---

## TL;DR

Korean subword tokenization is usually compared at two extremes: split every morpheme with an
analyzer, or split nothing. We measure the **middle ground** — splitting only the *k* most frequent
particles and endings (조사·어미) during pre-tokenization — across *k* ∈ {10, 30, 100} and five
vocabulary sizes (16K–256K), **without training a single language model**.

Three findings:

1. **The consistency gain tracks coverage and is almost independent of vocabulary size**, but the
   fertility cost **grows linearly in log₂(V)** — partial separation is most expensive exactly where
   modern LLMs live (large vocabularies).
2. **One split point costs less than one token** (0.5–0.8), because the baseline tokenizer was
   already fragmenting the word. This mechanism lets you **bound the cost from the base tokenizer
   and a frequency table alone**, with no tokenizer training.
3. **An analyzer-free rule** matches analyzer-guided splitting at *k* = 10 and comes within
   1.1 %p at *k* = 30; its dominant failure is **under-splitting**, not over-splitting.

## Headline numbers (HPLT held-out, vocabulary 128K, mean of 3 corpus seeds)

| Condition | Split words | Fertility | Δ vs P0 | M2a (stem-ID consistency) |
|---|---|---|---|---|
| **P0** no split | 0 | 1.572 | — | 6.6 |
| **P2G-30** analyzer-guided, *k* = 30 | 44.8 % | 1.911 | +21.5 % | 92.2 |
| **P2R-30** rule-based, *k* = 30 | 41.0 % | 1.914 | +21.7 % | 91.1 |
| **P1** split every morpheme | 68.9 % | 2.140 | +36.1 % | 99.65 |

M2a's practical ceiling is **99.65 %**, the analyzer's own self-consistency — not 100 %.
Full tables for all 8 conditions × 5 vocabulary sizes are in [`results/stage2/`](results/stage2/).

## Conditions

| Name | Pre-tokenization |
|---|---|
| `P0` | none (byte-level BPE with the GPT-2 regex only) |
| `P1` | split at **every** Mecab-ko morpheme boundary inside a word |
| `P2G-k` | **analyzer-guided**: split before a morpheme whose tag is in *T* and whose surface form is in the top-*k* list |
| `P2R-k` | **rule-based, analyzer-free**: suffix-string tiling + a stem table; deployable at inference time |
| `P2R+-k` | post-hoc variant of `P2R` whose stem table also contains fused past-tense stems (했, 됐, 말했) |

All conditions share the same BPE algorithm, regex and hyperparameters. Splits are expressed by
inserting a private-use character (U+E000) into the corpus offline, so every tokenizer serializes
to a plain `tokenizer.json` with no custom Python.

## Tokenizer configuration

**No pretrained tokenizer or language model is used anywhere.** Every tokenizer in the paper is
trained from scratch on our own corpus with Hugging Face `tokenizers` 0.23.2
(`scripts/train_tokenizers.py`):

| Setting | Value |
|---|---|
| Model | byte-level BPE (`models.BPE`, `byte_fallback=False`) |
| Pre-tokenizer | `Split(U+E000, removed)` → `ByteLevel(add_prefix_space=False, use_regex=True)` |
| Regex | the GPT-2 pattern, identical in every condition |
| Decoder | `ByteLevel` |
| Vocabulary sizes | 16K, 32K, 64K, 128K, 256K |
| `min_frequency` | 2 |
| Special tokens | none — so byte tokens occupy ids 0–255 |
| Initial alphabet | `ByteLevel.alphabet()` (all 256 bytes) |
| Training data | 1,000,000 Korean sentences from HPLT v2.0 cleaned |
| Corpus seeds | 20260907 (reference), 20260908, 20260909 |

Only the pre-tokenization differs between conditions: the `Split` step removes the private-use
marks that `scripts/pretok.py` inserted, so `P0` and `P2G-30` see the same text under different
chunk boundaries. The morphological analyzer (`python-mecab-ko` 1.3.7 with `mecab-ko-dic` 2.1.1)
is used only to place those marks and to define the gold boundaries for M2 — never inside the
tokenizer. `P2R` needs no analyzer at inference time at all.

`120 tokenizers` = 8 conditions × 5 vocabulary sizes × 3 corpus seeds. Each takes 14–66 s to train.

## What is in this repository

```
scripts/            stage-1 pipeline: corpus, k-list, stem table, tokenizer training, metrics
  config.py           seeds, paths, tag sets — single source of truth
  prepare_corpus.py   HPLT shards -> train / held-out split (no document overlap)
  particle_freq.py    top-k particle/ending list + coverage
  build_stem_table.py stem table for the analyzer-free rule
  pretok.py           the four pre-tokenization conditions
  train_tokenizers.py byte-level BPE training
  metrics.py          M1 fertility, M3 byte fallback
  stem_consistency.py M2a / M2b (byte-accumulation boundary alignment)
  stage2/             every analysis reported in the paper (S0-S11)
results/            every number in the paper, as produced
  particle_freq.csv   frequency table and cumulative coverage
  metrics.csv         stage-1 metrics, 8 conditions at 128K
  stage2/             24-condition metrics, cost decomposition, prediction,
                      OOD, bootstrap CIs, 3-seed resampling, vocabulary extension
data/               small derived artifacts (see below)
figures/            the two figures in the paper, as PDF and PNG
```

**Not included**, and why:

- `tokenizers/` — 120 trained tokenizers, ~1.1 GB. Rebuild with `train_tokenizers.py`
  (14–66 s each) or open an issue if you would like them hosted.
- the corpora themselves — HPLT v2.0 cleaned Korean is public; `prepare_corpus.py` reproduces
  our exact split from the two shards named in `data/corpus_stats.json`, and
  `data/heldout_docids.txt` pins the held-out documents.
- `*.mecab.jsonl` tagged corpora — regenerate with `tag_corpus.py`.

Included derived artifacts:

| File | What it is |
|---|---|
| `data/particles_k{10,30,100}.txt` | the split-target lists (coverage 61.7 / 87.9 / 97.9 %) |
| `data/stem_table.tsv` | 370,878 stem surface forms with counts; rule (b) uses count ≥ 5 |
| `data/stem_table_plus.tsv` | the `P2R+` variant of the same table |
| `data/heldout_docids.txt` | the 2,491 held-out document ids |
| `data/corpus_stats.json` | shard names, seed, split statistics |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv/Scripts/pip
```

Python 3.11. The analyzer is `python-mecab-ko` 1.3.7 with `mecab-ko-dic` 2.1.1; the tokenizer
library is Hugging Face `tokenizers` 0.23.2. `OPENBLAS_NUM_THREADS=1` is recommended — BPE
training at 128K peaks around 2.5 GB of committed memory for 1M sentences.

## Reproducing the paper

```bash
# 1. corpus: 167,530 HPLT documents -> 100k held-out + 1M training sentences, no overlap
python scripts/prepare_corpus.py

# 2. morphological tagging
python scripts/tag_corpus.py data/heldout_100k.txt data/heldout_100k.mecab.jsonl 8
python scripts/tag_corpus.py data/train_1M.txt     data/train_1M.mecab.jsonl     8

# 3. split-target list and stem table (training corpus only — never the held-out set)
python scripts/particle_freq.py
python scripts/build_stem_table.py

# 4. marked corpora, then one tokenizer per (condition, vocabulary)
python scripts/make_marked_corpora.py
python scripts/train_tokenizers.py P0 128000        # repeat per condition and size

# 5. metrics
python scripts/metrics.py P0 128000 P1 128000 ...   # -> results/metrics.csv
python scripts/sanity_check.py

# 6. the analyses in the paper
python scripts/stage2/s1_run.py                     # 24 conditions, 32K/64K/128K
python scripts/stage2/s2_cost_decomposition.py      # delta fertility = m(1 + phi)
python scripts/stage2/s3_prediction.py              # training-free predictors
python scripts/stage2/s5_ood.py                     # Korean Wikipedia, out of domain
python scripts/stage2/s8_bootstrap.py               # document-level cluster bootstrap
python scripts/stage2/s10_vocab_ext.py              # 16K and 256K
python scripts/make_figures_v2.py                   # figures/
```

The resampling study runs the same pipeline under `KOTOK_SEED=20260908` and `20260909`;
`config.py` redirects every train-side path so the held-out set stays fixed.

## Where each number in the paper comes from

| Paper | File |
|---|---|
| Table 1 (fertility), Table 2 (M2a/M2b) | `results/stage2/metrics_all_vocab.csv`, `s10_vocab_ext.csv` |
| Table 3 (cost decomposition, φ) | `results/stage2/s2_cost_decomposition.csv` |
| Figure 1 (a)–(d) | `scripts/make_figures_v2.py` over the two files above |
| Figure 2 (prediction) | `results/stage2/s3_prediction.csv` |
| §5.3 confidence intervals | `results/stage2/s8_bootstrap.csv`, `s8_bootstrap_diff.csv` |
| §5.3 out-of-domain | `results/stage2/s5_ood.csv` |
| §5.4 error types, `P2R+` | `results/stage2/s6_error_types.csv`, `s6_p2r_vs_p2rp.md` |
| §5.5 vocabulary reallocation | `results/stage2/s4_vocab_composition.csv`, `s11_realloc.md` |
| 3-seed ranges | `results/stage2/s9_seed_summary.csv`, `results/stage2/seeds/` |

The acceptance criterion for the training-free predictors was **recorded before the numbers were
computed** and is in [`results/stage2/s3_criteria.md`](results/stage2/s3_criteria.md). We report
the estimator that missed it as missing it.

## Notes

- No language model is trained anywhere in this repository. Every metric is computed from
  tokenizer output alone, which is the paper's main limitation as well as its design.
- `M2` excludes fused words (했다 = 하 + 았 + 다), so it cannot see split quality on fused verb
  inflection. `results/stage2/s7_m2b_fused.md` reports a supplementary metric for those.
- `P2R+` and the corrected `M2a` formula were devised **after** seeing the results. They are
  reported separately from the original `P2R` numbers throughout.

## Citation

```bibtex
@inproceedings{nam2026kotokpartial,
  title     = {조사를 몇 개 떼어야 하는가? 한국어 서브워드 토큰화에서 부분 분리의 토큰 비용과 어간 일관성},
  author    = {남성현 and 이지민 and 차정민 and 서재형},
  booktitle = {제38회 한글 및 한국어 정보처리 학술대회 (HCLT)},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)).

The derived artifacts under `data/` and `results/` are computed from
[HPLT v2.0 cleaned Korean](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned), which is
released under **CC0 1.0**, so they carry no additional restriction. They are frequency tables and
aggregate statistics, not corpus text: `stem_table.tsv` is a list of stem surface forms with
counts, and no source sentence is reproduced anywhere in this repository. As with any frequency
list built from a web crawl, the long tail contains proper nouns that occur in public web pages.
If you find an entry that should not be there, open an issue and we will remove it.

The out-of-domain evaluation uses Korean Wikipedia (CC BY-SA 4.0); only aggregate metrics computed
over it are published here.
