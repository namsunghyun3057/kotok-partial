"""S10 seed extension: with env KOTOK_SEED=<seed>, train 8 conds x {16K, 256K} on that seed's
(existing) marked corpora, then metrics + S2 cost decomposition on the fixed held-out.
Output: results/stage2/seeds/<seed>/s10_raw.json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, IS_BASE, TOK_DIR, STAGE2
assert not IS_BASE
import train_tokenizers
from metrics import evaluate
from s2_cost_decomposition import run as s2_run

CONDS = ["P0", "P1", "P2G-10", "P2G-30", "P2G-100", "P2R-10", "P2R-30", "P2R-100"]
NEW_VOCABS = (16000, 256000)
log = lambda *a: print(f"[s10 seed {SEED} {time.strftime('%H:%M:%S')}]", *a, flush=True)
cache_p = STAGE2 / "s10_raw.json"
raw = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
for v in NEW_VOCABS:
    for c in CONDS:
        if not (TOK_DIR / f"{c}_{v}" / "tokenizer.json").exists():
            log("train", c, v); train_tokenizers.train(c, v)
for v in NEW_VOCABS:
    tok0 = None
    for c in CONDS:
        key = f"{c}_{v}"
        if key in raw and (c == "P0" or "from_splits" in raw[key]):
            continue
        r, e = evaluate(c, v)
        row = {"fertility": r["fertility"], "m2a": r["m2a"], "m2b": r["m2b"], "split_rate": e["split_rate"], "m3_nonascii": e["m3_nonascii"]}
        if c != "P0":
            tok0 = tok0 or train_tokenizers.load("P0", v)
            s2 = s2_run(c, v, tok0)
            row.update({"delta_fert": s2["delta_fert"], "from_splits": s2["from_splits"], "from_fragmentation": s2["from_fragmentation"],
                        "stem_single_token_rate": s2["stem_single_token_rate"]})
        raw[key] = row
        cache_p.write_text(json.dumps(raw, indent=1), encoding="utf-8")
        log("done", key)
log("all finished")
