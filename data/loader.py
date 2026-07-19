"""
Corpus loading and sampling for the three retrieval knowledge bases.

Run this once per corpus to produce a fixed dev slice (1,000 docs) and,
later, a fixed eval slice (10,000 docs). Sampling is seeded so re-running
produces the exact same split — this matters because Phase 2/3/4 attack
results need to be measured against the same documents the baseline was
measured on, not a fresh random draw each time. The pool being sampled is
itself fixed too: every load_dataset() call is pinned to the revision in
config.CORPORA, so an upstream re-upload can't silently change the corpus.
"""

import json
import os
import random
import tempfile

from datasets import load_dataset

from config import CORPORA, DATA_DIR, SEED


def _sample(dataset, n, seed=SEED):
    rng = random.Random(seed)
    n = min(n, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    return dataset.select(indices)


def load_corpus(name: str, split: str = "dev"):
    """
    Load and cache a fixed sample of the named corpus.

    name: one of "nq_open", "hotpot_qa", "ms_marco" (see config.CORPORA)
    split: "dev" (1,000 docs) or "eval" (10,000 docs)
    """
    if name not in CORPORA:
        raise ValueError(f"Unknown corpus '{name}'. Options: {list(CORPORA)}")

    cfg = CORPORA[name]
    cache_path = DATA_DIR / f"{name}_{split}.jsonl"

    if cache_path.exists():
        # Explicit encoding: Python defaults to the locale encoding until 3.15
        # (PEP 686) — cp1252 on Windows — which corrupts non-ASCII corpus text.
        with cache_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    n = cfg["dev_n"] if split == "dev" else cfg["eval_n"]

    # revision pins the dataset snapshot, same shape as harness/model_loader.py
    # does for model weights — see the CORPORA comment in config.py.
    load_kwargs = {"path": cfg["hf_id"], "revision": cfg["revision"]}
    if "hf_config" in cfg:
        load_kwargs["name"] = cfg["hf_config"]

    raw = load_dataset(**load_kwargs, split="train", trust_remote_code=False)
    sampled = _sample(raw, n)

    records = [dict(row) for row in sampled]
    # Atomic write (temp in same dir + os.replace), same discipline as
    # data/build_index.py: an interrupted write truncated exactly at a line
    # boundary would otherwise SILENTLY load as a smaller corpus on the next
    # run — the one truncation case in this pipeline that doesn't hard-crash.
    fd, tmp_path = tempfile.mkstemp(dir=str(DATA_DIR), prefix=".tmp_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_path, str(cache_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"[loader] {name}/{split}: cached {len(records)} records -> {cache_path}")
    return records


if __name__ == "__main__":
    # Build every dev slice up front. Run again with split="eval" once the
    # harness is validated on the smaller dev set.
    for corpus_name in CORPORA:
        load_corpus(corpus_name, split="dev")
