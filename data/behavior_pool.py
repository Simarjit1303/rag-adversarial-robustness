"""
Loads Crescendo's target-behavior pool: JBB-Behaviors + HarmBench, per
config.BEHAVIOR_DATASETS -- see phase2_crescendo_task.md's Grounding
section ("pull directly, don't hand-write behaviors"). Same shape as
data/loader.py's corpus loading: pinned revision, cached to disk as JSONL,
atomic write. Two differences from that loader: no sampling here (the full
pool is cached; attacks.crescendo.sample_behaviors does the seeded n=100
draw over the cached pool, mirroring how attacks.poisonedrag.
sample_target_questions samples over data.build_index's full record list)
and no split argument (each dataset config carries its own fixed split).
"""

import json
import os
import tempfile

from datasets import load_dataset

from config import BEHAVIOR_DATASETS, DATA_DIR

_CACHE_PATH = DATA_DIR / "behavior_pool.jsonl"


def _load_one(name: str) -> list:
    cfg = BEHAVIOR_DATASETS[name]
    if cfg["revision"] is None:
        print(
            f"[behavior_pool] WARNING: no revision pinned for '{name}' "
            f"({cfg['hf_id']}). Loading '{cfg['split']}'@main. Pin this "
            f"with scripts/fetch_corpus_revisions.py before the real "
            f"Crescendo sweep -- see config.BEHAVIOR_DATASETS."
        )
    load_kwargs = {"path": cfg["hf_id"], "revision": cfg["revision"]}
    if cfg.get("hf_config"):
        load_kwargs["name"] = cfg["hf_config"]
    raw = load_dataset(**load_kwargs, split=cfg["split"], trust_remote_code=False)

    column = cfg["behavior_column"]
    behaviors = [
        {"behavior": row[column].strip(), "source": name}
        for row in raw
        if row.get(column) and row[column].strip()
    ]
    if len(behaviors) != cfg["expected_n"]:
        print(
            f"[behavior_pool] WARNING: '{name}' loaded {len(behaviors)} behaviors, "
            f"expected {cfg['expected_n']} -- the dataset's split/config may have "
            f"changed upstream since this project's config.BEHAVIOR_DATASETS entry "
            f"was written. Verify before trusting a sweep sampled from this pool."
        )
    return behaviors


def load_behavior_pool(use_cache: bool = True) -> list:
    """
    Returns the full combined pool: [{"behavior": str, "source": "jbb_behaviors"
    | "harmbench"}, ...]. Cached as one JSONL across both datasets, same
    atomic-write discipline as data/loader.py (an interrupted write must
    never leave a partially-written cache that silently loads as a smaller
    pool on the next run).
    """
    if use_cache and _CACHE_PATH.exists():
        with _CACHE_PATH.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    pool = []
    for name in BEHAVIOR_DATASETS:
        pool.extend(_load_one(name))

    if use_cache:
        fd, tmp_path = tempfile.mkstemp(dir=str(DATA_DIR), prefix=".tmp_", suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for row in pool:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp_path, str(_CACHE_PATH))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    print(f"[behavior_pool] cached {len(pool)} behaviors -> {_CACHE_PATH}")
    return pool


if __name__ == "__main__":
    load_behavior_pool()
