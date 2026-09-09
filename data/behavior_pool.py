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
from datasets.exceptions import DatasetNotFoundError

from config import BEHAVIOR_DATASETS, DATA_DIR

_CACHE_PATH = DATA_DIR / "behavior_pool.jsonl"


def _load_one_hf_config(hf_id: str, revision, hf_config, split: str, name: str):
    """One load_dataset() call for one (hf_id, hf_config, split) combination
    -- factored out of _load_one so a dataset spanning several configs
    (HarmBench's real behavior set is standard+contextual+copyright, not
    one config -- see config.BEHAVIOR_DATASETS' harmbench comment) can call
    this once per config and concatenate, instead of duplicating the
    load/error-handling logic per subset."""
    load_kwargs = {"path": hf_id, "revision": revision}
    if hf_config:
        load_kwargs["name"] = hf_config
    try:
        return load_dataset(**load_kwargs, split=split, trust_remote_code=False)
    except DatasetNotFoundError as e:
        # walledai/HarmBench is gated -- confirmed live 2026-09-09, see
        # config.BEHAVIOR_DATASETS' comment. A plain DatasetNotFoundError
        # traceback here reads identically to "wrong repo id"; this makes
        # the actual, actionable cause (access not yet granted or an
        # HF_API_TOKEN not set/passed to datasets) explicit instead.
        raise RuntimeError(
            f"[behavior_pool] '{name}' ({hf_id}, config={hf_config!r}) failed to load -- "
            f"if this is a gated dataset, request access at "
            f"https://huggingface.co/datasets/{hf_id} and make sure "
            f"HF_TOKEN/HF_API_TOKEN is set in the environment before retrying. "
            f"Original error: {e}"
        ) from e


def _load_one(name: str) -> list:
    cfg = BEHAVIOR_DATASETS[name]
    if cfg["revision"] is None:
        print(
            f"[behavior_pool] WARNING: no revision pinned for '{name}' "
            f"({cfg['hf_id']}). Loading '{cfg['split']}'@main. Pin this "
            f"with scripts/fetch_corpus_revisions.py before the real "
            f"Crescendo sweep -- see config.BEHAVIOR_DATASETS."
        )

    # hf_config may be a single config name or a list of them -- HarmBench's
    # real public behavior set (400: 200 standard + 100 contextual + 100
    # copyright, confirmed live 2026-09-09) spans three configs on one
    # dataset repo, not one. A plain string is normalized to a 1-item list
    # so this loop covers both shapes uniformly.
    hf_configs = cfg.get("hf_config")
    if hf_configs is None or isinstance(hf_configs, str):
        hf_configs = [hf_configs]

    column = cfg["behavior_column"]
    behaviors = []
    for hf_config in hf_configs:
        raw = _load_one_hf_config(cfg["hf_id"], cfg["revision"], hf_config, cfg["split"], name)
        behaviors.extend(
            {"behavior": row[column].strip(), "source": name}
            for row in raw
            if row.get(column) and row[column].strip()
        )

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
