import json
import os
import tempfile
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from config import BEHAVIOR_DATASETS, DATA_DIR
_CACHE_PATH = DATA_DIR / 'behavior_pool.jsonl'

def _load_one_hf_config(hf_id: str, revision, hf_config, split: str, name: str):
    load_kwargs = {'path': hf_id, 'revision': revision}
    if hf_config:
        load_kwargs['name'] = hf_config
    try:
        return load_dataset(**load_kwargs, split=split, trust_remote_code=False)
    except DatasetNotFoundError as e:
        raise RuntimeError(f"[behavior_pool] '{name}' ({hf_id}, config={hf_config!r}) failed to load -- if this is a gated dataset, request access at https://huggingface.co/datasets/{hf_id} and make sure HF_TOKEN/HF_API_TOKEN is set in the environment before retrying. Original error: {e}") from e

def _load_one(name: str) -> list:
    cfg = BEHAVIOR_DATASETS[name]
    if cfg['revision'] is None:
        print(f"[behavior_pool] WARNING: no revision pinned for '{name}' ({cfg['hf_id']}). Loading '{cfg['split']}'@main. Pin this with scripts/fetch_corpus_revisions.py before the real Crescendo sweep -- see config.BEHAVIOR_DATASETS.")
    hf_configs = cfg.get('hf_config')
    if hf_configs is None or isinstance(hf_configs, str):
        hf_configs = [hf_configs]
    column = cfg['behavior_column']
    behaviors = []
    for hf_config in hf_configs:
        raw = _load_one_hf_config(cfg['hf_id'], cfg['revision'], hf_config, cfg['split'], name)
        behaviors.extend(({'behavior': row[column].strip(), 'source': name} for row in raw if row.get(column) and row[column].strip()))
    if len(behaviors) != cfg['expected_n']:
        print(f"[behavior_pool] WARNING: '{name}' loaded {len(behaviors)} behaviors, expected {cfg['expected_n']} -- the dataset's split/config may have changed upstream since this project's config.BEHAVIOR_DATASETS entry was written. Verify before trusting a sweep sampled from this pool.")
    return behaviors

def load_behavior_pool(use_cache: bool=True) -> list:
    if use_cache and _CACHE_PATH.exists():
        with _CACHE_PATH.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    pool = []
    for name in BEHAVIOR_DATASETS:
        pool.extend(_load_one(name))
    if use_cache:
        fd, tmp_path = tempfile.mkstemp(dir=str(DATA_DIR), prefix='.tmp_', suffix='.jsonl')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for row in pool:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            os.replace(tmp_path, str(_CACHE_PATH))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    print(f'[behavior_pool] cached {len(pool)} behaviors -> {_CACHE_PATH}')
    return pool
if __name__ == '__main__':
    load_behavior_pool()
