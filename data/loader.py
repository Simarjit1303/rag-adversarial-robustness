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

def load_corpus(name: str, split: str='dev'):
    if name not in CORPORA:
        raise ValueError(f"Unknown corpus '{name}'. Options: {list(CORPORA)}")
    cfg = CORPORA[name]
    cache_path = DATA_DIR / f'{name}_{split}.jsonl'
    if cache_path.exists():
        with cache_path.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    n = cfg['dev_n'] if split == 'dev' else cfg['eval_n']
    load_kwargs = {'path': cfg['hf_id'], 'revision': cfg['revision']}
    if 'hf_config' in cfg:
        load_kwargs['name'] = cfg['hf_config']
    raw = load_dataset(**load_kwargs, split='train', trust_remote_code=False)
    sampled = _sample(raw, n)
    records = [dict(row) for row in sampled]
    fd, tmp_path = tempfile.mkstemp(dir=str(DATA_DIR), prefix='.tmp_', suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.replace(tmp_path, str(cache_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f'[loader] {name}/{split}: cached {len(records)} records -> {cache_path}')
    return records
if __name__ == '__main__':
    for corpus_name in CORPORA:
        load_corpus(corpus_name, split='dev')
