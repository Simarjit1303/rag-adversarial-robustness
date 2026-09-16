import sys
from pathlib import Path
from huggingface_hub import HfApi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CORPORA

def main():
    api = HfApi()
    for name, cfg in CORPORA.items():
        info = api.dataset_info(cfg['hf_id'])
        pinned = cfg.get('revision')
        if pinned is None:
            status = 'unpinned'
        elif pinned == info.sha:
            status = 'match'
        else:
            status = 'DRIFT — pinned snapshot is no longer the live head'
        print(f"{name}: {cfg['hf_id']}")
        print(f'  live sha: {info.sha}')
        print(f'  pinned:   {pinned}  [{status}]')
if __name__ == '__main__':
    main()
