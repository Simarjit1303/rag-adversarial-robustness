"""
Fetches the current commit hash of each corpus dataset repo on the
HuggingFace Hub — the corpus-side equivalent of the "Files and versions"
tab / API `sha` method already used to pin every entry in config.MODELS.

config.CORPORA pins each corpus to a "revision" so every phase samples from
a byte-identical dataset snapshot; the sampling seed alone can't guarantee
that if the upstream dataset is ever re-uploaded. Run this to get the real,
current hashes when setting or bumping those pins — never guess a hash or
reuse one from memory:

    python -m scripts.fetch_corpus_revisions

Public datasets need no token. The pinned revision (if any) is printed next
to the live one so drift is visible at a glance.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CORPORA  # noqa: E402


def main():
    api = HfApi()
    for name, cfg in CORPORA.items():
        info = api.dataset_info(cfg["hf_id"])
        pinned = cfg.get("revision")
        if pinned is None:
            status = "unpinned"
        elif pinned == info.sha:
            status = "match"
        else:
            status = "DRIFT — pinned snapshot is no longer the live head"
        print(f"{name}: {cfg['hf_id']}")
        print(f"  live sha: {info.sha}")
        print(f"  pinned:   {pinned}  [{status}]")


if __name__ == "__main__":
    main()
