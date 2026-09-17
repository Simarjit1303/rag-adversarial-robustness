"""
Builds a FAISS index over a corpus using sentence-transformers embeddings.

One index per corpus, cached to disk so it only gets built once. Loading an
existing index takes a couple of seconds; building one from scratch depends
on corpus size and whether you have a GPU available.
"""

import os
import pickle
import sys
import tempfile
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import CORPORA, EMBEDDING_MODEL, INDEX_DIR
from data.loader import load_corpus
from data.normalize import extract_passage_text

_embedder = None


# Atomic writes: the cache check below trusts exists(), and the Stage 3 Job
# runs with --replica-retry-limit 0, so an interrupted write would leave a
# truncated file that permanently wedges the cache (faiss.read_index and
# pickle.load both hard-crash on truncated input — verified empirically, see
# PR; safe but requires manual cleanup). Writing to a temp file in the SAME
# directory and os.replace()-ing it in means the final path only ever holds
# nothing or a complete file. Same-directory matters: os.replace is only
# atomic within one filesystem. NOTE: atomicity over an Azure Files SMB
# mount is NOT yet verified — re-test on /mnt/rag-scratch after Part B.

def _atomic_write_index(index, final_path):
    final_path = str(final_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(final_path), prefix=".tmp_", suffix=".faiss"
    )
    os.close(fd)
    try:
        faiss.write_index(index, tmp_path)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_write_pickle(obj, final_path):
    final_path = str(final_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(final_path), prefix=".tmp_", suffix=".pkl"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# The marker is what makes the TWO-file cache trustworthy: each atomic write
# above only guarantees its own file, so a crash between the two renames can
# still leave a mismatched (stale + fresh) pair. The marker is written last,
# only after both files landed, and the cache check below trusts ONLY the
# marker — leftover index/metadata files from an interrupted build are not
# evidence of anything. One marker per (corpus, split), NOT per directory:
# INDEX_DIR holds every corpus, and a directory-level marker would validate
# corpus B after only corpus A finished building.
#
# The marker also CARRIES the config.CORPORA revision the cache was built
# from: a marker whose content doesn't match the currently pinned revision is
# treated exactly like a missing marker. Bumping a pin in config.py therefore
# auto-invalidates every cache built under the old snapshot — no manual
# force_rebuild=True, no silent staleness. (Content-less markers from before
# this change fail the comparison too, so those caches rebuild once.)

def _mark_build_complete(marker_path, revision):
    marker_path = str(marker_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(marker_path), prefix=".tmp_", suffix=".marker"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(revision)
        os.replace(tmp_path, marker_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _is_build_complete(marker_path, expected_revision):
    marker_path = str(marker_path)
    if not os.path.exists(marker_path):
        return False
    with open(marker_path, encoding="utf-8") as f:
        return f.read().strip() == expected_revision


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Explicit device selection so embedding runs on GPU whenever CUDA is
        # present (SentenceTransformer would pick it up anyway, but this makes
        # the choice visible in logs and fails loudly on CPU-only installs).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print(
                "[index] WARNING: CUDA not available — embedding on CPU. This works "
                "but is slow for 10k-doc corpora; see requirements.txt for the "
                "CUDA-enabled torch install."
            )
        else:
            print(f"[index] Embedding on GPU: {torch.cuda.get_device_name(0)}")
        # Defensive retry: harness/model_loader.py's torch.cuda.synchronize()
        # after phi-4-mini's load closes most of the race window against this
        # embedder claiming the same GPU, but the race was observed as
        # intermittent (not deterministic), so this is cheap insurance in
        # case the window isn't fully closed. Same bounded-retry-with-backoff
        # pattern already used for pip's network retries earlier in this
        # project -- a transient failure gets a few attempts, not treated as
        # immediately fatal, but a persistent one still raises loudly rather
        # than being silently swallowed.
        last_exc = None
        for attempt in range(3):
            try:
                _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
                break
            except Exception as e:
                last_exc = e
                print(
                    f"[index] embedder init attempt {attempt + 1} failed: {e!r}"
                    + (" -- retrying" if attempt < 2 else " -- giving up"),
                    file=sys.stderr,
                )
                if attempt < 2:
                    time.sleep(2)
        else:
            raise last_exc
    return _embedder


def build_index(corpus_name: str, split: str = "dev", force_rebuild: bool = False):
    index_path = INDEX_DIR / f"{corpus_name}_{split}.faiss"
    meta_path = INDEX_DIR / f"{corpus_name}_{split}_meta.pkl"
    marker_path = INDEX_DIR / f"{corpus_name}_{split}.build_complete"
    revision = CORPORA[corpus_name]["revision"]

    # Cache validity = marker only, and the marker must record the currently
    # pinned corpus revision. Files present without a matching marker mean an
    # interrupted build or a stale snapshot — rebuild either way.
    if _is_build_complete(marker_path, revision) and not force_rebuild:
        index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            records = pickle.load(f)
        return index, records

    # Invalidate BEFORE touching the files: a crash mid-rebuild over an
    # existing valid cache must not leave the old marker validating a
    # mismatched (old + new) file pair.
    if marker_path.exists():
        os.remove(marker_path)

    records = load_corpus(corpus_name, split=split)
    texts = [extract_passage_text(corpus_name, r) for r in records]

    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    # Write order between these two is no longer load-bearing — the marker,
    # written last, is the only thing the cache check trusts.
    _atomic_write_pickle(records, meta_path)
    _atomic_write_index(index, index_path)
    _mark_build_complete(marker_path, revision)

    print(f"[index] {corpus_name}/{split}: {len(records)} vectors -> {index_path}")
    return index, records


def retrieve(index, records, query: str, k: int = 5):
    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    return [(records[i], float(scores[0][rank])) for rank, i in enumerate(idxs[0])]


def main() -> None:
    """
    Entry point for `python -m data.build_index`. Pulled out of the
    __main__ block (rather than left inline) so tests can call it directly
    with build_index() stubbed, instead of needing a real subprocess.

    RAG_CORPORA mirrors evaluation/run_baseline.py's RAG_MODELS/RAG_CORPORA
    handling -- without this, this stage (which runs BEFORE run_baseline in
    the pipeline) built every corpus in CORPORA regardless of the env var,
    which is what a real RunPod smoke test with RAG_CORPORA=nq_open actually
    hit: indices got built for nq_open AND ms_marco even though only nq_open
    was requested.
    """
    corpora_env = os.environ.get("RAG_CORPORA")
    corpus_names = (
        [c.strip() for c in corpora_env.split(",") if c.strip()]
        if corpora_env
        else list(CORPORA)
    )
    unknown_corpora = [c for c in corpus_names if c not in CORPORA]
    if unknown_corpora:
        raise ValueError(f"Unknown corpus names {unknown_corpora}. Options: {list(CORPORA)}")

    for corpus_name in corpus_names:
        build_index(corpus_name, split="dev")


if __name__ == "__main__":
    main()
