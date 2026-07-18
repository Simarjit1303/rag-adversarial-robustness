"""
Builds a FAISS index over a corpus using sentence-transformers embeddings.

One index per corpus, cached to disk so it only gets built once. Loading an
existing index takes a couple of seconds; building one from scratch depends
on corpus size and whether you have a GPU available.
"""

import os
import pickle
import tempfile

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, INDEX_DIR
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
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _embedder


def build_index(corpus_name: str, split: str = "dev", force_rebuild: bool = False):
    index_path = INDEX_DIR / f"{corpus_name}_{split}.faiss"
    meta_path = INDEX_DIR / f"{corpus_name}_{split}_meta.pkl"

    if index_path.exists() and meta_path.exists() and not force_rebuild:
        index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            records = pickle.load(f)
        return index, records

    records = load_corpus(corpus_name, split=split)
    texts = [extract_passage_text(corpus_name, r) for r in records]

    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    # Metadata first, index second: the cache check requires BOTH files, and
    # a crash between the two renames then leaves new-meta + missing/old
    # index. Records only append-extend between rebuilds of the same split,
    # so new-meta/old-index is the harmless pairing; the reverse (new index,
    # old records) could silently misalign retrieval results.
    _atomic_write_pickle(records, meta_path)
    _atomic_write_index(index, index_path)

    print(f"[index] {corpus_name}/{split}: {len(records)} vectors -> {index_path}")
    return index, records


def retrieve(index, records, query: str, k: int = 5):
    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    return [(records[i], float(scores[0][rank])) for rank, i in enumerate(idxs[0])]


if __name__ == "__main__":
    from config import CORPORA
    for corpus_name in CORPORA:
        build_index(corpus_name, split="dev")
