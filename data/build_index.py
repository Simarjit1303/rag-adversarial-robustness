"""
Builds a FAISS index over a corpus using sentence-transformers embeddings.

One index per corpus, cached to disk so it only gets built once. Loading an
existing index takes a couple of seconds; building one from scratch depends
on corpus size and whether you have a GPU available.
"""

import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, INDEX_DIR
from data.loader import load_corpus
from data.normalize import extract_passage_text

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
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

    faiss.write_index(index, str(index_path))
    with meta_path.open("wb") as f:
        pickle.dump(records, f)

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
