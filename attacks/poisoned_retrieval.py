"""
Retrieval side of PoisonedRAG (see attacks/poisonedrag.py for the poison-
text generation this consumes). Two jobs:

1. Retrieve top-k over the UNION of the real corpus and one target
   question's poisoned passages, without rebuilding/re-embedding the
   (large, already-cached) base FAISS index per target question -- only
   the handful of poison texts (ADV_PER_QUERY=5) get embedded fresh, then
   merged with the base index's own top-k scores by a plain sort.
   data.build_index.build_index()'s IndexFlatIP is exact brute-force
   (inner product on normalized vectors == cosine similarity, no ANN
   approximation), so merging exact scores from two sources is
   mathematically identical to a single combined index built from
   everything -- the roadmap's "rebuild the FAISS index" wording is honored
   in effect, just without 100 real full-corpus rebuilds per corpus.
2. The retrieval-verification step PHASE2_ROADMAP.md calls out explicitly:
   confirm poisoned passages actually land in top-k before generation runs,
   logged per question via evaluation.metrics.retrieval_f1_at_k -- a
   poisoned passage that never gets retrieved can't be blamed for the
   model's answer either way, so this is a real measured rate, not an
   assumed 100%.
"""

import numpy as np

from data.build_index import _get_embedder


def embed_poison_texts(poison_texts: list[str]) -> np.ndarray:
    """Same embedder singleton build_index() uses -- identical embedding
    space is what makes merging scores across the two sources valid."""
    model = _get_embedder()
    embeddings = model.encode(poison_texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32")


def retrieve_with_poison(base_index, base_records, poison_texts: list[str],
                          poison_embeddings: np.ndarray, query: str, k: int = 5):
    """
    Returns (retrieved, poison_doc_ids) where retrieved is a list of
    (record_or_poison_text, score, doc_id) tuples, top-k over the union,
    highest score first. doc_id is a real corpus-relative int for organic
    records (matching the rest of the pipeline's convention) or a string
    sentinel "poison::{i}" for a poisoned passage -- distinct types by
    construction, so a poison hit can never be mistaken for coincidentally
    matching a real integer doc_id. poison_doc_ids is the full set of this
    question's sentinel ids, for evaluation.metrics.retrieval_f1_at_k.
    """
    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

    base_scores, base_idxs = base_index.search(q_emb, k)
    base_candidates = [
        (base_records[i], float(base_scores[0][rank]), i)
        for rank, i in enumerate(base_idxs[0])
    ]

    poison_doc_ids = {f"poison::{i}" for i in range(len(poison_texts))}
    poison_scores = (poison_embeddings @ q_emb[0])  # inner product, same metric as the FAISS index
    poison_candidates = [
        (poison_texts[i], float(poison_scores[i]), f"poison::{i}")
        for i in range(len(poison_texts))
    ]

    merged = sorted(base_candidates + poison_candidates, key=lambda c: c[1], reverse=True)
    return merged[:k], poison_doc_ids


def render_poisoned_context(retrieved: list, corpus_name: str) -> str:
    """
    Mirrors harness.pipeline.build_rag_user_prompt's context rendering
    exactly (numbered [1], [2], ... lines) so a delta against the clean
    Phase 1 baseline isolates the poisoning's effect, not a formatting
    change. Poison entries are already final display text (built by
    attacks.poisonedrag.build_poisoned_passage) -- only real corpus records
    need extract_passage_text; a poison entry's "record" IS its text.
    """
    from data.normalize import extract_passage_text

    lines = []
    for i, (item, _score, doc_id) in enumerate(retrieved):
        text = item if isinstance(doc_id, str) else extract_passage_text(corpus_name, item)
        lines.append(f"[{i + 1}] {text}")
    return "\n\n".join(lines)
