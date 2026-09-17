"""
Covers attacks/poisoned_retrieval.py's score-merge logic: does searching a
fake base index + a handful of fake poison embeddings and merging by score
correctly reproduce "as if" everything were one combined index. Stubs
_get_embedder (no real SentenceTransformer / network / GPU in tests).
"""

import json

import numpy as np
import pytest

import attacks.poisoned_retrieval as pr


class _FakeEmbedder:
    """Deterministic: encode() returns a fixed 1D vector per known string,
    letting a test control exactly which "documents" score highest."""

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        return np.array([self._vectors[t] for t in texts], dtype="float32")


class _FakeIndex:
    """Mimics faiss.IndexFlatIP.search(q_emb, k) -> (scores, idxs)."""

    def __init__(self, doc_vectors: list):
        self._doc_vectors = np.array(doc_vectors, dtype="float32")

    def search(self, q_emb, k):
        scores = self._doc_vectors @ q_emb[0]
        order = np.argsort(-scores)[:k]
        return scores[order].reshape(1, -1), order.reshape(1, -1)


@pytest.fixture
def stub_embedder(monkeypatch):
    def _install(vectors):
        monkeypatch.setattr(pr, "_get_embedder", lambda: _FakeEmbedder(vectors))
    return _install


def test_poison_texts_that_score_higher_than_base_docs_are_retrieved(stub_embedder):
    query = "query"
    base_records = ["real0", "real1", "real2"]
    base_vectors = [[1.0, 0.0], [0.5, 0.0], [0.1, 0.0]]  # scores vs query below: 0.1, 0.05, 0.01
    poison_texts = ["poison0", "poison1"]

    stub_embedder({
        query: [0.1, 0.0],
        "poison0": [10.0, 0.0],   # score = 1.0, dominates
        "poison1": [0.05, 0.0],   # score = 0.005, near the bottom
    })
    base_index = _FakeIndex(base_vectors)

    poison_embeddings = pr.embed_poison_texts(poison_texts)
    retrieved, poison_doc_ids = pr.retrieve_with_poison(
        base_index, base_records, poison_texts, poison_embeddings, query, k=3
    )

    assert poison_doc_ids == {"poison::0", "poison::1"}
    # top-1 must be the dominant poison text
    assert retrieved[0][2] == "poison::0"
    assert retrieved[0][0] == "poison0"
    # merged set is sorted by score, highest first, across both sources
    scores = [item[1] for item in retrieved]
    assert scores == sorted(scores, reverse=True)


def test_no_poison_texts_score_high_enough_none_appear_in_top_k(stub_embedder):
    query = "query"
    base_records = ["real0", "real1"]
    base_vectors = [[1.0, 0.0], [0.9, 0.0]]
    poison_texts = ["poison0"]

    stub_embedder({
        query: [1.0, 0.0],
        "poison0": [0.01, 0.0],  # scores far below both real docs
    })
    base_index = _FakeIndex(base_vectors)

    poison_embeddings = pr.embed_poison_texts(poison_texts)
    retrieved, poison_doc_ids = pr.retrieve_with_poison(
        base_index, base_records, poison_texts, poison_embeddings, query, k=2
    )

    retrieved_ids = {item[2] for item in retrieved}
    assert retrieved_ids.isdisjoint(poison_doc_ids)  # verified, not assumed


def test_retrieved_base_doc_ids_are_json_serializable(stub_embedder):
    # Regression: faiss.IndexFlatIP.search() (mimicked by _FakeIndex via
    # np.argsort) returns numpy.int64 indices, not native int. A base-record
    # doc_id left uncast passes every in-memory check (indexing, equality,
    # set membership) fine but raises "TypeError: Object of type int64 is
    # not JSON serializable" the moment a caller (evaluation/run_poisonedrag.
    # py's build_poisoned_contexts) puts it in a dict and json.dumps that
    # dict -- exactly what happened building fresh ms_marco contexts. A dict
    # equality/isinstance check on the in-memory tuple would NOT catch this;
    # only an actual json.dumps() call does.
    query = "query"
    base_records = ["real0", "real1"]
    base_vectors = [[1.0, 0.0], [0.5, 0.0]]
    poison_texts = ["poison0"]

    stub_embedder({
        query: [1.0, 0.0],
        "poison0": [0.01, 0.0],  # scores below both real docs -- base doc_ids dominate top-k
    })
    base_index = _FakeIndex(base_vectors)

    poison_embeddings = pr.embed_poison_texts(poison_texts)
    retrieved, _poison_doc_ids = pr.retrieve_with_poison(
        base_index, base_records, poison_texts, poison_embeddings, query, k=2
    )

    base_doc_ids = [doc_id for _, _, doc_id in retrieved if isinstance(doc_id, int)]
    assert base_doc_ids  # sanity: this run actually produced base-record hits
    for doc_id in base_doc_ids:
        assert type(doc_id) is int  # not numpy.int64 -- isinstance(np.int64(0), int) is False anyway, but be explicit
    json.dumps({"retrieved_doc_ids": base_doc_ids})  # raises TypeError pre-fix


def test_render_poisoned_context_numbers_lines_and_uses_poison_text_directly():
    # doc_id is a string ("poison::0") for a poison entry, an int for a real
    # one -- render_poisoned_context must route each through the right path
    retrieved = [
        ("poison passage text", 0.9, "poison::0"),
    ]
    context = pr.render_poisoned_context(retrieved, corpus_name="hotpot_qa")
    assert context == "[1] poison passage text"
