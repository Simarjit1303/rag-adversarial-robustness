import json
import numpy as np
import pytest
import attacks.poisoned_retrieval as pr

class _FakeEmbedder:

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        return np.array([self._vectors[t] for t in texts], dtype='float32')

class _FakeIndex:

    def __init__(self, doc_vectors: list):
        self._doc_vectors = np.array(doc_vectors, dtype='float32')

    def search(self, q_emb, k):
        scores = self._doc_vectors @ q_emb[0]
        order = np.argsort(-scores)[:k]
        return (scores[order].reshape(1, -1), order.reshape(1, -1))

@pytest.fixture
def stub_embedder(monkeypatch):

    def _install(vectors):
        monkeypatch.setattr(pr, '_get_embedder', lambda: _FakeEmbedder(vectors))
    return _install

def test_poison_texts_that_score_higher_than_base_docs_are_retrieved(stub_embedder):
    query = 'query'
    base_records = ['real0', 'real1', 'real2']
    base_vectors = [[1.0, 0.0], [0.5, 0.0], [0.1, 0.0]]
    poison_texts = ['poison0', 'poison1']
    stub_embedder({query: [0.1, 0.0], 'poison0': [10.0, 0.0], 'poison1': [0.05, 0.0]})
    base_index = _FakeIndex(base_vectors)
    poison_embeddings = pr.embed_poison_texts(poison_texts)
    retrieved, poison_doc_ids = pr.retrieve_with_poison(base_index, base_records, poison_texts, poison_embeddings, query, k=3)
    assert poison_doc_ids == {'poison::0', 'poison::1'}
    assert retrieved[0][2] == 'poison::0'
    assert retrieved[0][0] == 'poison0'
    scores = [item[1] for item in retrieved]
    assert scores == sorted(scores, reverse=True)

def test_no_poison_texts_score_high_enough_none_appear_in_top_k(stub_embedder):
    query = 'query'
    base_records = ['real0', 'real1']
    base_vectors = [[1.0, 0.0], [0.9, 0.0]]
    poison_texts = ['poison0']
    stub_embedder({query: [1.0, 0.0], 'poison0': [0.01, 0.0]})
    base_index = _FakeIndex(base_vectors)
    poison_embeddings = pr.embed_poison_texts(poison_texts)
    retrieved, poison_doc_ids = pr.retrieve_with_poison(base_index, base_records, poison_texts, poison_embeddings, query, k=2)
    retrieved_ids = {item[2] for item in retrieved}
    assert retrieved_ids.isdisjoint(poison_doc_ids)

def test_retrieved_base_doc_ids_are_json_serializable(stub_embedder):
    query = 'query'
    base_records = ['real0', 'real1']
    base_vectors = [[1.0, 0.0], [0.5, 0.0]]
    poison_texts = ['poison0']
    stub_embedder({query: [1.0, 0.0], 'poison0': [0.01, 0.0]})
    base_index = _FakeIndex(base_vectors)
    poison_embeddings = pr.embed_poison_texts(poison_texts)
    retrieved, _poison_doc_ids = pr.retrieve_with_poison(base_index, base_records, poison_texts, poison_embeddings, query, k=2)
    base_doc_ids = [doc_id for _, _, doc_id in retrieved if isinstance(doc_id, int)]
    assert base_doc_ids
    for doc_id in base_doc_ids:
        assert type(doc_id) is int
    json.dumps({'retrieved_doc_ids': base_doc_ids})

def test_render_poisoned_context_numbers_lines_and_uses_poison_text_directly():
    retrieved = [('poison passage text', 0.9, 'poison::0')]
    context = pr.render_poisoned_context(retrieved, corpus_name='hotpot_qa')
    assert context == '[1] poison passage text'
