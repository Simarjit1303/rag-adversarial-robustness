import numpy as np
from data.build_index import _get_embedder

def embed_poison_texts(poison_texts: list[str]) -> np.ndarray:
    model = _get_embedder()
    embeddings = model.encode(poison_texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype='float32')

def retrieve_with_poison(base_index, base_records, poison_texts: list[str], poison_embeddings: np.ndarray, query: str, k: int=5):
    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    base_scores, base_idxs = base_index.search(q_emb, k)
    base_candidates = [(base_records[i], float(base_scores[0][rank]), int(i)) for rank, i in enumerate(base_idxs[0])]
    poison_doc_ids = {f'poison::{i}' for i in range(len(poison_texts))}
    poison_scores = poison_embeddings @ q_emb[0]
    poison_candidates = [(poison_texts[i], float(poison_scores[i]), f'poison::{i}') for i in range(len(poison_texts))]
    merged = sorted(base_candidates + poison_candidates, key=lambda c: c[1], reverse=True)
    return (merged[:k], poison_doc_ids)

def render_poisoned_context(retrieved: list, corpus_name: str) -> str:
    from data.normalize import extract_passage_text
    lines = []
    for i, (item, _score, doc_id) in enumerate(retrieved):
        text = item if isinstance(doc_id, str) else extract_passage_text(corpus_name, item)
        lines.append(f'[{i + 1}] {text}')
    return '\n\n'.join(lines)
