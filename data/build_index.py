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

def _atomic_write_index(index, final_path):
    final_path = str(final_path)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix='.tmp_', suffix='.faiss')
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
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix='.tmp_', suffix='.pkl')
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def _mark_build_complete(marker_path, revision):
    marker_path = str(marker_path)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(marker_path), prefix='.tmp_', suffix='.marker')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
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
    with open(marker_path, encoding='utf-8') as f:
        return f.read().strip() == expected_revision

def _get_embedder():
    global _embedder
    if _embedder is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('[index] WARNING: CUDA not available — embedding on CPU. This works but is slow for 10k-doc corpora; see requirements.txt for the CUDA-enabled torch install.')
        else:
            print(f'[index] Embedding on GPU: {torch.cuda.get_device_name(0)}')
        last_exc = None
        for attempt in range(3):
            try:
                _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
                break
            except Exception as e:
                last_exc = e
                print(f'[index] embedder init attempt {attempt + 1} failed: {e!r}' + (' -- retrying' if attempt < 2 else ' -- giving up'), file=sys.stderr)
                if attempt < 2:
                    time.sleep(2)
        else:
            raise last_exc
    return _embedder

def build_index(corpus_name: str, split: str='dev', force_rebuild: bool=False):
    index_path = INDEX_DIR / f'{corpus_name}_{split}.faiss'
    meta_path = INDEX_DIR / f'{corpus_name}_{split}_meta.pkl'
    marker_path = INDEX_DIR / f'{corpus_name}_{split}.build_complete'
    revision = CORPORA[corpus_name]['revision']
    if _is_build_complete(marker_path, revision) and (not force_rebuild):
        index = faiss.read_index(str(index_path))
        with meta_path.open('rb') as f:
            records = pickle.load(f)
        return (index, records)
    if marker_path.exists():
        os.remove(marker_path)
    records = load_corpus(corpus_name, split=split)
    texts = [extract_passage_text(corpus_name, r) for r in records]
    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype='float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    _atomic_write_pickle(records, meta_path)
    _atomic_write_index(index, index_path)
    _mark_build_complete(marker_path, revision)
    print(f'[index] {corpus_name}/{split}: {len(records)} vectors -> {index_path}')
    return (index, records)

def retrieve(index, records, query: str, k: int=5):
    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(q_emb, k)
    return [(records[i], float(scores[0][rank])) for rank, i in enumerate(idxs[0])]

def main() -> None:
    corpora_env = os.environ.get('RAG_CORPORA')
    corpus_names = [c.strip() for c in corpora_env.split(',') if c.strip()] if corpora_env else list(CORPORA)
    unknown_corpora = [c for c in corpus_names if c not in CORPORA]
    if unknown_corpora:
        raise ValueError(f'Unknown corpus names {unknown_corpora}. Options: {list(CORPORA)}')
    for corpus_name in corpus_names:
        build_index(corpus_name, split='dev')
if __name__ == '__main__':
    main()
