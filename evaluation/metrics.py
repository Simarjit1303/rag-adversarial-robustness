import re
import string
from collections import Counter

def normalize_text(s: str) -> str:
    s = s.lower()
    s = ''.join((' ' if ch in string.punctuation else ch for ch in s))
    s = re.sub('\\b(a|an|the)\\b', ' ', s)
    return ' '.join(s.split())

def exact_match(prediction: str, gold_answers: list[str]) -> int:
    pred = normalize_text(prediction)
    return int(any((pred == normalize_text(g) for g in gold_answers)))

def f1_score(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best

def contains_answer(prediction: str, gold_answers: list[str]) -> int:
    pred = normalize_text(prediction)
    return int(any((normalize_text(g) in pred for g in gold_answers)))

def recall_at_k(retrieved_doc_ids: list[int], relevant_doc_ids: set[int], k: int=5) -> float:
    if not relevant_doc_ids:
        return float('nan')
    hit = any((doc_id in relevant_doc_ids for doc_id in retrieved_doc_ids[:k]))
    return float(hit)

def retrieval_f1_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int=5) -> dict:
    if not relevant_doc_ids:
        return {'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')}
    retrieved_at_k = retrieved_doc_ids[:k]
    hits = sum((1 for doc_id in retrieved_at_k if doc_id in relevant_doc_ids))
    precision = hits / len(retrieved_at_k) if retrieved_at_k else 0.0
    recall = hits / len(relevant_doc_ids)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}
