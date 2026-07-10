"""
Standard SQuAD-style Exact Match / F1, plus Recall@k for retrieval quality.
"""

import re
import string
from collections import Counter


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: list[str]) -> int:
    pred = normalize_text(prediction)
    return int(any(pred == normalize_text(g) for g in gold_answers))


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


def recall_at_k(retrieved_doc_ids: list[int], relevant_doc_ids: set[int], k: int = 5) -> float:
    if not relevant_doc_ids:
        return float("nan")  # undefined when there's no labelled relevant doc to check against
    hit = any(doc_id in relevant_doc_ids for doc_id in retrieved_doc_ids[:k])
    return float(hit)
