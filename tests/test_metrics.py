"""
Covers evaluation/metrics.py's retrieval_f1_at_k -- PoisonedRAG's
retrieval-verification metric (Meeting 3 checkpoint, PHASE2_ROADMAP.md).
The rest of this module (exact_match/f1_score/contains_answer/recall_at_k)
predates this test file and isn't retroactively covered here -- out of
scope for this addition.
"""

from evaluation.metrics import retrieval_f1_at_k


def test_all_five_poison_texts_retrieved_gives_f1_one():
    r = retrieval_f1_at_k(["p0", "p1", "p2", "p3", "p4"], {"p0", "p1", "p2", "p3", "p4"}, k=5)
    assert r == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_none_retrieved_gives_f1_zero():
    r = retrieval_f1_at_k(["r0", "r1", "r2", "r3", "r4"], {"p0", "p1", "p2", "p3", "p4"}, k=5)
    assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_partial_hit_precision_recall_and_f1_match_by_hand():
    # 2 of the 5 retrieved slots are poison, poison set has 5 members --
    # precision = 2/5, recall = 2/5, f1 equals that common value
    retrieved = ["p0", "r1", "p1", "r3", "r4"]
    r = retrieval_f1_at_k(retrieved, {"p0", "p1", "p2", "p3", "p4"}, k=5)
    assert abs(r["precision"] - 0.4) < 1e-9
    assert abs(r["recall"] - 0.4) < 1e-9
    assert abs(r["f1"] - 0.4) < 1e-9


def test_empty_relevant_set_returns_nan_not_a_crash():
    r = retrieval_f1_at_k(["r0", "r1"], set(), k=5)
    assert r["precision"] != r["precision"]  # NaN != NaN
