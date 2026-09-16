"""
Phase 3 pre-sweep validation: instruction-detection false-positive rate on
CLEAN (non-adversarial) text.

Runs the real protectai/deberta-v3-base-prompt-injection-v2 classifier (via
defenses.instruction_detection.detect_injection with no `classifier`
override -- the one real local model load this repo's hard local-execution
constraint explicitly allows, see CLAUDE.md) against:
  - a sample of clean hotpot_qa and ms_marco passages (this repo's two
    attack-eligible RAG corpora -- data.loader.load_corpus, no embedding
    model needed, so this never touches sentence-transformers/faiss)
  - XSTest (config.XSTEST_HF_ID), if reachable this session
  - OR-Bench-Hard-1K (bench-llm/or-bench, "or-bench-hard-1k" config), if
    reachable this session

Neither XSTest nor OR-Bench-Hard-1K has an existing loader in this repo
(defenses/instruction_detection.py's PassageLog docstring names both as the
intended false-positive benchmarks but nothing loads them yet) -- loaded
directly via datasets.load_dataset here, best-effort: a network/gating
failure is caught and reported, not fatal to the corpus-side check.

A flag rate above ~10-15% on genuinely clean text is a real concern: it
means a defended sweep would be losing a meaningful fraction of legitimate
retrieved context to false positives, confounding any utility-cost
comparison against the undefended baseline.
"""

import json
import sys

from config import CORPORA, XSTEST_HF_ID
from data.loader import load_corpus
from data.normalize import extract_passage_text
from defenses.instruction_detection import detect_injection

SAMPLE_SIZE = 150


def _flag_rate(texts, label):
    n_flagged = 0
    scores = []
    for text in texts:
        if not text:
            continue
        result = detect_injection(text)
        scores.append(result.score)
        if result.flagged:
            n_flagged += 1
    n = len(scores)
    rate = n_flagged / n if n else float("nan")
    print(f"[fp-check] {label}: n={n}  flagged={n_flagged}  flag_rate={rate:.4f}")
    return {"label": label, "n": n, "flagged": n_flagged, "flag_rate": rate}


def check_corpus(corpus_name, sample_size=SAMPLE_SIZE):
    records = load_corpus(corpus_name, split="dev")[:sample_size]
    texts = [extract_passage_text(corpus_name, r) for r in records]
    return _flag_rate(texts, f"{corpus_name} (clean passages)")


def check_xstest(sample_size=SAMPLE_SIZE):
    try:
        from datasets import load_dataset
        ds = load_dataset(XSTEST_HF_ID, split="train")
    except Exception as e:
        print(f"[fp-check] XSTest not accessible this session: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    text_col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    texts = ds[text_col][:sample_size]
    return _flag_rate(texts, "XSTest (safe prompts)")


def check_or_bench_hard_1k(sample_size=SAMPLE_SIZE):
    try:
        from datasets import load_dataset
        ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    except Exception as e:
        print(f"[fp-check] OR-Bench-Hard-1K not accessible this session: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    text_col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    texts = ds[text_col][:sample_size]
    return _flag_rate(texts, "OR-Bench-Hard-1K (hard-but-safe prompts)")


def main():
    results = []
    for corpus_name in ("hotpot_qa", "ms_marco"):
        assert corpus_name in CORPORA
        results.append(check_corpus(corpus_name))
    xstest_result = check_xstest()
    if xstest_result:
        results.append(xstest_result)
    or_bench_result = check_or_bench_hard_1k()
    if or_bench_result:
        results.append(or_bench_result)

    print("\n[fp-check] summary:")
    for r in results:
        flag = " ** ABOVE 10-15% SANITY BAR **" if r["flag_rate"] > 0.15 else ""
        print(f"  {r['label']}: {r['flag_rate']:.2%} ({r['flagged']}/{r['n']}){flag}")

    with open("instruction_detection_fp_check_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[fp-check] results -> instruction_detection_fp_check_results.json")


if __name__ == "__main__":
    main()
