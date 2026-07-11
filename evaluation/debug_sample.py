"""
Prints N raw (question, gold, generated_answer) triples per model x corpus
straight to stdout as JSON lines — for eyeballing answer formatting without
depending on the results file, which does not survive a container restart.

Motivated by the 2026-07-11 baseline run: qwen3-8b scored EM=0.088 / F1=0.43
on nq_open, a signature of verbose-but-correct answers rather than wrong
ones. metrics.exact_match compares the ENTIRE generation against gold, so
the raw strings are the only way to tell formatting failure from capability
failure.

Selected by the container entrypoint when RAG_MODE=debug. Respects the same
RAG_MODELS / RAG_CORPORA env filters as run_baseline; RAG_DEBUG_N sets the
sample size (default 20).
"""

import json
import os

import torch

from config import CORPORA, MODELS
from data.build_index import build_index
from data.normalize import extract_gold_answers, extract_question
from evaluation.metrics import contains_answer, exact_match, f1_score
from harness.model_loader import load_model
from harness.pipeline import run_query


def main():
    model_keys = [m.strip() for m in os.environ.get("RAG_MODELS", "").split(",") if m.strip()]
    model_keys = model_keys or list(MODELS)
    corpus_names = [c.strip() for c in os.environ.get("RAG_CORPORA", "").split(",") if c.strip()]
    corpus_names = corpus_names or list(CORPORA)
    n = int(os.environ.get("RAG_DEBUG_N", "20"))

    for model_key in model_keys:
        print(f"\n=== Loading {model_key} ===")
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f"[debug_sample] SKIPPING {model_key}: failed to load — "
                  f"{type(e).__name__}: {e}")
            continue

        for corpus_name in corpus_names:
            print(f"--- {model_key} x {corpus_name}: first {n} answers ---")
            index, records = build_index(corpus_name, split="dev")

            shown = 0
            for record in records:
                if shown >= n:
                    break
                question = extract_question(corpus_name, record)
                gold = extract_gold_answers(corpus_name, record)
                if not question or not gold:
                    continue

                result = run_query(
                    model, tokenizer, model_key, corpus_name, question,
                    index=index, records=records,
                )
                answer = result["generated_answer"]

                # json.dumps keeps newlines/tabs escaped, so the raw shape of
                # the answer (prefixes, multi-line prose) stays visible in logs.
                print(json.dumps({
                    "model": model_key,
                    "corpus": corpus_name,
                    "i": shown,
                    "question": question,
                    "gold": gold,
                    "generated_answer": answer,
                    "exact_match": exact_match(answer, gold),
                    "f1": round(f1_score(answer, gold), 4),
                    "contains_answer": contains_answer(answer, gold),
                }, ensure_ascii=False))
                shown += 1

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[debug_sample] done")


if __name__ == "__main__":
    main()
