"""
Phase 1 baseline sweep: all four models x all three corpora, clean queries only.

Produces:
  results/baseline_raw.jsonl   -- one line per (model, corpus, question), the
                                    full generated answer + metrics, kept for
                                    the McNemar significance tests in Phase 4
  results/baseline_summary.csv -- aggregated EM / F1 per model x corpus

Run this AFTER data/loader.py and data/build_index.py have been run once
(or let it build indices on the fly the first time — slower, but works).
"""

import csv
import json
import os
import time

import torch

from config import CORPORA, MODELS, RESULTS_DIR, SEED
from data.build_index import build_index
from data.normalize import extract_gold_answers, extract_question
from evaluation.metrics import contains_answer, exact_match, f1_score
from harness.model_loader import load_model
from harness.pipeline import run_query


def run_baseline_sweep(model_keys=None, corpus_names=None, split="dev"):
    # RAG_MODELS lets a deployment pick models without a code change, e.g.
    # RAG_MODELS="qwen3-8b,phi-4-mini" — useful when a gated model (Llama)
    # is still awaiting HF access approval.
    if model_keys is None and os.environ.get("RAG_MODELS"):
        model_keys = [m.strip() for m in os.environ["RAG_MODELS"].split(",") if m.strip()]
    model_keys = model_keys or list(MODELS)
    corpus_names = corpus_names or list(CORPORA)

    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys {unknown}. Options: {list(MODELS)}")

    # Surface the compute device up front — a full sweep on CPU would take
    # days and should never start silently.
    if torch.cuda.is_available():
        print(f"[baseline] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda})")
    else:
        print("[baseline] WARNING: CUDA not available — the sweep will run on CPU. "
              "For 4 models x 3 corpora this is impractical. Install the CUDA torch "
              "wheel (see requirements.txt) or run on a GPU machine.")

    raw_path = RESULTS_DIR / "baseline_raw.jsonl"
    summary_rows = []

    # Explicit encoding: Python 3.14 still defaults to the locale encoding
    # (cp1252 on Windows), which would corrupt non-ASCII model output.
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for model_key in model_keys:
            print(f"\n=== Loading {model_key} ===")
            try:
                model, tokenizer = load_model(model_key)
            except Exception as e:
                # One model failing to load (gated repo, missing class, OOM)
                # must not kill the whole paid sweep — skip and keep going.
                print(f"[baseline] SKIPPING {model_key}: failed to load — "
                      f"{type(e).__name__}: {e}")
                continue

            for corpus_name in corpus_names:
                print(f"--- {model_key} x {corpus_name} ---")
                index, records = build_index(corpus_name, split=split)

                em_scores, f1_scores, ca_scores = [], [], []
                start = time.time()

                for i, record in enumerate(records):
                    question = extract_question(corpus_name, record)
                    gold = extract_gold_answers(corpus_name, record)
                    if not question or not gold:
                        continue

                    result = run_query(
                        model, tokenizer, model_key, corpus_name, question,
                        index=index, records=records,
                    )

                    em = exact_match(result["generated_answer"], gold)
                    f1 = f1_score(result["generated_answer"], gold)
                    ca = contains_answer(result["generated_answer"], gold)
                    em_scores.append(em)
                    f1_scores.append(f1)
                    ca_scores.append(ca)

                    raw_f.write(json.dumps({
                        "model": model_key,
                        "corpus": corpus_name,
                        "seed": SEED,
                        "question": question,
                        "gold_answers": gold,
                        "generated_answer": result["generated_answer"],
                        "exact_match": em,
                        "f1": f1,
                        "contains_answer": ca,
                        "retrieved_doc_ids": result["retrieved_doc_ids"],
                    }, ensure_ascii=False) + "\n")

                    if (i + 1) % 50 == 0:
                        print(f"  {i + 1}/{len(records)} done "
                              f"({time.time() - start:.0f}s elapsed)")

                mean_em = sum(em_scores) / len(em_scores) if em_scores else float("nan")
                mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else float("nan")
                mean_ca = sum(ca_scores) / len(ca_scores) if ca_scores else float("nan")

                summary_rows.append({
                    "model": model_key,
                    "corpus": corpus_name,
                    "n": len(em_scores),
                    "exact_match": round(mean_em, 4),
                    "f1": round(mean_f1, 4),
                    "contains_answer": round(mean_ca, 4),
                })

                print(f"  {model_key} x {corpus_name}: "
                      f"EM={mean_em:.4f}  F1={mean_f1:.4f}  "
                      f"Contains={mean_ca:.4f}  n={len(em_scores)}")

            del model  # free memory before loading the next model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = RESULTS_DIR / "baseline_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "corpus", "n", "exact_match", "f1", "contains_answer"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nRaw per-question results -> {raw_path}")
    print(f"Summary table -> {summary_path}")
    return summary_rows


if __name__ == "__main__":
    run_baseline_sweep()
