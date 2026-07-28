"""
Phase 1 baseline sweep: all four models x all three corpora, clean queries only.

Produces, ONE PAIR PER (model, corpus, engine) CELL -- see
evaluation/result_paths.py for the exact naming, and its module docstring
for why: a single fixed filename silently overwrote the prior run's data
every time a different model/corpus/engine combination finished:
  results/baseline_raw_{model}_{corpus}_{engine}.jsonl    -- one line per
                                    question in that cell, the full
                                    generated answer + metrics, kept for
                                    the McNemar significance tests in Phase 4
  results/baseline_summary_{model}_{corpus}_{engine}.csv  -- that cell's
                                    aggregated metrics. f1_clean (F1 on the
                                    cleaned string) is the primary utility
                                    metric (headline number in every summary
                                    table); EM, also on the cleaned string,
                                    is secondary; f1_raw is kept for
                                    comparability with pre-cleanup runs;
                                    contains_answer is diagnostic only and
                                    never feeds a headline table.

Run scripts/aggregate_baseline_summaries.py once ALL cells you care about
have finished writing, to combine their per-cell summary CSVs into one
baseline_summary_all.csv -- not done live here, since a shared file every
concurrent/sequential run appends to would need its own append-safe write
pattern over the Network Volume, whose atomicity under concurrent access is
unverified (see data/build_index.py's atomic-write notes).

Run this AFTER data/loader.py and data/build_index.py have been run once
(or let it build indices on the fly the first time — slower, but works).
"""

import csv
import gc
import json
import os
import tempfile
import time
from contextlib import contextmanager

import torch

from config import CORPORA, MODELS, RESULTS_DIR, SEED, TOP_K, RAG_VLLM_MAX_MODEL_LEN
from data.build_index import build_index
from data.normalize import extract_gold_answers, extract_question
from evaluation.metrics import contains_answer, exact_match, f1_score
from evaluation.result_paths import resolve_sweep_selection, result_file_paths
from harness.model_loader import load_model
from harness.pipeline import SYSTEM_PROMPT, build_rag_user_prompt, clean_generation, run_query


@contextmanager
def _atomic_open(final_path, newline=None):
    """Open for writing via a temp file in the SAME directory, os.replace()d
    onto final_path only after the with-block completes without raising.

    Same rationale as data/build_index.py's atomic writes: the final path
    only ever holds nothing, the last complete version, or the new complete
    version — never a partial file. For the hours-long raw JSONL this means
    a crashed sweep leaves its partial rows in a .tmp_* file (recoverable by
    hand) instead of a truncated file at the path later tooling trusts.
    Explicit encoding: locale default (cp1252 on Windows) until PEP 686/3.15.
    """
    final_path = str(final_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(final_path), prefix=".tmp_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as f:
            yield f
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def run_baseline_sweep(model_keys=None, corpus_names=None, split="dev"):
    # RAG_MODELS/RAG_CORPORA/INFERENCE_ENGINE let a deployment restrict or
    # select a subset without a code change, e.g. RAG_MODELS="phi-4-mini"
    # for a smoke test, or one pod per cell for a parallel sweep. An explicit
    # function argument still wins over the env var, same precedence as
    # before this refactor -- resolve_sweep_selection() only supplies the
    # env-var-driven default, kept in evaluation/result_paths.py so this
    # module and scripts/run_and_terminate.py can never read it differently.
    default_model_keys, default_corpus_names, engine = resolve_sweep_selection()
    if model_keys is None:
        model_keys = default_model_keys
    if corpus_names is None:
        corpus_names = default_corpus_names

    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys {unknown}. Options: {list(MODELS)}")

    unknown_corpora = [c for c in corpus_names if c not in CORPORA]
    if unknown_corpora:
        raise ValueError(f"Unknown corpus names {unknown_corpora}. Options: {list(CORPORA)}")

    # INFERENCE_ENGINE selects the generation backend. "hf" (the default —
    # zero config changes needed) is the original per-question transformers
    # path below, byte-for-byte unchanged. "vllm" routes to a genuinely
    # different control flow: one batched generate() call per
    # (model, corpus) pair, which is where the throughput gain comes from.
    if engine not in ("hf", "vllm"):
        raise ValueError(
            f"INFERENCE_ENGINE must be 'hf' or 'vllm', got '{engine}'"
        )
    if engine == "vllm":
        return _run_vllm_sweep(model_keys, corpus_names, split)

    # Surface the compute device up front — a full sweep on CPU would take
    # days and should never start silently.
    if torch.cuda.is_available():
        print(f"[baseline] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda})")
    else:
        print("[baseline] WARNING: CUDA not available — the sweep will run on CPU. "
              "For 4 models x 3 corpora this is impractical. Install the CUDA torch "
              "wheel (see requirements.txt) or run on a GPU machine.")

    summary_rows = []

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

            # One raw JSONL + one summary CSV per (model, corpus, engine)
            # cell -- a single fixed filename for the whole sweep silently
            # overwrote the prior cell's results every time a different
            # model/corpus/engine combination finished (see
            # evaluation/result_paths.py).
            raw_path, summary_path = result_file_paths(
                RESULTS_DIR, model_key, corpus_name, engine
            )

            em_scores, f1_clean_scores, f1_raw_scores, ca_scores = [], [], [], []
            start = time.time()

            with _atomic_open(raw_path) as raw_f:
                for i, record in enumerate(records):
                    question = extract_question(corpus_name, record)
                    gold = extract_gold_answers(corpus_name, record)
                    if not question or not gold:
                        continue

                    result = run_query(
                        model, tokenizer, model_key, corpus_name, question,
                        index=index, records=records,
                    )

                    # f1_clean (primary) and EM (secondary) are scored on the
                    # cleaned string so they measure content, not formatting;
                    # f1_raw is kept for comparability with pre-cleanup runs;
                    # contains_answer is diagnostic only.
                    em = exact_match(result["generated_answer_clean"], gold)
                    f1_clean = f1_score(result["generated_answer_clean"], gold)
                    f1_raw = f1_score(result["generated_answer"], gold)
                    ca = contains_answer(result["generated_answer"], gold)
                    em_scores.append(em)
                    f1_clean_scores.append(f1_clean)
                    f1_raw_scores.append(f1_raw)
                    ca_scores.append(ca)

                    raw_f.write(json.dumps({
                        "model": model_key,
                        "corpus": corpus_name,
                        "seed": SEED,
                        "question": question,
                        "gold_answers": gold,
                        "generated_answer": result["generated_answer"],
                        "generated_answer_clean": result["generated_answer_clean"],
                        "f1_clean": f1_clean,
                        "f1_raw": f1_raw,
                        "exact_match": em,
                        "contains_answer_diagnostic": ca,
                        "retrieved_doc_ids": result["retrieved_doc_ids"],
                    }, ensure_ascii=False) + "\n")

                    if (i + 1) % 50 == 0:
                        print(f"  {i + 1}/{len(records)} done "
                              f"({time.time() - start:.0f}s elapsed)")

            mean_em = sum(em_scores) / len(em_scores) if em_scores else float("nan")
            mean_f1_clean = (sum(f1_clean_scores) / len(f1_clean_scores)
                             if f1_clean_scores else float("nan"))
            mean_f1_raw = (sum(f1_raw_scores) / len(f1_raw_scores)
                           if f1_raw_scores else float("nan"))
            mean_ca = sum(ca_scores) / len(ca_scores) if ca_scores else float("nan")

            summary_row = {
                "model": model_key,
                "corpus": corpus_name,
                "n": len(em_scores),
                "f1_clean": round(mean_f1_clean, 4),
                "exact_match": round(mean_em, 4),
                "f1_raw": round(mean_f1_raw, 4),
                "contains_answer_diagnostic": round(mean_ca, 4),
            }
            summary_rows.append(summary_row)

            with _atomic_open(summary_path, newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["model", "corpus", "n", "f1_clean", "exact_match",
                                   "f1_raw", "contains_answer_diagnostic"]
                )
                writer.writeheader()
                writer.writerow(summary_row)

            print(f"  {model_key} x {corpus_name}: "
                  f"F1(clean)={mean_f1_clean:.4f}  EM(clean)={mean_em:.4f}  "
                  f"F1(raw)={mean_f1_raw:.4f}  "
                  f"contains(diagnostic)={mean_ca:.4f}  n={len(em_scores)}")
            print(f"  raw -> {raw_path}")
            print(f"  summary -> {summary_path}")

        del model  # free memory before loading the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


def _resolve_vllm_max_model_len():
    """
    max_model_len is COMPUTED from real built prompts, never guessed — the
    4096 used during Colab testing was a T4-VRAM compromise, not a measured
    value. Run scripts/compute_max_model_len.py in an environment where the
    corpora/indices are available, then either set
    config.RAG_VLLM_MAX_MODEL_LEN or export RAG_VLLM_MAX_MODEL_LEN (the env
    var wins, for machine-specific overrides without a code change).

    RAG_ prefix (not VLLM_) because vLLM's own env var validator treats any
    VLLM_* name as reserved for its internal use — a bare VLLM_MAX_MODEL_LEN
    silently did nothing.
    """
    env_val = os.environ.get("RAG_VLLM_MAX_MODEL_LEN")
    if env_val:
        return int(env_val)
    if RAG_VLLM_MAX_MODEL_LEN:
        return RAG_VLLM_MAX_MODEL_LEN
    raise RuntimeError(
        "RAG_VLLM_MAX_MODEL_LEN is not set. This value must be computed "
        "from real prompts, not guessed: run scripts/compute_max_model_len.py "
        "where the corpora/indices exist, then set "
        "config.RAG_VLLM_MAX_MODEL_LEN or export RAG_VLLM_MAX_MODEL_LEN."
    )


def _run_vllm_sweep(model_keys, corpus_names, split):
    """
    vLLM control flow: for each (model, corpus) pair, retrieval runs
    per-question as usual, but ALL prompts for the corpus go to vLLM in a
    single generate_batch() call — one llm.generate() per corpus per model,
    not one per question. Metrics and JSONL rows are computed from the
    returned list, keeping the exact schema of the HF path above.

    The scoring/JSONL/summary blocks below deliberately mirror the HF loop
    line-for-line (keep them in lockstep when editing either; parity of the
    metric layer over plain string lists is guarded by
    scripts/verify_engine_parity.py).
    """
    # Imported here, not at module top, so INFERENCE_ENGINE=hf keeps working
    # on machines without vllm installed (it is Linux/GPU-only).
    from harness.vllm_engine import load_vllm_model, generate_batch

    max_model_len = _resolve_vllm_max_model_len()

    if torch.cuda.is_available():
        print(f"[baseline] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda}) — engine: vLLM, "
              f"max_model_len={max_model_len}")
    else:
        print("[baseline] WARNING: CUDA not available — vLLM requires a GPU; "
              "the load below is expected to fail on this machine.")

    summary_rows = []

    for model_key in model_keys:
        print(f"\n=== Loading {model_key} (vLLM) ===")
        # Deliberately NOT wrapped in try/except, unlike the HF loop: a
        # failed vLLM load leaves GPU/NCCL state dirty, so loading the
        # next model in this same process would fail confusingly anyway.
        # Let it crash and let the container restart policy handle it —
        # see the design note in harness/vllm_engine.py.
        llm = load_vllm_model(model_key, max_model_len=max_model_len)

        for corpus_name in corpus_names:
            print(f"--- {model_key} x {corpus_name} (vLLM, batched) ---")
            index, records = build_index(corpus_name, split=split)

            # One raw JSONL + one summary CSV per (model, corpus, engine)
            # cell -- same fix, same reason as the HF loop above (see
            # evaluation/result_paths.py).
            raw_path, summary_path = result_file_paths(
                RESULTS_DIR, model_key, corpus_name, "vllm"
            )

            # Pass 1: retrieval + prompt construction for every valid
            # question, identical filtering to the HF loop.
            questions, golds, user_prompts, retrieved_ids = [], [], [], []
            for record in records:
                question = extract_question(corpus_name, record)
                gold = extract_gold_answers(corpus_name, record)
                if not question or not gold:
                    continue
                user_prompt, retrieved = build_rag_user_prompt(
                    index, records, question, top_k=TOP_K
                )
                questions.append(question)
                golds.append(gold)
                user_prompts.append(user_prompt)
                retrieved_ids.append([idx for idx, _ in enumerate(retrieved)])

            # Pass 2: one batched generate call for the whole corpus.
            start = time.time()
            print(f"  sending {len(user_prompts)} prompts in one batch")
            outputs = generate_batch(llm, model_key, SYSTEM_PROMPT, user_prompts)
            print(f"  batch generated in {time.time() - start:.0f}s")

            # Pass 3: score + write rows — same fields, same metric
            # hierarchy as the HF loop (f1_clean primary, EM secondary,
            # f1_raw comparability, contains_answer diagnostic).
            em_scores, f1_clean_scores, f1_raw_scores, ca_scores = [], [], [], []
            with _atomic_open(raw_path) as raw_f:
                for question, gold, doc_ids, generated in zip(
                    questions, golds, retrieved_ids, outputs
                ):
                    generated = generated.strip()  # parity with run_query's decode-then-strip
                    generated_clean = clean_generation(generated)

                    em = exact_match(generated_clean, gold)
                    f1_clean = f1_score(generated_clean, gold)
                    f1_raw = f1_score(generated, gold)
                    ca = contains_answer(generated, gold)
                    em_scores.append(em)
                    f1_clean_scores.append(f1_clean)
                    f1_raw_scores.append(f1_raw)
                    ca_scores.append(ca)

                    raw_f.write(json.dumps({
                        "model": model_key,
                        "corpus": corpus_name,
                        "seed": SEED,
                        "question": question,
                        "gold_answers": gold,
                        "generated_answer": generated,
                        "generated_answer_clean": generated_clean,
                        "f1_clean": f1_clean,
                        "f1_raw": f1_raw,
                        "exact_match": em,
                        "contains_answer_diagnostic": ca,
                        "retrieved_doc_ids": doc_ids,
                    }, ensure_ascii=False) + "\n")

            mean_em = sum(em_scores) / len(em_scores) if em_scores else float("nan")
            mean_f1_clean = (sum(f1_clean_scores) / len(f1_clean_scores)
                             if f1_clean_scores else float("nan"))
            mean_f1_raw = (sum(f1_raw_scores) / len(f1_raw_scores)
                           if f1_raw_scores else float("nan"))
            mean_ca = sum(ca_scores) / len(ca_scores) if ca_scores else float("nan")

            summary_row = {
                "model": model_key,
                "corpus": corpus_name,
                "n": len(em_scores),
                "f1_clean": round(mean_f1_clean, 4),
                "exact_match": round(mean_em, 4),
                "f1_raw": round(mean_f1_raw, 4),
                "contains_answer_diagnostic": round(mean_ca, 4),
            }
            summary_rows.append(summary_row)

            with _atomic_open(summary_path, newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["model", "corpus", "n", "f1_clean", "exact_match",
                                   "f1_raw", "contains_answer_diagnostic"]
                )
                writer.writeheader()
                writer.writerow(summary_row)

            print(f"  {model_key} x {corpus_name}: "
                  f"F1(clean)={mean_f1_clean:.4f}  EM(clean)={mean_em:.4f}  "
                  f"F1(raw)={mean_f1_raw:.4f}  "
                  f"contains(diagnostic)={mean_ca:.4f}  n={len(em_scores)}")
            print(f"  raw -> {raw_path}")
            print(f"  summary -> {summary_path}")

        # Best-effort teardown between models. Known vLLM weak point:
        # multi-model-per-process reuse is not fully reliable even after
        # successful runs — Stage 3's Job design should isolate one
        # model per container instead of relying on this.
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


if __name__ == "__main__":
    run_baseline_sweep()
