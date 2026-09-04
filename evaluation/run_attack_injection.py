"""
Phase 2, Attack 1 of 3: indirect prompt injection sweep.

Produces, ONE PAIR PER (model, corpus, injection_template, engine) CELL --
see evaluation/result_paths.py's attack_result_file_paths for the exact
naming, extending Phase 1's baseline_raw_{model}_{corpus}_{engine} pattern
with the injection_template axis:
  results/attack_raw_{model}_{corpus}_{injection_template}_{engine}.jsonl
  results/attack_summary_{model}_{corpus}_{injection_template}_{engine}.csv

Same atomic-write pattern as evaluation/run_baseline.py's _atomic_open --
no exceptions (that module's own docstring explains why: a fixed/collided
filename silently overwrites a prior cell's results on the shared Network
Volume).

nq_open is excluded by default (evaluation.result_paths.ATTACK_ELIGIBLE_CORPORA)
-- see nq_open_leakage_finding.md. This sweep runs against hotpot_qa and
ms_marco.

Utility metrics (f1_clean/exact_match/f1_raw/contains_answer_diagnostic)
are scored against the ORIGINAL question's gold answers, exactly as
evaluation.metrics scores the Phase 1 baseline -- this is what makes a
per-question utility-under-attack comparison against baseline_raw_*.jsonl
meaningful (evaluation.stats.paired_bootstrap_ci, McNemar). attack_success
is the separate, attack-specific metric (attacks.asr_scoring.score_asr
against the injected template's target_string) -- did the injected
instruction get followed, independent of whether the original question
also got answered correctly.
"""

import csv
import gc
import json
import os
import tempfile
import time
from contextlib import contextmanager

import torch

from attacks.asr_scoring import score_asr
from attacks.indirect_injection import build_attack_user_prompt, run_attack_query
from attacks.injection_templates import TEMPLATES
from config import MODELS, RESULTS_DIR, SEED, TOP_K, RAG_VLLM_MAX_MODEL_LEN
from data.build_index import build_index
from data.normalize import extract_gold_answers, extract_question
from evaluation.metrics import contains_answer, exact_match, f1_score
from evaluation.result_paths import ATTACK_ELIGIBLE_CORPORA, attack_result_file_paths, resolve_attack_sweep_selection
from harness.model_loader import load_model
from harness.pipeline import SYSTEM_PROMPT, clean_generation


@contextmanager
def _atomic_open(final_path, newline=None):
    """Identical to evaluation.run_baseline._atomic_open -- not imported
    from there to avoid coupling this module's atomicity to a baseline-sweep
    internal, but kept byte-for-byte in sync. See that module's docstring
    for the full rationale (crash leaves a recoverable .tmp_* file, never a
    truncated file at the trusted path)."""
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


def run_attack_sweep(model_keys=None, corpus_names=None, injection_templates=None,
                      split="dev", sample_n=None):
    """
    sample_n: cap the number of QUESTIONS ACTUALLY PROCESSED per (model,
    corpus, injection_template) cell (falls back to the RAG_SAMPLE_N env
    var, None/unset = the full dev/eval slice). Exists for smoke-testing
    real infrastructure (a new RunPod pod, a fresh container image) on a
    handful of questions before committing the full sweep's GPU budget --
    same lesson as Phase 1's own RunPod experience: verify on real
    hardware first, don't trust code review alone. Counted post
    question/gold filtering, so it is an approximate cap (matches
    scripts/compute_max_model_len.py's --sample semantics), not an exact
    slice of the raw corpus.
    """
    if sample_n is None:
        env_val = os.environ.get("RAG_SAMPLE_N")
        sample_n = int(env_val) if env_val else None

    default_model_keys, default_corpus_names, default_templates, engine = (
        resolve_attack_sweep_selection()
    )
    if model_keys is None:
        model_keys = default_model_keys
    if corpus_names is None:
        corpus_names = default_corpus_names
    if injection_templates is None:
        injection_templates = default_templates

    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys {unknown}. Options: {list(MODELS)}")

    unknown_corpora = [c for c in corpus_names if c not in ATTACK_ELIGIBLE_CORPORA]
    if unknown_corpora:
        raise ValueError(
            f"Unknown or excluded corpus names {unknown_corpora}. "
            f"Options: {ATTACK_ELIGIBLE_CORPORA} (nq_open is excluded -- "
            f"see nq_open_leakage_finding.md)"
        )

    unknown_templates = [t for t in injection_templates if t not in TEMPLATES]
    if unknown_templates:
        raise ValueError(
            f"Unknown injection templates {unknown_templates}. Options: {list(TEMPLATES)}"
        )

    if engine not in ("hf", "vllm"):
        raise ValueError(f"INFERENCE_ENGINE must be 'hf' or 'vllm', got '{engine}'")
    if engine == "vllm":
        return _run_vllm_attack_sweep(model_keys, corpus_names, injection_templates, split, sample_n)

    if torch.cuda.is_available():
        print(f"[attack] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda})")
    else:
        print("[attack] WARNING: CUDA not available -- the sweep will run on CPU. "
              "For a real sweep this is impractical.")

    summary_rows = []

    for model_key in model_keys:
        print(f"\n=== Loading {model_key} ===")
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f"[attack] SKIPPING {model_key}: failed to load -- "
                  f"{type(e).__name__}: {e}")
            continue

        for corpus_name in corpus_names:
            index, records = build_index(corpus_name, split=split)

            for injection_template in injection_templates:
                print(f"--- {model_key} x {corpus_name} x {injection_template} ---")
                raw_path, summary_path = attack_result_file_paths(
                    RESULTS_DIR, model_key, corpus_name, injection_template, engine
                )

                em_scores, f1_clean_scores, f1_raw_scores = [], [], []
                ca_scores, asr_scores = [], []
                start = time.time()

                with _atomic_open(raw_path) as raw_f:
                    n_processed = 0
                    for i, record in enumerate(records):
                        question = extract_question(corpus_name, record)
                        gold = extract_gold_answers(corpus_name, record)
                        if not question or not gold:
                            continue
                        if sample_n and n_processed >= sample_n:
                            break

                        result = run_attack_query(
                            model, tokenizer, model_key, corpus_name, question,
                            injection_template, index=index, records=records,
                        )

                        em = exact_match(result["generated_answer_clean"], gold)
                        f1_clean = f1_score(result["generated_answer_clean"], gold)
                        f1_raw = f1_score(result["generated_answer"], gold)
                        ca = contains_answer(result["generated_answer"], gold)
                        asr = score_asr(result["generated_answer"], result["target_string"])
                        em_scores.append(em)
                        f1_clean_scores.append(f1_clean)
                        f1_raw_scores.append(f1_raw)
                        ca_scores.append(ca)
                        asr_scores.append(asr)

                        raw_f.write(json.dumps({
                            "model": model_key,
                            "corpus": corpus_name,
                            "injection_template": injection_template,
                            "hijack_type": result["hijack_type"],
                            "seed": SEED,
                            "question": question,
                            "gold_answers": gold,
                            "generated_answer": result["generated_answer"],
                            "generated_answer_clean": result["generated_answer_clean"],
                            "f1_clean": f1_clean,
                            "f1_raw": f1_raw,
                            "exact_match": em,
                            "contains_answer_diagnostic": ca,
                            "target_string": result["target_string"],
                            "attack_success": asr,
                            "retrieved_doc_ids": result["retrieved_doc_ids"],
                        }, ensure_ascii=False) + "\n")
                        n_processed += 1

                        if (i + 1) % 50 == 0:
                            print(f"  {i + 1}/{len(records)} done "
                                  f"({time.time() - start:.0f}s elapsed)")

                summary_row = _summarize(
                    model_key, corpus_name, injection_template,
                    em_scores, f1_clean_scores, f1_raw_scores, ca_scores, asr_scores,
                )
                summary_rows.append(summary_row)
                _write_summary_csv(summary_path, summary_row)

                print(f"  {model_key} x {corpus_name} x {injection_template}: "
                      f"F1(clean)={summary_row['f1_clean']:.4f}  "
                      f"ASR={summary_row['attack_success_rate']:.4f}  "
                      f"n={summary_row['n']}")
                print(f"  raw -> {raw_path}")
                print(f"  summary -> {summary_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


def _run_vllm_attack_sweep(model_keys, corpus_names, injection_templates, split, sample_n=None):
    """
    vLLM control flow: retrieval + injection + prompt construction runs
    per-question, but all prompts for a (corpus, injection_template) go to
    vLLM in a single generate_batch() call -- mirrors
    evaluation.run_baseline._run_vllm_sweep exactly (see that function's
    docstring); kept in lockstep deliberately.
    """
    from harness.vllm_engine import load_vllm_model, generate_batch

    max_model_len = _resolve_vllm_max_model_len()

    if torch.cuda.is_available():
        print(f"[attack] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda}) -- engine: vLLM, "
              f"max_model_len={max_model_len}")
    else:
        print("[attack] WARNING: CUDA not available -- vLLM requires a GPU.")

    summary_rows = []

    for model_key in model_keys:
        print(f"\n=== Loading {model_key} (vLLM) ===")
        llm = load_vllm_model(model_key, max_model_len=max_model_len)

        for corpus_name in corpus_names:
            index, records = build_index(corpus_name, split=split)

            for injection_template in injection_templates:
                print(f"--- {model_key} x {corpus_name} x {injection_template} (vLLM, batched) ---")
                raw_path, summary_path = attack_result_file_paths(
                    RESULTS_DIR, model_key, corpus_name, injection_template, "vllm"
                )

                questions, golds, user_prompts, retrieved_ids = [], [], [], []
                target_strings, hijack_types = [], []
                for record in records:
                    question = extract_question(corpus_name, record)
                    gold = extract_gold_answers(corpus_name, record)
                    if not question or not gold:
                        continue
                    if sample_n and len(questions) >= sample_n:
                        break
                    user_prompt, retrieved, target_string, hijack_type = build_attack_user_prompt(
                        index, records, question, corpus_name, injection_template, top_k=TOP_K
                    )
                    questions.append(question)
                    golds.append(gold)
                    user_prompts.append(user_prompt)
                    retrieved_ids.append([idx for idx, _ in enumerate(retrieved)])
                    target_strings.append(target_string)
                    hijack_types.append(hijack_type)

                start = time.time()
                print(f"  sending {len(user_prompts)} prompts in one batch")
                outputs = generate_batch(llm, model_key, SYSTEM_PROMPT, user_prompts)
                print(f"  batch generated in {time.time() - start:.0f}s")

                em_scores, f1_clean_scores, f1_raw_scores = [], [], []
                ca_scores, asr_scores = [], []
                with _atomic_open(raw_path) as raw_f:
                    for question, gold, doc_ids, generated, target_string, hijack_type in zip(
                        questions, golds, retrieved_ids, outputs, target_strings, hijack_types
                    ):
                        generated = generated.strip()
                        generated_clean = clean_generation(generated)

                        em = exact_match(generated_clean, gold)
                        f1_clean = f1_score(generated_clean, gold)
                        f1_raw = f1_score(generated, gold)
                        ca = contains_answer(generated, gold)
                        asr = score_asr(generated, target_string)
                        em_scores.append(em)
                        f1_clean_scores.append(f1_clean)
                        f1_raw_scores.append(f1_raw)
                        ca_scores.append(ca)
                        asr_scores.append(asr)

                        raw_f.write(json.dumps({
                            "model": model_key,
                            "corpus": corpus_name,
                            "injection_template": injection_template,
                            "hijack_type": hijack_type,
                            "seed": SEED,
                            "question": question,
                            "gold_answers": gold,
                            "generated_answer": generated,
                            "generated_answer_clean": generated_clean,
                            "f1_clean": f1_clean,
                            "f1_raw": f1_raw,
                            "exact_match": em,
                            "contains_answer_diagnostic": ca,
                            "target_string": target_string,
                            "attack_success": asr,
                            "retrieved_doc_ids": doc_ids,
                        }, ensure_ascii=False) + "\n")

                summary_row = _summarize(
                    model_key, corpus_name, injection_template,
                    em_scores, f1_clean_scores, f1_raw_scores, ca_scores, asr_scores,
                )
                summary_rows.append(summary_row)
                _write_summary_csv(summary_path, summary_row)

                print(f"  {model_key} x {corpus_name} x {injection_template}: "
                      f"F1(clean)={summary_row['f1_clean']:.4f}  "
                      f"ASR={summary_row['attack_success_rate']:.4f}  "
                      f"n={summary_row['n']}")
                print(f"  raw -> {raw_path}")
                print(f"  summary -> {summary_path}")

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


def _resolve_vllm_max_model_len():
    """Identical logic to evaluation.run_baseline._resolve_vllm_max_model_len
    -- not imported from there to keep this module's dependency surface
    matching its own imports (config.RAG_VLLM_MAX_MODEL_LEN), same
    RAG_-prefix rationale as that function's docstring."""
    env_val = os.environ.get("RAG_VLLM_MAX_MODEL_LEN")
    if env_val:
        return int(env_val)
    if RAG_VLLM_MAX_MODEL_LEN:
        return RAG_VLLM_MAX_MODEL_LEN
    raise RuntimeError(
        "RAG_VLLM_MAX_MODEL_LEN is not set. Run scripts/compute_max_model_len.py "
        "where the corpora/indices exist, then set config.RAG_VLLM_MAX_MODEL_LEN "
        "or export RAG_VLLM_MAX_MODEL_LEN."
    )


def _summarize(model_key, corpus_name, injection_template,
               em_scores, f1_clean_scores, f1_raw_scores, ca_scores, asr_scores):
    n = len(em_scores)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {
        "model": model_key,
        "corpus": corpus_name,
        "injection_template": injection_template,
        "n": n,
        "f1_clean": round(mean(f1_clean_scores), 4),
        "exact_match": round(mean(em_scores), 4),
        "f1_raw": round(mean(f1_raw_scores), 4),
        "contains_answer_diagnostic": round(mean(ca_scores), 4),
        "attack_success_rate": round(mean(asr_scores), 4),
    }


_SUMMARY_FIELDNAMES = [
    "model", "corpus", "injection_template", "n",
    "f1_clean", "exact_match", "f1_raw", "contains_answer_diagnostic",
    "attack_success_rate",
]


def _write_summary_csv(summary_path, summary_row):
    with _atomic_open(summary_path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary_row)


if __name__ == "__main__":
    run_attack_sweep()
