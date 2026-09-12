"""
Phase 2, Attack 2 of 3: PoisonedRAG corpus poisoning sweep.

Two-phase design, not the single model-outer loop Attack 1/baseline use --
required by the black-box methodology itself, not just an optimization.
Poisoning is per (corpus, target question), model-agnostic (one shared
retriever across all four target LLMs, per PHASE2_ROADMAP.md's black-box
reasoning) and costs a real, paid, rate-limited API call. Generating it
once and reusing it across all four models is the whole point of "black-
box, retriever-only" -- regenerating per model would silently 4x the API
spend for a result the paper's own methodology treats as shared.

  Phase A (build_poisoned_contexts, per corpus): sample 100 target
  questions -> generate each one's poison via
  attacks.poisonedrag.generate_poison_texts (retried, skip-and-log on
  exhausted failure) -> merge-retrieve + verify
  (attacks.poisoned_retrieval) -> cache the rendered per-question context
  to disk (atomic write -- this is real spent API budget, must survive a
  later stage crashing). Model-independent, GPU-independent.

  Phase B (run_poisonedrag_sweep): for each model (loaded once, like
  Attack 1), for each corpus, for each cached poisoned context, generate +
  score (attacks.poison_scoring) + write raw JSONL / summary CSV via
  evaluation.result_paths's poison_result_file_paths.

Same atomic-write discipline as evaluation/run_attack_injection.py's
_atomic_open (kept byte-for-byte in sync, not imported, same reasoning as
that module's own docstring). nq_open excluded by default
(ATTACK_ELIGIBLE_CORPORA) -- see nq_open_leakage_finding.md.
"""

import csv
import gc
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager

import torch

from attacks.poison_scoring import score_poison_success
from attacks.poisoned_retrieval import embed_poison_texts, render_poisoned_context, retrieve_with_poison
from attacks.poisonedrag import (
    DEFAULT_POISON_CONFIG,
    build_poisoned_passage,
    generate_poison_texts,
    sample_target_questions,
)
from config import MODELS, RESULTS_DIR, SEED, TOP_K, RAG_VLLM_MAX_MODEL_LEN
from data.build_index import build_index
from data.normalize import extract_gold_answers, extract_question
from defenses.instruction_detection import detect_injection
from defenses.output_filter import run_output_filter
from defenses.spotlighting import SPOTLIGHTING_SYSTEM_INSTRUCTION, encode_passage_base64
from evaluation.metrics import retrieval_f1_at_k
from evaluation.result_paths import (
    ATTACK_ELIGIBLE_CORPORA,
    poison_result_file_paths,
    resolve_defense,
    resolve_poison_sweep_selection,
)
from harness.model_loader import build_chat_prompt, load_model
from harness.pipeline import SYSTEM_PROMPT, clean_generation

# Same deliberate constraint as evaluation/run_attack_injection.py's
# SPOTLIGHTING_TOP_K -- see that module's comment and defenses/
# spotlighting.py's "TOKEN BUDGET IMPACT ANALYSIS" docstring for the real
# measured ~4.12x token-expansion numbers this is based on. top_k=5
# overflows RAG_VLLM_MAX_MODEL_LEN=13056 on both attack-eligible corpora;
# top_k=2 keeps both under budget.
SPOTLIGHTING_TOP_K = 2


def _defended_top_k(defense: str) -> int:
    return SPOTLIGHTING_TOP_K if defense == "spotlighting" else TOP_K


def _render_defended_poisoned_context(retrieved: list, corpus_name: str, defense: str) -> str:
    """
    Mirrors attacks.poisoned_retrieval.render_poisoned_context's per-item
    text resolution exactly (kept in sync, not imported -- same reasoning
    as this module's own _atomic_open): poison entries (doc_id is the
    "poison::{i}" string sentinel) are already-final text, real corpus
    records need extract_passage_text. Applies a pre-generation defense to
    EACH item's text -- including poison entries, the whole point of
    testing a defense against PoisonedRAG -- before rendering.
    """
    from data.normalize import extract_passage_text

    lines = []
    for i, (item, _score, doc_id) in enumerate(retrieved):
        text = item if isinstance(doc_id, str) else extract_passage_text(corpus_name, item)
        if defense == "instruction_detection":
            if detect_injection(text).flagged:
                continue
            lines.append(f"[{i + 1}] {text}")
        elif defense == "spotlighting":
            lines.append(f"[{i + 1}] {encode_passage_base64(text)}")
        else:
            lines.append(f"[{i + 1}] {text}")
    return "\n\n".join(lines)


@contextmanager
def _atomic_open(final_path, newline=None):
    """Identical to evaluation.run_attack_injection._atomic_open -- see
    that module's docstring for the full rationale."""
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


def _generate_poison_with_retry(question: str, correct_answer: str, api_token: str,
                                 max_attempts: int = 3):
    """
    Bounded retry with backoff -- mirrors data.build_index._get_embedder's
    pattern (a transient failure gets a few attempts, not immediately
    fatal). UNLIKE that function, does NOT re-raise after exhausting
    attempts: at 200 real API calls across a sweep, one persistently-
    failing question must not crash an otherwise-complete run -- same
    failure-isolation precedent as run_baseline_sweep skipping a model
    that fails to load rather than aborting the whole sweep. Returns
    (incorrect_answer, corpus_texts) or None on exhausted failure.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return generate_poison_texts(question, correct_answer, api_token=api_token)
        except Exception as e:
            last_exc = e
            print(
                f"[poison] generation attempt {attempt + 1}/{max_attempts} failed "
                f"for {question!r}: {type(e).__name__}: {e}"
                + (" -- retrying" if attempt < max_attempts - 1 else " -- giving up"),
                file=sys.stderr,
            )
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # 1s, 2s
    print(
        f"[poison] SKIPPING question after {max_attempts} failed attempts "
        f"(last error: {type(last_exc).__name__}: {last_exc}): {question!r}",
        file=sys.stderr,
    )
    return None


def _poisoned_context_cache_path(corpus_name: str, poison_config: str, defense: str = "none"):
    from evaluation.result_paths import _defense_suffix
    return RESULTS_DIR / f"poisoned_contexts_{corpus_name}_{poison_config}{_defense_suffix(defense)}.json"


def build_poisoned_contexts(corpus_name: str, split: str = "dev", sample_n: int = None,
                             poison_config: str = DEFAULT_POISON_CONFIG,
                             api_token: str = None, top_k: int = TOP_K,
                             use_cache: bool = True, defense: str = "none"):
    """
    Phase A. Returns a list of per-question context dicts:
      {question, gold_answers, target_answer, context, retrieved_doc_ids,
       retrieval_f1, retrieval_recall, retrieval_precision}
    Cached to disk (atomic write, cache path namespaced by defense too --
    see _poisoned_context_cache_path) so a Phase B crash doesn't waste
    already-spent API budget on a rerun -- use_cache=False forces
    regeneration (e.g. after a poison-generation bug fix invalidates the
    cache). defense="instruction_detection"/"spotlighting" render the
    context via _render_defended_poisoned_context instead of
    render_poisoned_context; "output_filter"/"none" render unchanged (that
    defense is post-generation only).
    """
    from attacks.poisonedrag import SAMPLE_SIZE

    # Resolve the ACTUAL target count before touching the cache. Bug found
    # live: the old check was `sample_n is None or len(cached) >= sample_n`
    # -- when sample_n is None (RAG_SAMPLE_N unset, meaning "use the full
    # SAMPLE_SIZE=100 default"), `sample_n is None` short-circuited the
    # whole condition to True regardless of cache size, so a 2-question
    # smoke-test cache silently satisfied a real 100-question run. The
    # cache is only ever valid against the size actually intended, never
    # against the raw (possibly-None) argument.
    sample_size = sample_n if sample_n is not None else SAMPLE_SIZE

    cache_path = _poisoned_context_cache_path(corpus_name, poison_config, defense)
    if use_cache and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= sample_size:
            print(f"[poison] using cached contexts: {cache_path} "
                  f"({len(cached)} questions, need {sample_size})")
            return cached[:sample_size]
        print(f"[poison] cache at {cache_path} has only {len(cached)} questions "
              f"but {sample_size} are needed -- regenerating")

    api_token = api_token or os.environ.get("NVIDIA_NIM_API_KEY")
    index, records = build_index(corpus_name, split=split)
    target_records = sample_target_questions(records, corpus_name, sample_size=sample_size, seed=SEED)

    contexts = []
    for i, record in enumerate(target_records):
        question = extract_question(corpus_name, record)
        gold_answers = extract_gold_answers(corpus_name, record)
        if not question or not gold_answers:
            continue

        result = _generate_poison_with_retry(question, gold_answers[0], api_token)
        if result is None:
            continue
        target_answer, corpus_texts = result

        poison_texts = [build_poisoned_passage(question, c) for c in corpus_texts]
        poison_embeddings = embed_poison_texts(poison_texts)
        retrieved, poison_doc_ids = retrieve_with_poison(
            index, records, poison_texts, poison_embeddings, question, k=top_k
        )
        retrieval = retrieval_f1_at_k(
            [doc_id for _, _, doc_id in retrieved], poison_doc_ids, k=top_k
        )
        if defense in ("instruction_detection", "spotlighting"):
            context = _render_defended_poisoned_context(retrieved, corpus_name, defense)
        else:
            context = render_poisoned_context(retrieved, corpus_name)

        contexts.append({
            "question": question,
            "gold_answers": gold_answers,
            "target_answer": target_answer,
            "context": context,
            "retrieved_doc_ids": [doc_id for _, _, doc_id in retrieved],
            "retrieval_precision": retrieval["precision"],
            "retrieval_recall": retrieval["recall"],
            "retrieval_f1": retrieval["f1"],
        })

        if (i + 1) % 10 == 0:
            print(f"  [poison] {i + 1}/{len(target_records)} target questions processed "
                  f"({corpus_name})")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(cache_path.parent), prefix=".tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(contexts, f, ensure_ascii=False)
            os.replace(tmp_path, str(cache_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    n_skipped = len(target_records) - len(contexts)
    print(f"[poison] {corpus_name}: {len(contexts)} poisoned contexts built"
          + (f", {n_skipped} skipped after exhausted retries" if n_skipped else ""))
    return contexts


def run_poisonedrag_sweep(model_keys=None, corpus_names=None, poison_configs=None,
                           split: str = "dev", sample_n: int = None, defense=None):
    """
    sample_n: caps target questions PER CORPUS for the smoke-test path
    (falls back to RAG_SAMPLE_N, same semantics as Attack 1's
    run_attack_sweep) -- verify the full pipeline (poison generation, real
    API call, retrieval-verification, generation, scoring, file output) on
    a handful of questions before committing the real 100-question sweep's
    API and GPU budget.

    defense: one of evaluation.result_paths.DEFENSE_OPTIONS ("none" by
    default, falls back to RAG_DEFENSE via resolve_defense()). See
    evaluation/run_attack_injection.py's run_attack_sweep docstring for the
    same pre-/post-generation split -- identical here.
    """
    if sample_n is None:
        env_val = os.environ.get("RAG_SAMPLE_N")
        sample_n = int(env_val) if env_val else None
    if defense is None:
        defense = resolve_defense()

    default_model_keys, default_corpus_names, default_poison_configs, engine = (
        resolve_poison_sweep_selection()
    )
    if model_keys is None:
        model_keys = default_model_keys
    if corpus_names is None:
        corpus_names = default_corpus_names
    if poison_configs is None:
        poison_configs = default_poison_configs

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
    if engine not in ("hf", "vllm"):
        raise ValueError(f"INFERENCE_ENGINE must be 'hf' or 'vllm', got '{engine}'")

    # Phase A -- once per (corpus, poison_config), shared across every model.
    # top_k is forced to SPOTLIGHTING_TOP_K for the Spotlighting condition
    # specifically -- see this module's SPOTLIGHTING_TOP_K comment.
    top_k = _defended_top_k(defense)
    contexts_by_corpus = {}
    for corpus_name in corpus_names:
        for poison_config in poison_configs:
            contexts_by_corpus[(corpus_name, poison_config)] = build_poisoned_contexts(
                corpus_name, split=split, sample_n=sample_n, poison_config=poison_config,
                top_k=top_k, defense=defense,
            )

    if engine == "vllm":
        return _run_vllm_poison_sweep(model_keys, corpus_names, poison_configs, contexts_by_corpus, defense)
    return _run_hf_poison_sweep(model_keys, corpus_names, poison_configs, contexts_by_corpus, defense)


def _run_hf_poison_sweep(model_keys, corpus_names, poison_configs, contexts_by_corpus, defense="none"):
    if torch.cuda.is_available():
        print(f"[poison] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda})")
    else:
        print("[poison] WARNING: CUDA not available -- the sweep will run on CPU.")

    summary_rows = []
    for model_key in model_keys:
        print(f"\n=== Loading {model_key} ===")
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f"[poison] SKIPPING {model_key}: failed to load -- {type(e).__name__}: {e}")
            continue

        system_prompt = (
            SYSTEM_PROMPT + "\n\n" + SPOTLIGHTING_SYSTEM_INSTRUCTION
            if defense == "spotlighting" else SYSTEM_PROMPT
        )

        for corpus_name in corpus_names:
            output_filter_log_path = None
            if defense == "output_filter":
                output_filter_log_path = RESULTS_DIR / f"output_filter_log_poison_{model_key}_{corpus_name}.jsonl"

            for poison_config in poison_configs:
                contexts = contexts_by_corpus[(corpus_name, poison_config)]
                print(f"--- {model_key} x {corpus_name} x {poison_config} (defense={defense}) ---")
                raw_path, summary_path = poison_result_file_paths(
                    RESULTS_DIR, model_key, corpus_name, poison_config, "hf", defense
                )

                rows = []
                start = time.time()
                with _atomic_open(raw_path) as raw_f:
                    for i, ctx in enumerate(contexts):
                        user_prompt = f"Context:\n{ctx['context']}\n\nQuestion: {ctx['question']}"
                        prompt = build_chat_prompt(model_key, tokenizer, system_prompt, user_prompt)
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                        generated = tokenizer.decode(
                            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                        ).strip()
                        if defense == "output_filter":
                            generated = run_output_filter(
                                generated, output_filter_log_path,
                                {"model": model_key, "corpus": corpus_name,
                                 "poison_config": poison_config, "question": ctx["question"]},
                            )
                        generated_clean = clean_generation(generated)

                        scores = score_poison_success(
                            generated, generated_clean, ctx["target_answer"], ctx["gold_answers"]
                        )
                        row = _build_row(model_key, corpus_name, poison_config, ctx,
                                          generated, generated_clean, scores)
                        rows.append(row)
                        raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                        if (i + 1) % 20 == 0:
                            print(f"  {i + 1}/{len(contexts)} done ({time.time() - start:.0f}s elapsed)")

                summary_row = _summarize(model_key, corpus_name, poison_config, rows)
                summary_rows.append(summary_row)
                _write_summary_csv(summary_path, summary_row)
                print(f"  {model_key} x {corpus_name} x {poison_config}: "
                      f"attack_success_rate={summary_row['attack_success_rate']:.4f}  "
                      f"f1_gold={summary_row['f1_clean']:.4f}  n={summary_row['n']}")
                print(f"  raw -> {raw_path}")
                print(f"  summary -> {summary_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


def _run_vllm_poison_sweep(model_keys, corpus_names, poison_configs, contexts_by_corpus, defense="none"):
    """vLLM control flow: mirrors evaluation.run_attack_injection.
    _run_vllm_attack_sweep -- one batched generate_batch() call per
    (model, corpus, poison_config), all prompts already pre-rendered by
    Phase A."""
    from harness.vllm_engine import generate_batch, load_vllm_model

    max_model_len = _resolve_vllm_max_model_len()

    if torch.cuda.is_available():
        print(f"[poison] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda}) -- engine: vLLM, max_model_len={max_model_len}")
    else:
        print("[poison] WARNING: CUDA not available -- vLLM requires a GPU.")

    summary_rows = []
    for model_key in model_keys:
        print(f"\n=== Loading {model_key} (vLLM) ===")
        llm = load_vllm_model(model_key, max_model_len=max_model_len)

        system_prompt = (
            SYSTEM_PROMPT + "\n\n" + SPOTLIGHTING_SYSTEM_INSTRUCTION
            if defense == "spotlighting" else SYSTEM_PROMPT
        )

        for corpus_name in corpus_names:
            output_filter_log_path = None
            if defense == "output_filter":
                output_filter_log_path = RESULTS_DIR / f"output_filter_log_poison_{model_key}_{corpus_name}.jsonl"

            for poison_config in poison_configs:
                contexts = contexts_by_corpus[(corpus_name, poison_config)]
                print(f"--- {model_key} x {corpus_name} x {poison_config} (vLLM, batched, defense={defense}) ---")
                raw_path, summary_path = poison_result_file_paths(
                    RESULTS_DIR, model_key, corpus_name, poison_config, "vllm", defense
                )

                user_prompts = [
                    f"Context:\n{ctx['context']}\n\nQuestion: {ctx['question']}" for ctx in contexts
                ]
                start = time.time()
                print(f"  sending {len(user_prompts)} prompts in one batch")
                outputs = generate_batch(llm, model_key, system_prompt, user_prompts)
                print(f"  batch generated in {time.time() - start:.0f}s")

                rows = []
                with _atomic_open(raw_path) as raw_f:
                    for ctx, generated in zip(contexts, outputs):
                        generated = generated.strip()
                        if defense == "output_filter":
                            generated = run_output_filter(
                                generated, output_filter_log_path,
                                {"model": model_key, "corpus": corpus_name,
                                 "poison_config": poison_config, "question": ctx["question"]},
                            )
                        generated_clean = clean_generation(generated)
                        scores = score_poison_success(
                            generated, generated_clean, ctx["target_answer"], ctx["gold_answers"]
                        )
                        row = _build_row(model_key, corpus_name, poison_config, ctx,
                                          generated, generated_clean, scores)
                        rows.append(row)
                        raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                summary_row = _summarize(model_key, corpus_name, poison_config, rows)
                summary_rows.append(summary_row)
                _write_summary_csv(summary_path, summary_row)
                print(f"  {model_key} x {corpus_name} x {poison_config}: "
                      f"attack_success_rate={summary_row['attack_success_rate']:.4f}  "
                      f"f1_gold={summary_row['f1_clean']:.4f}  n={summary_row['n']}")
                print(f"  raw -> {raw_path}")
                print(f"  summary -> {summary_path}")

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


def _resolve_vllm_max_model_len():
    """Identical logic to evaluation.run_attack_injection._resolve_vllm_max_model_len."""
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


def _build_row(model_key, corpus_name, poison_config, ctx, generated, generated_clean, scores):
    # poison_echoed_not_adopted: the named middle-case bucket
    # attacks/poison_scoring.py's docstring anticipates (a real-question
    # instance surfaced it: model echoed a poison passage's exact phrasing,
    # contains_target_diagnostic=1, but attack_success=0 because the
    # generated answer as a whole never equals target_answer) -- influenced
    # by the poison without adopting it as the answer. Kept invisible inside
    # a binary success/fail column undercounts a real, distinct outcome.
    poison_echoed_not_adopted = int(
        scores["contains_target_diagnostic"] == 1 and scores["attack_success"] == 0
    )
    return {
        "model": model_key,
        "corpus": corpus_name,
        "poison_config": poison_config,
        "seed": SEED,
        "question": ctx["question"],
        "gold_answers": ctx["gold_answers"],
        "target_answer": ctx["target_answer"],
        "generated_answer": generated,
        "generated_answer_clean": generated_clean,
        "attack_success": scores["attack_success"],
        "f1_target": scores["f1_target"],
        "contains_target_diagnostic": scores["contains_target_diagnostic"],
        "poison_echoed_not_adopted": poison_echoed_not_adopted,
        "exact_match": scores["em_gold"],
        "f1_clean": scores["f1_gold"],
        "retrieved_doc_ids": ctx["retrieved_doc_ids"],
        "retrieval_precision": ctx["retrieval_precision"],
        "retrieval_recall": ctx["retrieval_recall"],
        "retrieval_f1": ctx["retrieval_f1"],
    }


def _summarize(model_key, corpus_name, poison_config, rows):
    n = len(rows)
    mean = lambda key: (sum(r[key] for r in rows) / n) if n else float("nan")
    return {
        "model": model_key,
        "corpus": corpus_name,
        "poison_config": poison_config,
        "n": n,
        "attack_success_rate": round(mean("attack_success"), 4),
        "f1_target": round(mean("f1_target"), 4),
        "contains_target_diagnostic": round(mean("contains_target_diagnostic"), 4),
        "poison_echoed_not_adopted_rate": round(mean("poison_echoed_not_adopted"), 4),
        "exact_match": round(mean("exact_match"), 4),
        "f1_clean": round(mean("f1_clean"), 4),
        "retrieval_f1_at_k": round(mean("retrieval_f1"), 4),
    }


_SUMMARY_FIELDNAMES = [
    "model", "corpus", "poison_config", "n",
    "attack_success_rate", "f1_target", "contains_target_diagnostic",
    "poison_echoed_not_adopted_rate", "exact_match", "f1_clean", "retrieval_f1_at_k",
]


def _write_summary_csv(summary_path, summary_row):
    with _atomic_open(summary_path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary_row)


if __name__ == "__main__":
    run_poisonedrag_sweep()
