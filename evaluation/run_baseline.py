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
    final_path = str(final_path)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix='.tmp_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline=newline) as f:
            yield f
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def run_baseline_sweep(model_keys=None, corpus_names=None, split='dev'):
    default_model_keys, default_corpus_names, engine = resolve_sweep_selection()
    if model_keys is None:
        model_keys = default_model_keys
    if corpus_names is None:
        corpus_names = default_corpus_names
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f'Unknown model keys {unknown}. Options: {list(MODELS)}')
    unknown_corpora = [c for c in corpus_names if c not in CORPORA]
    if unknown_corpora:
        raise ValueError(f'Unknown corpus names {unknown_corpora}. Options: {list(CORPORA)}')
    if engine not in ('hf', 'vllm'):
        raise ValueError(f"INFERENCE_ENGINE must be 'hf' or 'vllm', got '{engine}'")
    if engine == 'vllm':
        return _run_vllm_sweep(model_keys, corpus_names, split)
    if torch.cuda.is_available():
        print(f'[baseline] Running on GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})')
    else:
        print('[baseline] WARNING: CUDA not available — the sweep will run on CPU. For 4 models x 3 corpora this is impractical. Install the CUDA torch wheel (see requirements.txt) or run on a GPU machine.')
    summary_rows = []
    for model_key in model_keys:
        print(f'\n=== Loading {model_key} ===')
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f'[baseline] SKIPPING {model_key}: failed to load — {type(e).__name__}: {e}')
            continue
        for corpus_name in corpus_names:
            print(f'--- {model_key} x {corpus_name} ---')
            index, records = build_index(corpus_name, split=split)
            raw_path, summary_path = result_file_paths(RESULTS_DIR, model_key, corpus_name, engine)
            em_scores, f1_clean_scores, f1_raw_scores, ca_scores = ([], [], [], [])
            start = time.time()
            with _atomic_open(raw_path) as raw_f:
                for i, record in enumerate(records):
                    question = extract_question(corpus_name, record)
                    gold = extract_gold_answers(corpus_name, record)
                    if not question or not gold:
                        continue
                    result = run_query(model, tokenizer, model_key, corpus_name, question, index=index, records=records)
                    em = exact_match(result['generated_answer_clean'], gold)
                    f1_clean = f1_score(result['generated_answer_clean'], gold)
                    f1_raw = f1_score(result['generated_answer'], gold)
                    ca = contains_answer(result['generated_answer'], gold)
                    em_scores.append(em)
                    f1_clean_scores.append(f1_clean)
                    f1_raw_scores.append(f1_raw)
                    ca_scores.append(ca)
                    raw_f.write(json.dumps({'model': model_key, 'corpus': corpus_name, 'seed': SEED, 'question': question, 'gold_answers': gold, 'generated_answer': result['generated_answer'], 'generated_answer_clean': result['generated_answer_clean'], 'f1_clean': f1_clean, 'f1_raw': f1_raw, 'exact_match': em, 'contains_answer_diagnostic': ca, 'retrieved_doc_ids': result['retrieved_doc_ids']}, ensure_ascii=False) + '\n')
                    if (i + 1) % 50 == 0:
                        print(f'  {i + 1}/{len(records)} done ({time.time() - start:.0f}s elapsed)')
            mean_em = sum(em_scores) / len(em_scores) if em_scores else float('nan')
            mean_f1_clean = sum(f1_clean_scores) / len(f1_clean_scores) if f1_clean_scores else float('nan')
            mean_f1_raw = sum(f1_raw_scores) / len(f1_raw_scores) if f1_raw_scores else float('nan')
            mean_ca = sum(ca_scores) / len(ca_scores) if ca_scores else float('nan')
            summary_row = {'model': model_key, 'corpus': corpus_name, 'n': len(em_scores), 'f1_clean': round(mean_f1_clean, 4), 'exact_match': round(mean_em, 4), 'f1_raw': round(mean_f1_raw, 4), 'contains_answer_diagnostic': round(mean_ca, 4)}
            summary_rows.append(summary_row)
            with _atomic_open(summary_path, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['model', 'corpus', 'n', 'f1_clean', 'exact_match', 'f1_raw', 'contains_answer_diagnostic'])
                writer.writeheader()
                writer.writerow(summary_row)
            print(f'  {model_key} x {corpus_name}: F1(clean)={mean_f1_clean:.4f}  EM(clean)={mean_em:.4f}  F1(raw)={mean_f1_raw:.4f}  contains(diagnostic)={mean_ca:.4f}  n={len(em_scores)}')
            print(f'  raw -> {raw_path}')
            print(f'  summary -> {summary_path}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary_rows

def _resolve_vllm_max_model_len():
    env_val = os.environ.get('RAG_VLLM_MAX_MODEL_LEN')
    if env_val:
        return int(env_val)
    if RAG_VLLM_MAX_MODEL_LEN:
        return RAG_VLLM_MAX_MODEL_LEN
    raise RuntimeError('RAG_VLLM_MAX_MODEL_LEN is not set. This value must be computed from real prompts, not guessed: run scripts/compute_max_model_len.py where the corpora/indices exist, then set config.RAG_VLLM_MAX_MODEL_LEN or export RAG_VLLM_MAX_MODEL_LEN.')

def _run_vllm_sweep(model_keys, corpus_names, split):
    from harness.vllm_engine import load_vllm_model, generate_batch
    max_model_len = _resolve_vllm_max_model_len()
    if torch.cuda.is_available():
        print(f'[baseline] Running on GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) — engine: vLLM, max_model_len={max_model_len}')
    else:
        print('[baseline] WARNING: CUDA not available — vLLM requires a GPU; the load below is expected to fail on this machine.')
    summary_rows = []
    for model_key in model_keys:
        print(f'\n=== Loading {model_key} (vLLM) ===')
        llm = load_vllm_model(model_key, max_model_len=max_model_len)
        for corpus_name in corpus_names:
            print(f'--- {model_key} x {corpus_name} (vLLM, batched) ---')
            index, records = build_index(corpus_name, split=split)
            raw_path, summary_path = result_file_paths(RESULTS_DIR, model_key, corpus_name, 'vllm')
            questions, golds, user_prompts, retrieved_ids = ([], [], [], [])
            for record in records:
                question = extract_question(corpus_name, record)
                gold = extract_gold_answers(corpus_name, record)
                if not question or not gold:
                    continue
                user_prompt, retrieved = build_rag_user_prompt(index, records, question, corpus_name, top_k=TOP_K)
                questions.append(question)
                golds.append(gold)
                user_prompts.append(user_prompt)
                retrieved_ids.append([idx for idx, _ in enumerate(retrieved)])
            start = time.time()
            print(f'  sending {len(user_prompts)} prompts in one batch')
            outputs = generate_batch(llm, model_key, SYSTEM_PROMPT, user_prompts)
            print(f'  batch generated in {time.time() - start:.0f}s')
            em_scores, f1_clean_scores, f1_raw_scores, ca_scores = ([], [], [], [])
            with _atomic_open(raw_path) as raw_f:
                for question, gold, doc_ids, generated in zip(questions, golds, retrieved_ids, outputs):
                    generated = generated.strip()
                    generated_clean = clean_generation(generated)
                    em = exact_match(generated_clean, gold)
                    f1_clean = f1_score(generated_clean, gold)
                    f1_raw = f1_score(generated, gold)
                    ca = contains_answer(generated, gold)
                    em_scores.append(em)
                    f1_clean_scores.append(f1_clean)
                    f1_raw_scores.append(f1_raw)
                    ca_scores.append(ca)
                    raw_f.write(json.dumps({'model': model_key, 'corpus': corpus_name, 'seed': SEED, 'question': question, 'gold_answers': gold, 'generated_answer': generated, 'generated_answer_clean': generated_clean, 'f1_clean': f1_clean, 'f1_raw': f1_raw, 'exact_match': em, 'contains_answer_diagnostic': ca, 'retrieved_doc_ids': doc_ids}, ensure_ascii=False) + '\n')
            mean_em = sum(em_scores) / len(em_scores) if em_scores else float('nan')
            mean_f1_clean = sum(f1_clean_scores) / len(f1_clean_scores) if f1_clean_scores else float('nan')
            mean_f1_raw = sum(f1_raw_scores) / len(f1_raw_scores) if f1_raw_scores else float('nan')
            mean_ca = sum(ca_scores) / len(ca_scores) if ca_scores else float('nan')
            summary_row = {'model': model_key, 'corpus': corpus_name, 'n': len(em_scores), 'f1_clean': round(mean_f1_clean, 4), 'exact_match': round(mean_em, 4), 'f1_raw': round(mean_f1_raw, 4), 'contains_answer_diagnostic': round(mean_ca, 4)}
            summary_rows.append(summary_row)
            with _atomic_open(summary_path, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['model', 'corpus', 'n', 'f1_clean', 'exact_match', 'f1_raw', 'contains_answer_diagnostic'])
                writer.writeheader()
                writer.writerow(summary_row)
            print(f'  {model_key} x {corpus_name}: F1(clean)={mean_f1_clean:.4f}  EM(clean)={mean_em:.4f}  F1(raw)={mean_f1_raw:.4f}  contains(diagnostic)={mean_ca:.4f}  n={len(em_scores)}')
            print(f'  raw -> {raw_path}')
            print(f'  summary -> {summary_path}')
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary_rows
if __name__ == '__main__':
    run_baseline_sweep()
