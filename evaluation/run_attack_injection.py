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
from data.build_index import build_index, retrieve
from data.normalize import extract_gold_answers, extract_passage_text, extract_question
from defenses.instruction_detection import PassageLog, detect_injection, log_passage_detection_event
from defenses.output_filter import run_output_filter
from defenses.spotlighting import SPOTLIGHTING_SYSTEM_INSTRUCTION, encode_passage_base64
from evaluation.metrics import contains_answer, exact_match, f1_score
from evaluation.result_paths import ATTACK_ELIGIBLE_CORPORA, attack_result_file_paths, resolve_attack_sweep_selection, resolve_defense
from harness.model_loader import build_chat_prompt, load_model
from harness.pipeline import SYSTEM_PROMPT, clean_generation
SPOTLIGHTING_TOP_K = 2

def _defended_top_k(defense: str) -> int:
    return SPOTLIGHTING_TOP_K if defense == 'spotlighting' else TOP_K

def _build_defended_attack_prompt(index, records, question: str, corpus_name: str, injection_template: str, top_k: int, defense: str):
    if injection_template not in TEMPLATES:
        raise ValueError(f"Unknown injection template '{injection_template}'. Options: {list(TEMPLATES)}")
    template = TEMPLATES[injection_template]
    retrieved = retrieve(index, records, question, k=top_k)
    if not retrieved:
        raise ValueError('No documents retrieved -- cannot inject into an empty top-k.')
    texts = []
    for i, (doc, score) in enumerate(retrieved):
        text = extract_passage_text(corpus_name, doc)
        if i == 0:
            text = template.render(text)
        texts.append(text)
    passage_log = None
    if defense == 'instruction_detection':
        detections = [detect_injection(text) for text in texts]
        lines = [f'[{i + 1}] {text}' for i, (text, result) in enumerate(zip(texts, detections)) if not result.flagged]
        passage_log = [PassageLog(passage_id=i, flagged=r.flagged, score=r.score, label=r.label) for i, r in enumerate(detections)]
        system_prompt = SYSTEM_PROMPT
    elif defense == 'spotlighting':
        lines = [f'[{i + 1}] {encode_passage_base64(text)}' for i, text in enumerate(texts)]
        system_prompt = SYSTEM_PROMPT + '\n\n' + SPOTLIGHTING_SYSTEM_INSTRUCTION
    else:
        raise ValueError(f'_build_defended_attack_prompt does not handle defense={defense!r}')
    context = '\n\n'.join(lines)
    user_prompt = f'Context:\n{context}\n\nQuestion: {question}'
    return (system_prompt, user_prompt, retrieved, template.target_string, template.hijack_type, passage_log)

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

def run_attack_sweep(model_keys=None, corpus_names=None, injection_templates=None, split='dev', sample_n=None, defense=None):
    if sample_n is None:
        env_val = os.environ.get('RAG_SAMPLE_N')
        sample_n = int(env_val) if env_val else None
    if defense is None:
        defense = resolve_defense()
    default_model_keys, default_corpus_names, default_templates, engine = resolve_attack_sweep_selection()
    if model_keys is None:
        model_keys = default_model_keys
    if corpus_names is None:
        corpus_names = default_corpus_names
    if injection_templates is None:
        injection_templates = default_templates
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f'Unknown model keys {unknown}. Options: {list(MODELS)}')
    unknown_corpora = [c for c in corpus_names if c not in ATTACK_ELIGIBLE_CORPORA]
    if unknown_corpora:
        raise ValueError(f'Unknown or excluded corpus names {unknown_corpora}. Options: {ATTACK_ELIGIBLE_CORPORA} (nq_open is excluded -- see docs/nq_open_leakage_finding.md)')
    unknown_templates = [t for t in injection_templates if t not in TEMPLATES]
    if unknown_templates:
        raise ValueError(f'Unknown injection templates {unknown_templates}. Options: {list(TEMPLATES)}')
    if engine not in ('hf', 'vllm'):
        raise ValueError(f"INFERENCE_ENGINE must be 'hf' or 'vllm', got '{engine}'")
    if engine == 'vllm':
        return _run_vllm_attack_sweep(model_keys, corpus_names, injection_templates, split, sample_n, defense)
    if torch.cuda.is_available():
        print(f'[attack] Running on GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})')
    else:
        print('[attack] WARNING: CUDA not available -- the sweep will run on CPU. For a real sweep this is impractical.')
    summary_rows = []
    for model_key in model_keys:
        print(f'\n=== Loading {model_key} ===')
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f'[attack] SKIPPING {model_key}: failed to load -- {type(e).__name__}: {e}')
            continue
        for corpus_name in corpus_names:
            index, records = build_index(corpus_name, split=split)
            top_k = _defended_top_k(defense)
            output_filter_log_path = None
            if defense == 'output_filter':
                output_filter_log_path = RESULTS_DIR / f'output_filter_log_attack_{model_key}_{corpus_name}.jsonl'
            instruction_detection_log_path = None
            if defense == 'instruction_detection':
                instruction_detection_log_path = RESULTS_DIR / f'instruction_detection_log_attack_{model_key}_{corpus_name}.jsonl'
            for injection_template in injection_templates:
                print(f'--- {model_key} x {corpus_name} x {injection_template} (defense={defense}) ---')
                raw_path, summary_path = attack_result_file_paths(RESULTS_DIR, model_key, corpus_name, injection_template, engine, defense)
                em_scores, f1_clean_scores, f1_raw_scores = ([], [], [])
                ca_scores, asr_scores = ([], [])
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
                        if defense in ('instruction_detection', 'spotlighting'):
                            system_prompt, user_prompt, retrieved, target_string, hijack_type, passage_log = _build_defended_attack_prompt(index, records, question, corpus_name, injection_template, top_k, defense)
                            if defense == 'instruction_detection':
                                log_passage_detection_event(instruction_detection_log_path, {'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'question': question, 'passages': [{'passage_id': p.passage_id, 'flagged': p.flagged, 'score': p.score, 'label': p.label} for p in passage_log]})
                            prompt = build_chat_prompt(model_key, tokenizer, system_prompt, user_prompt)
                            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                            with torch.no_grad():
                                output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                            generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                            result = {'question': question, 'retrieved_doc_ids': [idx for idx, _ in enumerate(retrieved)], 'generated_answer': generated, 'generated_answer_clean': clean_generation(generated), 'injection_template': injection_template, 'hijack_type': hijack_type, 'target_string': target_string}
                        else:
                            result = run_attack_query(model, tokenizer, model_key, corpus_name, question, injection_template, index=index, records=records, top_k=top_k)
                        if defense == 'output_filter':
                            filtered = run_output_filter(result['generated_answer'], output_filter_log_path, {'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'question': question})
                            result['generated_answer'] = filtered
                            result['generated_answer_clean'] = clean_generation(filtered)
                        em = exact_match(result['generated_answer_clean'], gold)
                        f1_clean = f1_score(result['generated_answer_clean'], gold)
                        f1_raw = f1_score(result['generated_answer'], gold)
                        ca = contains_answer(result['generated_answer'], gold)
                        asr = score_asr(result['generated_answer'], result['target_string'])
                        em_scores.append(em)
                        f1_clean_scores.append(f1_clean)
                        f1_raw_scores.append(f1_raw)
                        ca_scores.append(ca)
                        asr_scores.append(asr)
                        raw_f.write(json.dumps({'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'hijack_type': result['hijack_type'], 'seed': SEED, 'question': question, 'gold_answers': gold, 'generated_answer': result['generated_answer'], 'generated_answer_clean': result['generated_answer_clean'], 'f1_clean': f1_clean, 'f1_raw': f1_raw, 'exact_match': em, 'contains_answer_diagnostic': ca, 'target_string': result['target_string'], 'attack_success': asr, 'retrieved_doc_ids': result['retrieved_doc_ids']}, ensure_ascii=False) + '\n')
                        n_processed += 1
                        if (i + 1) % 50 == 0:
                            print(f'  {i + 1}/{len(records)} done ({time.time() - start:.0f}s elapsed)', flush=True)
                summary_row = _summarize(model_key, corpus_name, injection_template, em_scores, f1_clean_scores, f1_raw_scores, ca_scores, asr_scores)
                summary_rows.append(summary_row)
                _write_summary_csv(summary_path, summary_row)
                print(f"  {model_key} x {corpus_name} x {injection_template}: F1(clean)={summary_row['f1_clean']:.4f}  ASR={summary_row['attack_success_rate']:.4f}  n={summary_row['n']}")
                print(f'  raw -> {raw_path}')
                print(f'  summary -> {summary_path}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary_rows

def _run_vllm_attack_sweep(model_keys, corpus_names, injection_templates, split, sample_n=None, defense='none'):
    from harness.vllm_engine import load_vllm_model, generate_batch
    max_model_len = _resolve_vllm_max_model_len()
    if torch.cuda.is_available():
        print(f'[attack] Running on GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) -- engine: vLLM, max_model_len={max_model_len}')
    else:
        print('[attack] WARNING: CUDA not available -- vLLM requires a GPU.')
    summary_rows = []
    for model_key in model_keys:
        print(f'\n=== Loading {model_key} (vLLM) ===')
        llm = load_vllm_model(model_key, max_model_len=max_model_len)
        for corpus_name in corpus_names:
            index, records = build_index(corpus_name, split=split)
            top_k = _defended_top_k(defense)
            output_filter_log_path = None
            if defense == 'output_filter':
                output_filter_log_path = RESULTS_DIR / f'output_filter_log_attack_{model_key}_{corpus_name}.jsonl'
            instruction_detection_log_path = None
            if defense == 'instruction_detection':
                instruction_detection_log_path = RESULTS_DIR / f'instruction_detection_log_attack_{model_key}_{corpus_name}.jsonl'
            for injection_template in injection_templates:
                print(f'--- {model_key} x {corpus_name} x {injection_template} (vLLM, batched, defense={defense}) ---')
                raw_path, summary_path = attack_result_file_paths(RESULTS_DIR, model_key, corpus_name, injection_template, 'vllm', defense)
                questions, golds, user_prompts, retrieved_ids = ([], [], [], [])
                target_strings, hijack_types, passage_logs = ([], [], [])
                system_prompt = SYSTEM_PROMPT
                for record in records:
                    question = extract_question(corpus_name, record)
                    gold = extract_gold_answers(corpus_name, record)
                    if not question or not gold:
                        continue
                    if sample_n and len(questions) >= sample_n:
                        break
                    passage_log = None
                    if defense in ('instruction_detection', 'spotlighting'):
                        system_prompt, user_prompt, retrieved, target_string, hijack_type, passage_log = _build_defended_attack_prompt(index, records, question, corpus_name, injection_template, top_k, defense)
                    else:
                        user_prompt, retrieved, target_string, hijack_type = build_attack_user_prompt(index, records, question, corpus_name, injection_template, top_k=top_k)
                    questions.append(question)
                    golds.append(gold)
                    user_prompts.append(user_prompt)
                    retrieved_ids.append([idx for idx, _ in enumerate(retrieved)])
                    target_strings.append(target_string)
                    hijack_types.append(hijack_type)
                    passage_logs.append(passage_log)
                start = time.time()
                print(f'  sending {len(user_prompts)} prompts in one batch')
                outputs = generate_batch(llm, model_key, system_prompt, user_prompts)
                print(f'  batch generated in {time.time() - start:.0f}s')
                em_scores, f1_clean_scores, f1_raw_scores = ([], [], [])
                ca_scores, asr_scores = ([], [])
                with _atomic_open(raw_path) as raw_f:
                    for question, gold, doc_ids, generated, target_string, hijack_type, passage_log in zip(questions, golds, retrieved_ids, outputs, target_strings, hijack_types, passage_logs):
                        generated = generated.strip()
                        generated_clean = clean_generation(generated)
                        if defense == 'instruction_detection':
                            log_passage_detection_event(instruction_detection_log_path, {'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'question': question, 'passages': [{'passage_id': p.passage_id, 'flagged': p.flagged, 'score': p.score, 'label': p.label} for p in passage_log]})
                        if defense == 'output_filter':
                            generated = run_output_filter(generated, output_filter_log_path, {'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'question': question})
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
                        raw_f.write(json.dumps({'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'hijack_type': hijack_type, 'seed': SEED, 'question': question, 'gold_answers': gold, 'generated_answer': generated, 'generated_answer_clean': generated_clean, 'f1_clean': f1_clean, 'f1_raw': f1_raw, 'exact_match': em, 'contains_answer_diagnostic': ca, 'target_string': target_string, 'attack_success': asr, 'retrieved_doc_ids': doc_ids}, ensure_ascii=False) + '\n')
                summary_row = _summarize(model_key, corpus_name, injection_template, em_scores, f1_clean_scores, f1_raw_scores, ca_scores, asr_scores)
                summary_rows.append(summary_row)
                _write_summary_csv(summary_path, summary_row)
                print(f"  {model_key} x {corpus_name} x {injection_template}: F1(clean)={summary_row['f1_clean']:.4f}  ASR={summary_row['attack_success_rate']:.4f}  n={summary_row['n']}")
                print(f'  raw -> {raw_path}')
                print(f'  summary -> {summary_path}')
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary_rows

def _resolve_vllm_max_model_len():
    env_val = os.environ.get('RAG_VLLM_MAX_MODEL_LEN')
    if env_val:
        return int(env_val)
    if RAG_VLLM_MAX_MODEL_LEN:
        return RAG_VLLM_MAX_MODEL_LEN
    raise RuntimeError('RAG_VLLM_MAX_MODEL_LEN is not set. Run scripts/compute_max_model_len.py where the corpora/indices exist, then set config.RAG_VLLM_MAX_MODEL_LEN or export RAG_VLLM_MAX_MODEL_LEN.')

def _summarize(model_key, corpus_name, injection_template, em_scores, f1_clean_scores, f1_raw_scores, ca_scores, asr_scores):
    n = len(em_scores)
    mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
    return {'model': model_key, 'corpus': corpus_name, 'injection_template': injection_template, 'n': n, 'f1_clean': round(mean(f1_clean_scores), 4), 'exact_match': round(mean(em_scores), 4), 'f1_raw': round(mean(f1_raw_scores), 4), 'contains_answer_diagnostic': round(mean(ca_scores), 4), 'attack_success_rate': round(mean(asr_scores), 4)}
_SUMMARY_FIELDNAMES = ['model', 'corpus', 'injection_template', 'n', 'f1_clean', 'exact_match', 'f1_raw', 'contains_answer_diagnostic', 'attack_success_rate']

def _write_summary_csv(summary_path, summary_row):
    with _atomic_open(summary_path, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary_row)
if __name__ == '__main__':
    run_attack_sweep()
