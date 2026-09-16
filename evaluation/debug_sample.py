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
    model_keys = [m.strip() for m in os.environ.get('RAG_MODELS', '').split(',') if m.strip()]
    model_keys = model_keys or list(MODELS)
    corpus_names = [c.strip() for c in os.environ.get('RAG_CORPORA', '').split(',') if c.strip()]
    corpus_names = corpus_names or list(CORPORA)
    n = int(os.environ.get('RAG_DEBUG_N', '20'))
    for model_key in model_keys:
        print(f'\n=== Loading {model_key} ===')
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f'[debug_sample] SKIPPING {model_key}: failed to load — {type(e).__name__}: {e}')
            continue
        for corpus_name in corpus_names:
            print(f'--- {model_key} x {corpus_name}: first {n} answers ---')
            index, records = build_index(corpus_name, split='dev')
            shown = 0
            for record in records:
                if shown >= n:
                    break
                question = extract_question(corpus_name, record)
                gold = extract_gold_answers(corpus_name, record)
                if not question or not gold:
                    continue
                result = run_query(model, tokenizer, model_key, corpus_name, question, index=index, records=records)
                answer = result['generated_answer']
                answer_clean = result['generated_answer_clean']
                print(json.dumps({'model': model_key, 'corpus': corpus_name, 'i': shown, 'question': question, 'gold': gold, 'generated_answer': answer, 'generated_answer_clean': answer_clean, 'f1_clean': round(f1_score(answer_clean, gold), 4), 'f1_raw': round(f1_score(answer, gold), 4), 'exact_match': exact_match(answer_clean, gold), 'contains_answer_diagnostic': contains_answer(answer, gold)}, ensure_ascii=False))
                shown += 1
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print('\n[debug_sample] done')
if __name__ == '__main__':
    main()
