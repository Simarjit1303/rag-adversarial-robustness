import argparse
import math
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CORPORA, MODELS, TOP_K
from data.build_index import build_index
from data.normalize import extract_gold_answers, extract_question
from harness.model_loader import build_chat_prompt, load_tokenizer
from harness.pipeline import SYSTEM_PROMPT, build_rag_user_prompt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'attack'], default='baseline', help="'baseline' measures Phase 1 prompts (default); 'attack' measures Phase 2 indirect-injection prompts across every (corpus, injection_template)")
    parser.add_argument('--split', default='dev')
    parser.add_argument('--sample', type=int, default=0, help='cap questions per corpus (or per corpus x injection_template combo, in --mode attack) (0 = all)')
    parser.add_argument('--gen-budget', type=int, default=256, help="generation budget to add on top of the prompt (matches the sweep's max_new_tokens)")
    parser.add_argument('--headroom', type=float, default=1.1, help='multiplier applied to (max prompt + budget) before rounding up to a multiple of 256')
    args = parser.parse_args()
    model_keys = list(MODELS)
    if os.environ.get('RAG_MODELS'):
        model_keys = [m.strip() for m in os.environ['RAG_MODELS'].split(',') if m.strip()]
    corpus_names = list(CORPORA)
    if os.environ.get('RAG_CORPORA'):
        corpus_names = [c.strip() for c in os.environ['RAG_CORPORA'].split(',') if c.strip()]
    corpus_prompts = {}
    if args.mode == 'baseline':
        for corpus_name in corpus_names:
            index, records = build_index(corpus_name, split=args.split)
            prompts = []
            for record in records:
                question = extract_question(corpus_name, record)
                gold = extract_gold_answers(corpus_name, record)
                if not question or not gold:
                    continue
                user_prompt, _ = build_rag_user_prompt(index, records, question, corpus_name)
                prompts.append(user_prompt)
                if args.sample and len(prompts) >= args.sample:
                    break
            corpus_prompts[corpus_name] = prompts
            print(f'[max_len] {corpus_name}: {len(prompts)} prompts rendered')
    else:
        from attacks.indirect_injection import build_attack_user_prompt
        from attacks.injection_templates import TEMPLATES
        from evaluation.result_paths import ATTACK_ELIGIBLE_CORPORA
        for corpus_name in ATTACK_ELIGIBLE_CORPORA:
            index, records = build_index(corpus_name, split=args.split)
            for injection_template in TEMPLATES:
                prompts = []
                for record in records:
                    question = extract_question(corpus_name, record)
                    gold = extract_gold_answers(corpus_name, record)
                    if not question or not gold:
                        continue
                    user_prompt, _, _, _ = build_attack_user_prompt(index, records, question, corpus_name, injection_template, top_k=TOP_K)
                    prompts.append(user_prompt)
                    if args.sample and len(prompts) >= args.sample:
                        break
                key = f'{corpus_name}/{injection_template}'
                corpus_prompts[key] = prompts
                print(f'[max_len] {key}: {len(prompts)} prompts rendered')
    overall_max = 0
    incomplete = []
    print(f"\n{'model':<16} {'corpus':<28} {'n':>5} {'mean':>7} {'p95':>7} {'max':>7}")
    for model_key in model_keys:
        try:
            tokenizer = load_tokenizer(model_key)
        except Exception as e:
            print(f'[max_len] SKIPPING {model_key}: tokenizer failed to load — {type(e).__name__}: {e}')
            incomplete.append(model_key)
            continue
        for corpus_name, prompts in corpus_prompts.items():
            lengths = sorted((len(tokenizer(build_chat_prompt(model_key, tokenizer, SYSTEM_PROMPT, up), add_special_tokens=False).input_ids) for up in prompts))
            if not lengths:
                continue
            mean = sum(lengths) / len(lengths)
            p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
            print(f'{model_key:<16} {corpus_name:<28} {len(lengths):>5} {mean:>7.0f} {p95:>7} {lengths[-1]:>7}')
            overall_max = max(overall_max, lengths[-1])
    if overall_max == 0:
        print('\nNo prompts measured — nothing to recommend.')
        return 1
    raw = (overall_max + args.gen_budget) * args.headroom
    recommendation = int(math.ceil(raw / 256) * 256)
    print(f'\nLongest observed prompt: {overall_max} tokens')
    print(f'Recommended max_model_len: {recommendation}  (= ceil(({overall_max} + {args.gen_budget}) x {args.headroom} / 256) x 256)')
    if incomplete:
        print(f'WARNING: recommendation is INCOMPLETE — tokenizer(s) for {incomplete} could not be loaded; their prompts may tokenize longer. Re-run with all four tokenizers before trusting it.')
    print('Set config.RAG_VLLM_MAX_MODEL_LEN to this value (or export RAG_VLLM_MAX_MODEL_LEN).')
    return 0
if __name__ == '__main__':
    sys.exit(main())
