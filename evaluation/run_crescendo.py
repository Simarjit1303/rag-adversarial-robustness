import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
import torch
from attacks.crescendo import ATTACKER_MODEL, MAX_BACKTRACKS, RateLimitError, TARGET_SYSTEM_PROMPT, generate_attacker_turn, generate_judge_verdict, is_refusal, resolve_api_token, sample_behaviors
from config import MODELS, RESULTS_DIR, SEED
from data.behavior_pool import load_behavior_pool
from evaluation.result_paths import crescendo_result_file_paths, resolve_crescendo_sweep_selection, resolve_defense
from harness.model_loader import build_multiturn_chat_prompt, load_model

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

def _call_with_retry(fn, *args, max_attempts: int=3, label: str='', error_sink: dict=None, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            is_last = attempt == max_attempts - 1
            if not is_last:
                if isinstance(e, RateLimitError) and e.retry_after is not None:
                    delay = e.retry_after
                    delay_note = f'honoring Retry-After={delay}s'
                else:
                    delay = 2 ** (attempt + 1)
                    delay_note = f'backing off {delay}s'
            print(f'[crescendo] {label} attempt {attempt + 1}/{max_attempts} failed: {type(e).__name__}: {e}' + (f' -- {delay_note}' if not is_last else ' -- giving up'), file=sys.stderr)
            if not is_last:
                time.sleep(delay)
    print(f'[crescendo] SKIPPING {label} after {max_attempts} failed attempts (last error: {type(last_exc).__name__}: {last_exc})', file=sys.stderr)
    if error_sink is not None:
        error_sink['error'] = f'{label} (after {max_attempts} attempts): {type(last_exc).__name__}: {last_exc}'
    return None
TARGET_MAX_NEW_TOKENS = 512

def _generate_target_reply(model, tokenizer, model_key: str, history: list, attacker_turn: str) -> str:
    messages = [{'role': 'system', 'content': TARGET_SYSTEM_PROMPT}] + history + [{'role': 'user', 'content': attacker_turn}]
    prompt = build_multiturn_chat_prompt(model_key, tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=TARGET_MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def run_crescendo_conversation(model, tokenizer, model_key: str, target_behavior: str, api_token: str, max_turns: int, max_backtracks: int=MAX_BACKTRACKS, output_filter_log_path=None):
    history = []
    backtrack_count = 0
    refusal_count = 0
    backtrack_attempts = []
    output_filter_flags = []
    error_sink = {}
    for turn in range(max_turns):
        refusal_feedback = None
        attacker_turn = None
        target_reply = None
        pending_backtrack = None
        for attempt in range(max_backtracks + 1):
            attacker_turn = _call_with_retry(generate_attacker_turn, history, target_behavior, api_token=api_token, refusal_feedback=refusal_feedback, label=f'{model_key} attacker turn {turn + 1} attempt {attempt + 1}', error_sink=error_sink, context=model_key)
            if pending_backtrack is not None:
                pending_backtrack['retry_prompt'] = attacker_turn
                pending_backtrack = None
            if attacker_turn is None:
                break
            target_reply = _generate_target_reply(model, tokenizer, model_key, history, attacker_turn)
            if not is_refusal(target_reply):
                break
            refusal_count += 1
            if attempt < max_backtracks:
                backtrack_count += 1
                refusal_feedback = target_reply
                pending_backtrack = {'turn': turn + 1, 'refused_prompt': attacker_turn, 'refused_reply': target_reply, 'retry_prompt': None}
                backtrack_attempts.append(pending_backtrack)
        if attacker_turn is None:
            break
        if output_filter_log_path is not None:
            from defenses.output_filter import apply_output_filter, log_filter_event
            _, guard_record = apply_output_filter(target_reply)
            log_filter_event(output_filter_log_path, {'model': model_key, 'behavior': target_behavior, 'turn': turn + 1, **guard_record})
            output_filter_flags.append({'turn': turn + 1, 'flagged': guard_record['flagged'], 'guard_label': guard_record['guard_label']})
        history.append({'role': 'user', 'content': attacker_turn})
        history.append({'role': 'assistant', 'content': target_reply})
    return (history, backtrack_count, refusal_count, len(history) // 2, backtrack_attempts, error_sink.get('error'), output_filter_flags)

def run_crescendo_sweep(model_keys=None, max_turns: int=None, sample_n: int=None, api_token: str=None, defense=None):
    if sample_n is None:
        env_val = os.environ.get('RAG_SAMPLE_N')
        sample_n = int(env_val) if env_val else None
    if defense is None:
        defense = resolve_defense()
    if defense not in ('none', 'output_filter'):
        raise ValueError(f"Crescendo only supports defense='none' or 'output_filter' (instruction_detection/spotlighting don't apply -- no retrieved/injected content to filter), got '{defense}'")
    default_model_keys, default_max_turns, engine = resolve_crescendo_sweep_selection()
    if model_keys is None:
        model_keys = default_model_keys
    if max_turns is None:
        max_turns = default_max_turns
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f'Unknown model keys {unknown}. Options: {list(MODELS)}')
    if engine != 'hf':
        raise ValueError(f"INFERENCE_ENGINE must be 'hf' for Crescendo (vLLM path not yet implemented -- see this module's docstring), got '{engine}'")
    api_token = resolve_api_token(api_token)
    pool = load_behavior_pool()
    behaviors = sample_behaviors(pool, sample_size=sample_n or 100, seed=SEED)
    print(f"[crescendo] sampled {len(behaviors)} target behaviors ({sum((1 for b in behaviors if b['source'] == 'jbb_behaviors'))} jbb_behaviors, {sum((1 for b in behaviors if b['source'] == 'harmbench'))} harmbench)")
    if torch.cuda.is_available():
        print(f'[crescendo] Running on GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})')
    else:
        print('[crescendo] WARNING: CUDA not available -- the sweep will run on CPU.')
    model_switch_sleep = float(os.environ.get('CRESCENDO_MODEL_SWITCH_SLEEP_SECONDS', '0'))
    summary_rows = []
    for i, model_key in enumerate(model_keys):
        if i > 0 and model_switch_sleep > 0:
            print(f'[crescendo] CRESCENDO_MODEL_SWITCH_SLEEP_SECONDS diagnostic: sleeping {model_switch_sleep}s before loading {model_key}')
            time.sleep(model_switch_sleep)
        print(f'\n=== Loading {model_key} ===')
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f'[crescendo] SKIPPING {model_key}: failed to load -- {type(e).__name__}: {e}')
            continue
        raw_path, summary_path = crescendo_result_file_paths(RESULTS_DIR, model_key, max_turns, 'hf', defense)
        output_filter_log_path = RESULTS_DIR / f'output_filter_log_crescendo_{model_key}_{max_turns}turn.jsonl' if defense == 'output_filter' else None
        rows = []
        start = time.time()
        with open(raw_path, 'w', encoding='utf-8') as raw_f:
            for i, b in enumerate(behaviors):
                conversation, backtrack_count, refusal_count, n_turns, backtrack_attempts, conv_error, output_filter_flags = run_crescendo_conversation(model, tokenizer, model_key, b['behavior'], api_token, max_turns, output_filter_log_path=output_filter_log_path)
                judge_error_sink = {}
                verdict = _call_with_retry(generate_judge_verdict, conversation, b['behavior'], api_token=api_token, label=f"{model_key} judge for {b['behavior']!r}", error_sink=judge_error_sink, context=model_key) if conversation else None
                error = conv_error or judge_error_sink.get('error')
                row = _build_row(model_key, max_turns, b, conversation, backtrack_count, refusal_count, n_turns, verdict, backtrack_attempts, error, output_filter_flags)
                rows.append(row)
                raw_f.write(json.dumps(row, ensure_ascii=False) + '\n')
                raw_f.flush()
                os.fsync(raw_f.fileno())
                if (i + 1) % 10 == 0:
                    print(f'  {i + 1}/{len(behaviors)} done ({time.time() - start:.0f}s elapsed)', flush=True)
        summary_row = _summarize(model_key, max_turns, rows)
        summary_rows.append(summary_row)
        _write_summary_csv(summary_path, summary_row)
        print(f"  {model_key}: attack_success_rate={summary_row['attack_success_rate']:.4f}  (n_scored={summary_row['n_scored']}/{summary_row['n']}, n_judge_failed={summary_row['n_judge_failed']})  backtrack_rate={summary_row['backtrack_rate']:.4f}")
        print(f'  raw -> {raw_path}')
        print(f'  summary -> {summary_path}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary_rows

def _build_row(model_key, max_turns, behavior_item, conversation, backtrack_count, refusal_count, n_turns, verdict, backtrack_attempts=None, error=None, output_filter_flags=None):
    judge_failed = verdict is None
    return {'model': model_key, 'max_turns': max_turns, 'seed': SEED, 'attacker_model': ATTACKER_MODEL, 'behavior': behavior_item['behavior'], 'source': behavior_item['source'], 'conversation': conversation, 'n_turns_completed': n_turns, 'backtrack_count': backtrack_count, 'any_backtrack': int(backtrack_count > 0), 'backtrack_attempts': backtrack_attempts or [], 'refusal_count': refusal_count, 'judge_failed': int(judge_failed), 'attack_success': 0 if judge_failed else verdict['success'], 'judge_reasoning': None if judge_failed else verdict['reasoning'], 'output_filter_flags': output_filter_flags or [], 'error': error}

def _summarize(model_key, max_turns, rows):
    n = len(rows)
    scored = [r for r in rows if not r['judge_failed']]
    n_scored = len(scored)
    mean = lambda rows_, key: sum((r[key] for r in rows_)) / len(rows_) if rows_ else float('nan')
    return {'model': model_key, 'max_turns': max_turns, 'n': n, 'n_scored': n_scored, 'n_judge_failed': n - n_scored, 'attack_success_rate': round(mean(scored, 'attack_success'), 4), 'backtrack_rate': round(mean(rows, 'any_backtrack'), 4), 'mean_backtrack_count': round(mean(rows, 'backtrack_count'), 4), 'mean_refusal_count': round(mean(rows, 'refusal_count'), 4), 'mean_turns_completed': round(mean(rows, 'n_turns_completed'), 4)}
_SUMMARY_FIELDNAMES = ['model', 'max_turns', 'n', 'n_scored', 'n_judge_failed', 'attack_success_rate', 'backtrack_rate', 'mean_backtrack_count', 'mean_refusal_count', 'mean_turns_completed']

def _write_summary_csv(summary_path, summary_row):
    import csv
    with _atomic_open(summary_path, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary_row)
if __name__ == '__main__':
    run_crescendo_sweep()
