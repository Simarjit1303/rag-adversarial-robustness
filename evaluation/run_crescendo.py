"""
Phase 2, Attack 3 of 3: Crescendo multi-turn escalation sweep.

No Phase A/B split like PoisonedRAG (evaluation/run_poisonedrag.py) --
Crescendo's escalation genuinely depends on each TARGET model's own replies
turn by turn, so there is nothing model-agnostic to precompute and share.
One live conversation per (model, behavior): attacker turn (NIM, remote) ->
target reply (local GPU) -> rule-based refusal check -> optional backtrack
(regenerate the attacker turn, capped) -> next turn -> after max_turns,
one judge call (NIM) scores the whole conversation.

HF engine only for now -- the real PoisonedRAG sweep this project sized
Crescendo's GPU cost against also ran engine="hf" (see
poison_raw_*_hf.jsonl), and Crescendo's per-conversation sequential
dependency (each turn needs the PREVIOUS turn's target reply) doesn't batch
the way vLLM's generate_batch() does for the other two sweeps' independent
prompts -- a vLLM path would need cross-conversation turn-index batching,
a real optimization but out of scope for the initial build. Add when a
real sweep's wall-clock demands it.

Same atomic-write discipline as evaluation/run_poisonedrag.py's
_atomic_open (kept byte-for-byte in sync, not imported, same reasoning as
that module's own docstring).
"""

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager

import torch

from attacks.crescendo import (
    ATTACKER_MODEL,
    MAX_BACKTRACKS,
    TARGET_SYSTEM_PROMPT,
    generate_attacker_turn,
    generate_judge_verdict,
    is_refusal,
    sample_behaviors,
)
from config import MODELS, RESULTS_DIR, SEED
from data.behavior_pool import load_behavior_pool
from evaluation.result_paths import crescendo_result_file_paths, resolve_crescendo_sweep_selection
from harness.model_loader import build_multiturn_chat_prompt, load_model


@contextmanager
def _atomic_open(final_path, newline=None):
    """Identical to evaluation.run_poisonedrag._atomic_open -- see that
    module's docstring for the full rationale."""
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


def _call_with_retry(fn, *args, max_attempts: int = 3, label: str = "", **kwargs):
    """
    Bounded retry with backoff -- identical shape to
    evaluation.run_poisonedrag._generate_poison_with_retry (a transient NIM
    failure gets a few attempts, not immediately fatal; does NOT re-raise
    after exhausting attempts, since one persistently-failing call must not
    crash an otherwise-complete sweep). Generalized over the specific
    function (generate_attacker_turn vs generate_judge_verdict share this
    retry policy, not the call itself). Returns fn(*args, **kwargs) or None
    on exhausted failure.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            print(
                f"[crescendo] {label} attempt {attempt + 1}/{max_attempts} failed: "
                f"{type(e).__name__}: {e}"
                + (" -- retrying" if attempt < max_attempts - 1 else " -- giving up"),
                file=sys.stderr,
            )
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # 1s, 2s
    print(f"[crescendo] SKIPPING {label} after {max_attempts} failed attempts "
          f"(last error: {type(last_exc).__name__}: {last_exc})", file=sys.stderr)
    return None


def _generate_target_reply(model, tokenizer, model_key: str, history: list, attacker_turn: str) -> str:
    """One local generation call: target's reply to the next attacker turn,
    given the full conversation so far (conversation-state threading)."""
    messages = (
        [{"role": "system", "content": TARGET_SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": attacker_turn}]
    )
    prompt = build_multiturn_chat_prompt(model_key, tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_crescendo_conversation(model, tokenizer, model_key: str, target_behavior: str,
                                api_token: str, max_turns: int, max_backtracks: int = MAX_BACKTRACKS):
    """
    Runs one full Crescendo conversation against `model`. Returns
    (conversation, backtrack_count, refusal_count, n_turns_completed).
    conversation is the final [{"role", "content"}, ...] thread (excluding
    the system turn) -- what generate_judge_verdict scores.
    """
    history = []
    backtrack_count = 0
    refusal_count = 0

    for turn in range(max_turns):
        refusal_feedback = None
        attacker_turn = None
        target_reply = None

        for attempt in range(max_backtracks + 1):
            attacker_turn = _call_with_retry(
                generate_attacker_turn, history, target_behavior,
                api_token=api_token, refusal_feedback=refusal_feedback,
                label=f"{model_key} attacker turn {turn + 1} attempt {attempt + 1}",
            )
            if attacker_turn is None:
                break  # attacker generation itself exhausted retries -- end conversation early

            target_reply = _generate_target_reply(model, tokenizer, model_key, history, attacker_turn)
            if not is_refusal(target_reply):
                break

            refusal_count += 1
            if attempt < max_backtracks:
                backtrack_count += 1
                refusal_feedback = target_reply
            # else: backtrack cap reached -- accept the refusal reply and move on,
            # same "bounded retry then give up" discipline as poisonedrag's cap

        if attacker_turn is None:
            break

        history.append({"role": "user", "content": attacker_turn})
        history.append({"role": "assistant", "content": target_reply})

    return history, backtrack_count, refusal_count, len(history) // 2


def run_crescendo_sweep(model_keys=None, max_turns: int = None, sample_n: int = None,
                         api_token: str = None):
    """
    sample_n: caps target behaviors for the smoke-test path (falls back to
    RAG_SAMPLE_N, same semantics as the other two sweeps) -- verify the full
    pipeline (attacker call, target generation, backtrack, judge call, file
    output) on a handful of behaviors before committing the real n=100 sweep's
    NIM and GPU budget.
    """
    if sample_n is None:
        env_val = os.environ.get("RAG_SAMPLE_N")
        sample_n = int(env_val) if env_val else None

    default_model_keys, default_max_turns, engine = resolve_crescendo_sweep_selection()
    if model_keys is None:
        model_keys = default_model_keys
    if max_turns is None:
        max_turns = default_max_turns

    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys {unknown}. Options: {list(MODELS)}")
    if engine != "hf":
        raise ValueError(
            f"INFERENCE_ENGINE must be 'hf' for Crescendo (vLLM path not yet "
            f"implemented -- see this module's docstring), got '{engine}'"
        )

    api_token = api_token or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_token:
        raise RuntimeError("No NVIDIA_NIM_API_KEY set -- required for the attacker+judge model.")

    pool = load_behavior_pool()
    behaviors = sample_behaviors(pool, sample_size=sample_n or 100, seed=SEED)
    print(f"[crescendo] sampled {len(behaviors)} target behaviors "
          f"({sum(1 for b in behaviors if b['source'] == 'jbb_behaviors')} jbb_behaviors, "
          f"{sum(1 for b in behaviors if b['source'] == 'harmbench')} harmbench)")

    if torch.cuda.is_available():
        print(f"[crescendo] Running on GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda})")
    else:
        print("[crescendo] WARNING: CUDA not available -- the sweep will run on CPU.")

    summary_rows = []
    for model_key in model_keys:
        print(f"\n=== Loading {model_key} ===")
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f"[crescendo] SKIPPING {model_key}: failed to load -- {type(e).__name__}: {e}")
            continue

        raw_path, summary_path = crescendo_result_file_paths(RESULTS_DIR, model_key, max_turns, "hf")
        rows = []
        start = time.time()
        with _atomic_open(raw_path) as raw_f:
            for i, b in enumerate(behaviors):
                conversation, backtrack_count, refusal_count, n_turns = run_crescendo_conversation(
                    model, tokenizer, model_key, b["behavior"], api_token, max_turns
                )

                verdict = _call_with_retry(
                    generate_judge_verdict, conversation, b["behavior"],
                    api_token=api_token, label=f"{model_key} judge for {b['behavior']!r}",
                ) if conversation else None

                row = _build_row(model_key, max_turns, b, conversation, backtrack_count,
                                  refusal_count, n_turns, verdict)
                rows.append(row)
                raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(behaviors)} done ({time.time() - start:.0f}s elapsed)")

        summary_row = _summarize(model_key, max_turns, rows)
        summary_rows.append(summary_row)
        _write_summary_csv(summary_path, summary_row)
        # attack_success_rate's denominator is n_scored (judge_failed rows
        # excluded, see _summarize), NOT n -- print both so this line can't
        # be misread as "n out of n" the way the smoke-test console output
        # was (n=2 printed next to an ASR that was really 1/1, one judge
        # call having failed all 3 parse attempts).
        print(f"  {model_key}: attack_success_rate={summary_row['attack_success_rate']:.4f}  "
              f"(n_scored={summary_row['n_scored']}/{summary_row['n']}, "
              f"n_judge_failed={summary_row['n_judge_failed']})  "
              f"backtrack_rate={summary_row['backtrack_rate']:.4f}")
        print(f"  raw -> {raw_path}")
        print(f"  summary -> {summary_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_rows


def _build_row(model_key, max_turns, behavior_item, conversation, backtrack_count,
                refusal_count, n_turns, verdict):
    judge_failed = verdict is None
    return {
        "model": model_key,
        "max_turns": max_turns,
        "seed": SEED,
        "attacker_model": ATTACKER_MODEL,
        "behavior": behavior_item["behavior"],
        "source": behavior_item["source"],
        "conversation": conversation,
        "n_turns_completed": n_turns,
        "backtrack_count": backtrack_count,
        "any_backtrack": int(backtrack_count > 0),
        "refusal_count": refusal_count,
        "judge_failed": int(judge_failed),
        "attack_success": 0 if judge_failed else verdict["success"],
        "judge_reasoning": None if judge_failed else verdict["reasoning"],
    }


def _summarize(model_key, max_turns, rows):
    n = len(rows)
    # judge_failed / conversation-aborted rows are excluded from the ASR
    # denominator (a failed API call is not a target-model outcome), same
    # "skip-and-log, don't silently count as failure" precedent as
    # PoisonedRAG's build_poisoned_contexts skip handling.
    scored = [r for r in rows if not r["judge_failed"]]
    n_scored = len(scored)
    mean = lambda rows_, key: (sum(r[key] for r in rows_) / len(rows_)) if rows_ else float("nan")
    return {
        "model": model_key,
        "max_turns": max_turns,
        "n": n,
        "n_scored": n_scored,
        "n_judge_failed": n - n_scored,
        "attack_success_rate": round(mean(scored, "attack_success"), 4),
        "backtrack_rate": round(mean(rows, "any_backtrack"), 4),
        "mean_backtrack_count": round(mean(rows, "backtrack_count"), 4),
        "mean_refusal_count": round(mean(rows, "refusal_count"), 4),
        "mean_turns_completed": round(mean(rows, "n_turns_completed"), 4),
    }


_SUMMARY_FIELDNAMES = [
    "model", "max_turns", "n", "n_scored", "n_judge_failed",
    "attack_success_rate", "backtrack_rate", "mean_backtrack_count",
    "mean_refusal_count", "mean_turns_completed",
]


def _write_summary_csv(summary_path, summary_row):
    import csv
    with _atomic_open(summary_path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary_row)


if __name__ == "__main__":
    run_crescendo_sweep()
