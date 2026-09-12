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
    RateLimitError,
    TARGET_SYSTEM_PROMPT,
    generate_attacker_turn,
    generate_judge_verdict,
    is_refusal,
    resolve_api_token,
    sample_behaviors,
)
from config import MODELS, RESULTS_DIR, SEED
from data.behavior_pool import load_behavior_pool
from evaluation.result_paths import crescendo_result_file_paths, resolve_crescendo_sweep_selection, resolve_defense
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


def _call_with_retry(fn, *args, max_attempts: int = 3, label: str = "", error_sink: dict = None, **kwargs):
    """
    Bounded retry with backoff -- identical shape to
    evaluation.run_poisonedrag._generate_poison_with_retry (a transient NIM
    failure gets a few attempts, not immediately fatal; does NOT re-raise
    after exhausting attempts, since one persistently-failing call must not
    crash an otherwise-complete sweep). Generalized over the specific
    function (generate_attacker_turn vs generate_judge_verdict share this
    retry policy, not the call itself). Returns fn(*args, **kwargs) or None
    on exhausted failure.

    Rate-limit-aware, not just generic backoff: real smoke-test finding,
    2026-09-09 -- a 4-model sweep hit 429 on every NIM call, and the old
    1s/2s backoff never let NVIDIA's ~40 req/min window clear (attacks/
    crescendo.py's _nim_chat now also PACES every call up front, so this is
    the reactive half). On a RateLimitError, respect NIM's own Retry-After
    header exactly when it sent one; otherwise fall back to a real
    exponential backoff (2s, 4s, ...) meaningfully higher than the old 1s/2s.

    error_sink, when given a dict, gets last_exc written to it as
    error_sink["error"] on exhausted failure -- same "skip-and-log" call,
    but the reason no longer lives ONLY in a stderr print a real pod run may
    not have captured. Real bug, smoke test 2026-09-09: phi-4-mini and
    ministral-3-8b got n_turns_completed=0/conversation=[] on every one of 3
    behaviors with no record anywhere of why -- judge_reasoning: null and
    nothing else. Callers thread this into the output row (see _build_row's
    "error" field) instead of only into this print.
    """
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
                    delay_note = f"honoring Retry-After={delay}s"
                else:
                    delay = 2 ** (attempt + 1)  # 2s, 4s, 8s, ... -- not the old 1s/2s
                    delay_note = f"backing off {delay}s"
            print(
                f"[crescendo] {label} attempt {attempt + 1}/{max_attempts} failed: "
                f"{type(e).__name__}: {e}"
                + (f" -- {delay_note}" if not is_last else " -- giving up"),
                file=sys.stderr,
            )
            if not is_last:
                time.sleep(delay)
    print(f"[crescendo] SKIPPING {label} after {max_attempts} failed attempts "
          f"(last error: {type(last_exc).__name__}: {last_exc})", file=sys.stderr)
    if error_sink is not None:
        # label is the OUTER (backtrack-attempt) label from run_crescendo_conversation
        # and reads "attempt 1" whenever the very first outer attempt is the one that
        # exhausts retries -- it doesn't by itself say max_attempts inner retries ran.
        # Real bug, 2026-09-10: every phi-4-mini/ministral-3-8b error read "attempt 1"
        # even though this function had genuinely retried 3 times, reading as if only
        # one attempt was ever made. Append the real inner retry count explicitly.
        error_sink["error"] = f"{label} (after {max_attempts} attempts): {type(last_exc).__name__}: {last_exc}"
    return None


TARGET_MAX_NEW_TOKENS = 512  # real bug, 2026-09-10 (round 7): the project-wide baseline
# default (256, shared by harness/vllm_engine.py's generate_vllm and the single-shot
# attacks/poisonedrag.py, attacks/indirect_injection.py calls) was reused here too, but
# Crescendo's target reply is answering an already-escalated turn-5 prompt with growing
# conversation context, not a short baseline QA answer -- a real ministral-3-8b turn-5
# reply was observed cut off mid-word ("often **prior") at exactly the 256-token ceiling.
# Bumped for THIS call only, not the shared default: no context-length ceiling is at
# risk (load_model has no max_length/truncation cap -- see harness/model_loader.py), so
# this only spends a bit more local GPU time, never API cost.


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
        output_ids = model.generate(**inputs, max_new_tokens=TARGET_MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_crescendo_conversation(model, tokenizer, model_key: str, target_behavior: str,
                                api_token: str, max_turns: int, max_backtracks: int = MAX_BACKTRACKS,
                                output_filter_log_path=None):
    """
    Runs one full Crescendo conversation against `model`. Returns
    (conversation, backtrack_count, refusal_count, n_turns_completed,
    backtrack_attempts, error, output_filter_flags). conversation is the
    final [{"role", "content"}, ...] thread (excluding the system turn) --
    what generate_judge_verdict scores; the linear history only ever holds
    the FINAL (non-refused) reply for each turn, so a backtracked turn's
    refused attempt is otherwise invisible in the stored data.

    backtrack_attempts preserves what the linear history discards: one
    {"turn", "refused_prompt", "refused_reply", "retry_prompt"} entry per
    backtrack, so the mechanism is auditable from the raw data itself
    (retry_prompt is None if the regenerated attacker turn itself then
    exhausted retries, ending the conversation before a retry could run).

    error is None on a clean run, else the label + exception string from
    whichever attacker-turn _call_with_retry call last exhausted its
    attempts (see that function's error_sink) -- the real cause when a
    conversation ends early/empty, not just a bare n_turns_completed=0.

    output_filter_log_path: when given, defenses.output_filter.run_output_filter
    is called on each turn's FINAL (accepted, non-refused) target reply --
    per-turn, not just the final conversation -- so mechanism attribution
    (which turn the guard would have blocked) is possible, per
    defenses/output_filter.py's module docstring. Deliberately does NOT
    feed the filtered/blocked text back into `history`: Crescendo's own
    escalation logic (is_refusal, backtracking) must keep operating on the
    real target reply, since this defense measures what a deployed guard
    WOULD have blocked, without itself changing the attack's trajectory.
    output_filter_flags collects one {"turn", "flagged", "guard_label"}
    entry per turn actually checked.
    """
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
        pending_backtrack = None  # the entry awaiting this attempt's retry_prompt

        for attempt in range(max_backtracks + 1):
            attacker_turn = _call_with_retry(
                generate_attacker_turn, history, target_behavior,
                api_token=api_token, refusal_feedback=refusal_feedback,
                label=f"{model_key} attacker turn {turn + 1} attempt {attempt + 1}",
                error_sink=error_sink, context=model_key,
            )
            if pending_backtrack is not None:
                pending_backtrack["retry_prompt"] = attacker_turn
                pending_backtrack = None
            if attacker_turn is None:
                break  # attacker generation itself exhausted retries -- end conversation early

            target_reply = _generate_target_reply(model, tokenizer, model_key, history, attacker_turn)
            if not is_refusal(target_reply):
                break

            refusal_count += 1
            if attempt < max_backtracks:
                backtrack_count += 1
                refusal_feedback = target_reply
                pending_backtrack = {
                    "turn": turn + 1,
                    "refused_prompt": attacker_turn,
                    "refused_reply": target_reply,
                    "retry_prompt": None,
                }
                backtrack_attempts.append(pending_backtrack)
            # else: backtrack cap reached -- accept the refusal reply and move on,
            # same "bounded retry then give up" discipline as poisonedrag's cap

        if attacker_turn is None:
            break

        if output_filter_log_path is not None:
            from defenses.output_filter import apply_output_filter, log_filter_event
            # Classify only -- deliberately discard the filtered/blocked
            # text (see this function's docstring: history keeps the real
            # reply so Crescendo's own escalation logic is unaffected).
            _, guard_record = apply_output_filter(target_reply)
            log_filter_event(output_filter_log_path, {
                "model": model_key, "behavior": target_behavior, "turn": turn + 1, **guard_record,
            })
            output_filter_flags.append({
                "turn": turn + 1, "flagged": guard_record["flagged"], "guard_label": guard_record["guard_label"],
            })

        history.append({"role": "user", "content": attacker_turn})
        history.append({"role": "assistant", "content": target_reply})

    return (history, backtrack_count, refusal_count, len(history) // 2, backtrack_attempts,
            error_sink.get("error"), output_filter_flags)


def run_crescendo_sweep(model_keys=None, max_turns: int = None, sample_n: int = None,
                         api_token: str = None, defense=None):
    """
    sample_n: caps target behaviors for the smoke-test path (falls back to
    RAG_SAMPLE_N, same semantics as the other two sweeps) -- verify the full
    pipeline (attacker call, target generation, backtrack, judge call, file
    output) on a handful of behaviors before committing the real n=100 sweep's
    NIM and GPU budget.

    defense: "none" or "output_filter" only (falls back to RAG_DEFENSE via
    resolve_defense()) -- instruction_detection/spotlighting don't apply to
    Crescendo, a confirmed design decision (no retrieved/injected content to
    filter here; see evaluation/result_paths.py's crescendo_result_file_paths
    docstring).
    """
    if sample_n is None:
        env_val = os.environ.get("RAG_SAMPLE_N")
        sample_n = int(env_val) if env_val else None
    if defense is None:
        defense = resolve_defense()
    if defense not in ("none", "output_filter"):
        raise ValueError(
            f"Crescendo only supports defense='none' or 'output_filter' "
            f"(instruction_detection/spotlighting don't apply -- no retrieved/"
            f"injected content to filter), got '{defense}'"
        )

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

    api_token = resolve_api_token(api_token)

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

    # Crude, fast diagnostic -- NOT the real fix -- per user's explicit ask
    # 2026-09-10: a third consecutive real sweep hit 429 on all 9 attempts
    # (3 behaviors x 3 retries) for the two tail models even with genuine
    # backoff sleeps running, which points at the limiter's per-model
    # reset/carryover accounting rather than the per-call pacing math.
    # A flat sleep BEFORE loading each model after the first, independent of
    # _wait_for_rate_limit_slot's own logic entirely, isolates whether real
    # elapsed wall-clock time between model switches alone would have fixed
    # it. 0 (off) unless set -- this must never silently change default
    # sweep behavior.
    model_switch_sleep = float(os.environ.get("CRESCENDO_MODEL_SWITCH_SLEEP_SECONDS", "0"))

    summary_rows = []
    for i, model_key in enumerate(model_keys):
        if i > 0 and model_switch_sleep > 0:
            print(f"[crescendo] CRESCENDO_MODEL_SWITCH_SLEEP_SECONDS diagnostic: "
                  f"sleeping {model_switch_sleep}s before loading {model_key}")
            time.sleep(model_switch_sleep)
        print(f"\n=== Loading {model_key} ===")
        try:
            model, tokenizer = load_model(model_key)
        except Exception as e:
            print(f"[crescendo] SKIPPING {model_key}: failed to load -- {type(e).__name__}: {e}")
            continue

        raw_path, summary_path = crescendo_result_file_paths(RESULTS_DIR, model_key, max_turns, "hf", defense)
        output_filter_log_path = (
            RESULTS_DIR / f"output_filter_log_crescendo_{model_key}_{max_turns}turn.jsonl"
            if defense == "output_filter" else None
        )
        rows = []
        start = time.time()
        # Per-behavior checkpointing, NOT _atomic_open: _atomic_open buffers
        # the whole model's sweep in a temp file and only os.replace()'s it
        # into place when the `with` block exits normally -- an interruption
        # (crash, OOM, pod terminated mid-model) at behavior N loses ALL N
        # completed rows, not just the tail, because raw_path never comes
        # into existence until the model finishes every behavior. Writing
        # directly to raw_path and fsyncing after each row trades that
        # all-or-nothing atomicity for the opposite: raw_path grows visibly
        # on disk as the sweep proceeds, so a future interruption only costs
        # behaviors not yet completed, not the whole model.
        with open(raw_path, "w", encoding="utf-8") as raw_f:
            for i, b in enumerate(behaviors):
                conversation, backtrack_count, refusal_count, n_turns, backtrack_attempts, conv_error, output_filter_flags = run_crescendo_conversation(
                    model, tokenizer, model_key, b["behavior"], api_token, max_turns,
                    output_filter_log_path=output_filter_log_path,
                )

                judge_error_sink = {}
                verdict = _call_with_retry(
                    generate_judge_verdict, conversation, b["behavior"],
                    api_token=api_token, label=f"{model_key} judge for {b['behavior']!r}",
                    error_sink=judge_error_sink, context=model_key,
                ) if conversation else None

                # conv_error takes precedence -- it explains WHY conversation
                # may be empty/truncated in the first place; a judge error
                # only ever applies when the conversation itself succeeded.
                error = conv_error or judge_error_sink.get("error")

                row = _build_row(model_key, max_turns, b, conversation, backtrack_count,
                                  refusal_count, n_turns, verdict, backtrack_attempts, error,
                                  output_filter_flags)
                rows.append(row)
                raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_f.flush()
                os.fsync(raw_f.fileno())

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
                refusal_count, n_turns, verdict, backtrack_attempts=None, error=None,
                output_filter_flags=None):
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
        # The discarded/refused attempt(s) the linear `conversation` history
        # can't show -- see run_crescendo_conversation's docstring. Real bug,
        # smoke test 2026-09-09: a row with backtrack_count=1 had every
        # assistant turn in `conversation` comply readily, because the
        # refused reply never survives the backtrack -- only the final
        # rephrasing does. This makes the mechanism auditable from the raw
        # data itself instead of a bare count with no evidence behind it.
        "backtrack_attempts": backtrack_attempts or [],
        "refusal_count": refusal_count,
        "judge_failed": int(judge_failed),
        "attack_success": 0 if judge_failed else verdict["success"],
        "judge_reasoning": None if judge_failed else verdict["reasoning"],
        # Per-turn Llama-Guard verdicts (defenses/output_filter.py), for
        # mechanism attribution -- which turn a deployed guard would have
        # blocked. Empty when defense != "output_filter" (see
        # run_crescendo_conversation's docstring for why the raw reply, not
        # the filtered one, is what stays in `conversation` above).
        "output_filter_flags": output_filter_flags or [],
        # Real bug, smoke test 2026-09-09: phi-4-mini/ministral-3-8b rows had
        # n_turns_completed=0, judge_reasoning=null, and NO record anywhere
        # of why -- undiagnosable from the JSONL alone. error carries the
        # last exhausted _call_with_retry failure (attacker-turn generation
        # takes precedence over a judge failure -- see run_crescendo_sweep),
        # None when nothing failed.
        "error": error,
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
