"""
Computes the max_model_len for the vLLM path from REAL built prompts —
never guess this value. (The 4096 used during Colab testing was picked for
T4 VRAM reasons, not because it reflects the real prompt length.)

The actual prompt shape is SYSTEM_PROMPT + TOP_K retrieved docs + question,
and chunk sizes differ across the three corpora, so this script renders the
same prompts the sweep would send — same retrieval, same chat template per
model — tokenizes them, and recommends a max_model_len that comfortably
covers the longest observed prompt plus the generation budget, with
headroom.

Run it where the corpora/indices exist (the A100 container or any machine
that has run data/loader.py + data/build_index.py):

    python -m scripts.compute_max_model_len

then copy the recommended value into config.RAG_VLLM_MAX_MODEL_LEN (or
export RAG_VLLM_MAX_MODEL_LEN). Honors RAG_MODELS the same way run_baseline
does.
The gated Llama tokenizer needs HF_TOKEN + accepted license; if any
tokenizer fails to load, the recommendation is flagged INCOMPLETE.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CORPORA, MODELS  # noqa: E402
from data.build_index import build_index  # noqa: E402
from data.normalize import extract_gold_answers, extract_question  # noqa: E402
from harness.model_loader import build_chat_prompt, load_tokenizer  # noqa: E402
from harness.pipeline import SYSTEM_PROMPT, build_rag_user_prompt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sample", type=int, default=0,
                        help="cap questions per corpus (0 = all)")
    parser.add_argument("--gen-budget", type=int, default=256,
                        help="generation budget to add on top of the prompt "
                             "(matches the sweep's max_new_tokens)")
    parser.add_argument("--headroom", type=float, default=1.10,
                        help="multiplier applied to (max prompt + budget) "
                             "before rounding up to a multiple of 256")
    args = parser.parse_args()

    model_keys = list(MODELS)
    if os.environ.get("RAG_MODELS"):
        model_keys = [m.strip() for m in os.environ["RAG_MODELS"].split(",") if m.strip()]

    # Render user prompts once per corpus (retrieval per question, exactly
    # as the sweep does), then tokenize per model.
    corpus_prompts = {}
    for corpus_name in CORPORA:
        index, records = build_index(corpus_name, split=args.split)
        prompts = []
        for record in records:
            question = extract_question(corpus_name, record)
            gold = extract_gold_answers(corpus_name, record)
            if not question or not gold:
                continue  # same filter as run_baseline — measure what runs
            user_prompt, _ = build_rag_user_prompt(index, records, question)
            prompts.append(user_prompt)
            if args.sample and len(prompts) >= args.sample:
                break
        corpus_prompts[corpus_name] = prompts
        print(f"[max_len] {corpus_name}: {len(prompts)} prompts rendered")

    overall_max = 0
    incomplete = []
    print(f"\n{'model':<16} {'corpus':<12} {'n':>5} {'mean':>7} {'p95':>7} {'max':>7}")
    for model_key in model_keys:
        try:
            tokenizer = load_tokenizer(model_key)
        except Exception as e:
            # Almost always the gated Llama tokenizer without HF_TOKEN /
            # accepted license. Skipping keeps the other measurements
            # useful, but the final recommendation is then incomplete.
            print(f"[max_len] SKIPPING {model_key}: tokenizer failed to load "
                  f"— {type(e).__name__}: {e}")
            incomplete.append(model_key)
            continue

        for corpus_name, prompts in corpus_prompts.items():
            lengths = sorted(
                len(tokenizer(
                    build_chat_prompt(model_key, tokenizer, SYSTEM_PROMPT, up),
                    add_special_tokens=False,
                ).input_ids)
                for up in prompts
            )
            if not lengths:
                continue
            mean = sum(lengths) / len(lengths)
            p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
            print(f"{model_key:<16} {corpus_name:<12} {len(lengths):>5} "
                  f"{mean:>7.0f} {p95:>7} {lengths[-1]:>7}")
            overall_max = max(overall_max, lengths[-1])

    if overall_max == 0:
        print("\nNo prompts measured — nothing to recommend.")
        return 1

    raw = (overall_max + args.gen_budget) * args.headroom
    recommendation = int(math.ceil(raw / 256) * 256)
    print(f"\nLongest observed prompt: {overall_max} tokens")
    print(f"Recommended max_model_len: {recommendation}  "
          f"(= ceil(({overall_max} + {args.gen_budget}) x {args.headroom} / 256) x 256)")
    if incomplete:
        print(f"WARNING: recommendation is INCOMPLETE — tokenizer(s) for "
              f"{incomplete} could not be loaded; their prompts may tokenize "
              f"longer. Re-run with all four tokenizers before trusting it.")
    print("Set config.RAG_VLLM_MAX_MODEL_LEN to this value "
          "(or export RAG_VLLM_MAX_MODEL_LEN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
