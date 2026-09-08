"""
One-off recompute: rewrites already-collected raw JSONL result files' EM/F1/
contains_answer fields using the fixed evaluation.metrics.normalize_text
(see the "Fix normalize_text" commit). Pure reprocessing of each row's
already-stored generated_answer / generated_answer_clean / gold_answers
fields -- no re-generation, no GPU, no API calls. attack_success (ASR,
phase2_injection_results only) is left untouched: attacks/asr_scoring.py's
score_asr is a raw substring check that never calls normalize_text.

Rewrites raw JSONL atomically (same temp-file + os.replace() pattern used
throughout this codebase) and regenerates each cell's summary CSV from the
recomputed rows, with the exact fieldnames evaluation/run_baseline.py and
evaluation/run_attack_injection.py already write.

Usage: python -m scripts.recompute_metrics_after_normalize_fix [--dry-run]
"""

import argparse
import csv
import glob
import io
import json
import os
import tempfile

from evaluation.metrics import contains_answer, exact_match, f1_score

BASELINE_DIR = "phase1_results_complete"
ATTACK_DIR = "phase2_injection_results"

BASELINE_SUMMARY_FIELDS = ["model", "corpus", "n", "f1_clean", "exact_match",
                           "f1_raw", "contains_answer_diagnostic"]
ATTACK_SUMMARY_FIELDS = ["model", "corpus", "injection_template", "n",
                          "f1_clean", "exact_match", "f1_raw",
                          "contains_answer_diagnostic", "attack_success_rate"]


def _atomic_write_text(path, text):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def recompute_row(row: dict) -> dict:
    """Mutates and returns row with EM/F1/contains_answer recomputed from
    its own already-stored generated_answer/generated_answer_clean/
    gold_answers fields. attack_success (if present) is left alone."""
    gold = row["gold_answers"]
    gen = row["generated_answer"]
    gen_clean = row["generated_answer_clean"]
    row["exact_match"] = exact_match(gen_clean, gold)
    row["f1_clean"] = f1_score(gen_clean, gold)
    row["f1_raw"] = f1_score(gen, gold)
    row["contains_answer_diagnostic"] = contains_answer(gen, gold)
    return row


def _summarize(rows: list[dict], extra_keys: list[str]) -> dict:
    n = len(rows)
    mean = lambda key: (sum(r[key] for r in rows) / n) if n else float("nan")
    summary = {k: rows[0][k] for k in extra_keys} if rows else {}
    summary["n"] = n
    summary["f1_clean"] = round(mean("f1_clean"), 4)
    summary["exact_match"] = round(mean("exact_match"), 4)
    summary["f1_raw"] = round(mean("f1_raw"), 4)
    summary["contains_answer_diagnostic"] = round(mean("contains_answer_diagnostic"), 4)
    if rows and "attack_success" in rows[0]:
        summary["attack_success_rate"] = round(mean("attack_success"), 4)
    return summary


def recompute_file(raw_path: str, summary_path: str, extra_keys: list[str],
                    summary_fields: list[str], dry_run: bool) -> tuple[int, int]:
    with open(raw_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    changed = 0
    for row in rows:
        before = (row["exact_match"], row["f1_clean"], row["f1_raw"],
                   row["contains_answer_diagnostic"])
        recompute_row(row)
        after = (row["exact_match"], row["f1_clean"], row["f1_raw"],
                  row["contains_answer_diagnostic"])
        if before != after:
            changed += 1

    if not dry_run:
        raw_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        _atomic_write_text(raw_path, raw_text)

        summary_row = _summarize(rows, extra_keys)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(summary_row)
        _atomic_write_text(summary_path, buf.getvalue())

    return len(rows), changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="report what would change without writing anything")
    args = parser.parse_args()

    total_files = 0
    total_rows = 0
    total_changed = 0

    for raw_path in sorted(glob.glob(os.path.join(BASELINE_DIR, "baseline_raw_*.jsonl"))):
        summary_path = raw_path.replace("baseline_raw_", "baseline_summary_").replace(".jsonl", ".csv")
        n, changed = recompute_file(raw_path, summary_path, ["model", "corpus"],
                                     BASELINE_SUMMARY_FIELDS, args.dry_run)
        total_files += 1
        total_rows += n
        total_changed += changed
        if changed:
            print(f"  {raw_path}: {changed}/{n} rows changed")

    for raw_path in sorted(glob.glob(os.path.join(ATTACK_DIR, "attack_raw_*.jsonl"))):
        summary_path = raw_path.replace("attack_raw_", "attack_summary_").replace(".jsonl", ".csv")
        n, changed = recompute_file(raw_path, summary_path,
                                     ["model", "corpus", "injection_template"],
                                     ATTACK_SUMMARY_FIELDS, args.dry_run)
        total_files += 1
        total_rows += n
        total_changed += changed
        if changed:
            print(f"  {raw_path}: {changed}/{n} rows changed")

    print(f"\n{'[dry-run] ' if args.dry_run else ''}"
          f"{total_files} files, {total_rows} rows, {total_changed} rows changed")


if __name__ == "__main__":
    main()
