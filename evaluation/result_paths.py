"""
Canonical per-(model, corpus, engine) result file paths.

Bug: baseline_raw.jsonl and baseline_summary.csv were written to the same
fixed path regardless of which model, corpus, or engine produced them.
Confirmed directly via the Network Volume's S3 API: after a vLLM run, only
one baseline_raw.jsonl existed, timestamped to match that run exactly --
the prior HF-path run's data was silently overwritten with no error, no
warning. The planned Phase 1 sweep is 4 models x 3 corpora, likely split
across multiple pods/runs via RAG_MODELS/RAG_CORPORA -- every cell writing
to the same filename means only the last one to finish survives.

This module is the single source of truth for that naming, shared by:
  - evaluation/run_baseline.py, which writes these files
  - scripts/run_and_terminate.py, which verifies they exist before
    terminating the pod

Deliberately has NO heavy dependencies (no torch/transformers/vllm) so
run_and_terminate.py -- the lightweight orchestrator that must stay
importable even if the heavy ML libs it supervises are broken -- can import
it safely without pulling those in.
"""

import os
from pathlib import Path

from config import CORPORA, MODELS

# Corpora eligible for the indirect-injection attack sweep (and Phase 2/3
# sweeps generally going forward). nq_open is excluded: it has no
# independent supporting passage, so its "passage" text is the gold answer
# by construction -- unfixable gold-answer leakage, see
# nq_open_leakage_finding.md and config.py's CORPORA comment. This is a
# real default (not just a comment) so an attack sweep invoked without an
# explicit RAG_CORPORA doesn't silently include it.
ATTACK_ELIGIBLE_CORPORA = [c for c in CORPORA if c != "nq_open"]


def resolve_sweep_selection():
    """
    Which model keys, corpus names, and engine a sweep run will touch, read
    from the exact same RAG_MODELS/RAG_CORPORA/INFERENCE_ENGINE env vars
    evaluation.run_baseline.run_baseline_sweep() itself reads. Kept here
    (not duplicated in run_and_terminate.py) so the two can never drift
    apart on what "this run's cells" means.
    """
    model_keys_env = os.environ.get("RAG_MODELS")
    model_keys = (
        [m.strip() for m in model_keys_env.split(",") if m.strip()]
        if model_keys_env else list(MODELS)
    )
    corpus_names_env = os.environ.get("RAG_CORPORA")
    corpus_names = (
        [c.strip() for c in corpus_names_env.split(",") if c.strip()]
        if corpus_names_env else list(CORPORA)
    )
    engine = os.environ.get("INFERENCE_ENGINE", "hf")
    return model_keys, corpus_names, engine


def result_file_paths(results_dir, model_key: str, corpus_name: str, engine: str):
    """
    The (raw_jsonl, summary_csv) paths one (model, corpus, engine) cell
    writes. Namespaced by all three so separate runs -- different engines,
    different RAG_MODELS/RAG_CORPORA subsets (e.g. one pod per cell for a
    parallel sweep) -- never silently overwrite each other's results on the
    shared Network Volume.

    results_dir is taken as a parameter (not imported from config) so
    callers -- and tests -- control it directly rather than going through a
    shared mutable module global.
    """
    results_dir = Path(results_dir)
    raw_path = results_dir / f"baseline_raw_{model_key}_{corpus_name}_{engine}.jsonl"
    summary_path = results_dir / f"baseline_summary_{model_key}_{corpus_name}_{engine}.csv"
    return raw_path, summary_path


def expected_result_files(results_dir):
    """
    Every (raw, summary) path this process's env-var configuration will
    produce -- used by scripts/run_and_terminate.py's verify_success() to
    confirm a sweep actually wrote every cell it claimed to run, not just
    the last one.
    """
    model_keys, corpus_names, engine = resolve_sweep_selection()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            paths.extend(result_file_paths(results_dir, model_key, corpus_name, engine))
    return paths


def resolve_attack_sweep_selection():
    """
    Same shape as resolve_sweep_selection(), plus injection_templates, for
    evaluation.run_attack_injection.run_attack_sweep(). Model/engine env
    vars are shared with the baseline sweep (RAG_MODELS/INFERENCE_ENGINE) --
    same reasoning as resolve_sweep_selection's docstring. RAG_CORPORA
    defaults to ATTACK_ELIGIBLE_CORPORA here, not the full CORPORA list --
    an attack sweep invoked without an explicit RAG_CORPORA must not
    silently include nq_open. An explicit RAG_CORPORA=nq_open still works
    (this is a default, not a hard block) -- it will just be scored
    against a corpus already known to leak, which is on the caller.
    """
    model_keys_env = os.environ.get("RAG_MODELS")
    model_keys = (
        [m.strip() for m in model_keys_env.split(",") if m.strip()]
        if model_keys_env else list(MODELS)
    )
    corpus_names_env = os.environ.get("RAG_CORPORA")
    corpus_names = (
        [c.strip() for c in corpus_names_env.split(",") if c.strip()]
        if corpus_names_env else list(ATTACK_ELIGIBLE_CORPORA)
    )
    templates_env = os.environ.get("RAG_INJECTION_TEMPLATES")
    if templates_env:
        injection_templates = [t.strip() for t in templates_env.split(",") if t.strip()]
    else:
        # Imported here, not at module top, to preserve this module's "no
        # heavy dependencies" property for anything that only needs the
        # baseline-sweep functions above -- injection_templates.py itself
        # has no heavy deps either, but this keeps the import graph
        # minimal for run_and_terminate.py's sake regardless.
        from attacks.injection_templates import TEMPLATES
        injection_templates = list(TEMPLATES)
    engine = os.environ.get("INFERENCE_ENGINE", "hf")
    return model_keys, corpus_names, injection_templates, engine


def attack_result_file_paths(results_dir, model_key: str, corpus_name: str,
                              injection_template: str, engine: str):
    """
    The (raw_jsonl, summary_csv) paths one (model, corpus, injection_template,
    engine) cell writes. Extends result_file_paths' naming with the
    injection_template axis, same atomic-write pattern as the baseline
    sweep (evaluation/run_baseline.py's _atomic_open) -- no exceptions.
    """
    results_dir = Path(results_dir)
    raw_path = (
        results_dir
        / f"attack_raw_{model_key}_{corpus_name}_{injection_template}_{engine}.jsonl"
    )
    summary_path = (
        results_dir
        / f"attack_summary_{model_key}_{corpus_name}_{injection_template}_{engine}.csv"
    )
    return raw_path, summary_path


def expected_attack_result_files(results_dir):
    """
    Every (raw, summary) path this process's env-var configuration will
    produce for the attack sweep -- same role as expected_result_files(),
    for a future scripts/run_and_terminate.py extension to verify an attack
    sweep's completeness before terminating a paid pod.
    """
    model_keys, corpus_names, injection_templates, engine = resolve_attack_sweep_selection()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            for injection_template in injection_templates:
                paths.extend(
                    attack_result_file_paths(
                        results_dir, model_key, corpus_name, injection_template, engine
                    )
                )
    return paths
