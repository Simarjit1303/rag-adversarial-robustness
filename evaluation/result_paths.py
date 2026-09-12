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

# ---------------------------------------------------------------------------
# Phase 3 defense axis -- opt-in per sweep run via RAG_DEFENSE, shared by all
# three attack-sweep result-path families below (attack/poison/crescendo).
# "none" is the default and produces the EXACT undefended filename (no
# suffix) so this axis is purely additive: nothing already on disk (Phase
# 1/2 baselines, existing undefended attack/poison/crescendo results) can
# collide with or be silently overwritten by a defended run.
# ---------------------------------------------------------------------------
DEFENSE_OPTIONS = ("none", "instruction_detection", "spotlighting", "output_filter")


def _defense_suffix(defense: str) -> str:
    if defense not in DEFENSE_OPTIONS:
        raise ValueError(f"Unknown defense '{defense}'. Options: {list(DEFENSE_OPTIONS)}")
    return f"_defense-{defense}" if defense != "none" else ""


def resolve_defense():
    """Which Phase 3 defense (if any) a sweep run applies, read from
    RAG_DEFENSE (default "none" = undefended baseline). Kept here, next to
    resolve_*_sweep_selection, so every runner resolves this identically."""
    defense = os.environ.get("RAG_DEFENSE", "none")
    if defense not in DEFENSE_OPTIONS:
        raise ValueError(f"RAG_DEFENSE must be one of {list(DEFENSE_OPTIONS)}, got '{defense}'")
    return defense


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
                              injection_template: str, engine: str, defense: str = "none"):
    """
    The (raw_jsonl, summary_csv) paths one (model, corpus, injection_template,
    engine, defense) cell writes. Extends result_file_paths' naming with the
    injection_template axis, same atomic-write pattern as the baseline
    sweep (evaluation/run_baseline.py's _atomic_open) -- no exceptions.
    defense="none" (the default) produces the exact pre-Phase-3 filename;
    see DEFENSE_OPTIONS/_defense_suffix above for why.
    """
    results_dir = Path(results_dir)
    suffix = _defense_suffix(defense)
    raw_path = (
        results_dir
        / f"attack_raw_{model_key}_{corpus_name}_{injection_template}_{engine}{suffix}.jsonl"
    )
    summary_path = (
        results_dir
        / f"attack_summary_{model_key}_{corpus_name}_{injection_template}_{engine}{suffix}.csv"
    )
    return raw_path, summary_path


def expected_attack_result_files(results_dir, defense: str = None):
    """
    Every (raw, summary) path this process's env-var configuration will
    produce for the attack sweep -- same role as expected_result_files(),
    for a future scripts/run_and_terminate.py extension to verify an attack
    sweep's completeness before terminating a paid pod. defense defaults to
    resolve_defense() (RAG_DEFENSE, "none" if unset) when not given.
    """
    model_keys, corpus_names, injection_templates, engine = resolve_attack_sweep_selection()
    if defense is None:
        defense = resolve_defense()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            for injection_template in injection_templates:
                paths.extend(
                    attack_result_file_paths(
                        results_dir, model_key, corpus_name, injection_template, engine, defense
                    )
                )
    return paths


# ---------------------------------------------------------------------------
# PoisonedRAG (Attack 2) naming -- extends the {model}_{corpus}_{engine}
# pattern with a poison_config axis (mirrors injection_template's role for
# Attack 1), per PHASE2_ROADMAP.md's "Filenames carry the full cell
# identity" carry-forward rule. poison_config is an opaque label, not a
# hardcoded count, so a future poison-count sweep (1, 3, 5 -- flagged as an
# optional extension in the roadmap, not required for the confirmed
# condition) slots into this same naming without a redesign.

# adv5 = 5 poisoned texts per target question, confirmed at the Meeting 3
# checkpoint, matching PoisonedRAG's own paper default. Kept as a plain
# constant here (not imported from attacks.poisonedrag) so this module keeps
# its "no heavy dependencies" property -- attacks.poisonedrag needs
# `requests` for the poison-generation API call, this module must not.
DEFAULT_POISON_CONFIG = "adv5"


def resolve_poison_sweep_selection():
    """
    Same shape as resolve_attack_sweep_selection(), for
    evaluation.run_poisonedrag.run_poisonedrag_sweep(). RAG_MODELS/
    INFERENCE_ENGINE are shared with the other two sweeps (same reasoning as
    resolve_sweep_selection's docstring). RAG_CORPORA defaults to
    ATTACK_ELIGIBLE_CORPORA -- nq_open excluded by default here too, same
    reasoning as the injection sweep (unfixable gold-answer leakage, see
    nq_open_leakage_finding.md).
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
    poison_configs_env = os.environ.get("RAG_POISON_CONFIGS")
    poison_configs = (
        [p.strip() for p in poison_configs_env.split(",") if p.strip()]
        if poison_configs_env else [DEFAULT_POISON_CONFIG]
    )
    engine = os.environ.get("INFERENCE_ENGINE", "hf")
    return model_keys, corpus_names, poison_configs, engine


def poison_result_file_paths(results_dir, model_key: str, corpus_name: str,
                              poison_config: str, engine: str, defense: str = "none"):
    """
    The (raw_jsonl, summary_csv) paths one (model, corpus, poison_config,
    engine, defense) cell writes. Same atomic-write pattern as the other two
    sweeps (evaluation/run_baseline.py's _atomic_open) -- no exceptions.
    defense="none" (the default) produces the exact pre-Phase-3 filename.
    """
    results_dir = Path(results_dir)
    suffix = _defense_suffix(defense)
    raw_path = (
        results_dir
        / f"poison_raw_{model_key}_{corpus_name}_{poison_config}_{engine}{suffix}.jsonl"
    )
    summary_path = (
        results_dir
        / f"poison_summary_{model_key}_{corpus_name}_{poison_config}_{engine}{suffix}.csv"
    )
    return raw_path, summary_path


def expected_poison_result_files(results_dir, defense: str = None):
    """
    Every (raw, summary) path this process's env-var configuration will
    produce for the PoisonedRAG sweep -- same role as
    expected_attack_result_files().
    """
    model_keys, corpus_names, poison_configs, engine = resolve_poison_sweep_selection()
    if defense is None:
        defense = resolve_defense()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            for poison_config in poison_configs:
                paths.extend(
                    poison_result_file_paths(
                        results_dir, model_key, corpus_name, poison_config, engine, defense
                    )
                )
    return paths


# ---------------------------------------------------------------------------
# Crescendo (Attack 3) naming -- extends the {model}_{engine} pattern with a
# turn_count axis in place of the other two sweeps' corpus/poison_config
# axis. No corpus dimension here (per phase2_crescendo_task.md's Scope
# section: Crescendo is a direct conversational attack on the target model,
# not a RAG-corpus attack), so this doesn't reuse
# {result,attack,poison}_result_file_paths' 4-part shape -- a 3-part
# {model}_{turn_count}_{engine} pattern fits Crescendo's actual axes instead
# of forcing a corpus placeholder into a naming scheme that has none.

DEFAULT_MAX_TURNS = 5


def resolve_crescendo_sweep_selection():
    """
    Same shape as resolve_poison_sweep_selection(), for
    evaluation.run_crescendo.run_crescendo_sweep(). RAG_MODELS/
    INFERENCE_ENGINE are shared with the other sweeps (same reasoning as
    resolve_sweep_selection's docstring). RAG_MAX_TURNS lets a smoke test
    override the fixed max_turns=5 (see phase2_crescendo_task.md's Tasks
    section) without touching config.
    """
    model_keys_env = os.environ.get("RAG_MODELS")
    model_keys = (
        [m.strip() for m in model_keys_env.split(",") if m.strip()]
        if model_keys_env else list(MODELS)
    )
    max_turns_env = os.environ.get("RAG_MAX_TURNS")
    max_turns = int(max_turns_env) if max_turns_env else DEFAULT_MAX_TURNS
    engine = os.environ.get("INFERENCE_ENGINE", "hf")
    return model_keys, max_turns, engine


def crescendo_result_file_paths(results_dir, model_key: str, max_turns: int, engine: str,
                                 defense: str = "none"):
    """
    The (raw_jsonl, summary_csv) paths one (model, max_turns, engine, defense)
    cell writes. Same atomic-write pattern as the other two sweeps
    (evaluation/run_baseline.py's _atomic_open) -- no exceptions. defense
    is "none" or "output_filter" for Crescendo -- instruction_detection and
    spotlighting don't apply here (no retrieved/injected content to filter).
    defense="none" (the default) produces the exact pre-Phase-3 filename.
    """
    results_dir = Path(results_dir)
    suffix = _defense_suffix(defense)
    raw_path = results_dir / f"crescendo_raw_{model_key}_{max_turns}turn_{engine}{suffix}.jsonl"
    summary_path = results_dir / f"crescendo_summary_{model_key}_{max_turns}turn_{engine}{suffix}.csv"
    return raw_path, summary_path


def expected_crescendo_result_files(results_dir, defense: str = None):
    """
    Every (raw, summary) path this process's env-var configuration will
    produce for the Crescendo sweep -- same role as
    expected_poison_result_files().
    """
    model_keys, max_turns, engine = resolve_crescendo_sweep_selection()
    if defense is None:
        defense = resolve_defense()
    paths = []
    for model_key in model_keys:
        paths.extend(crescendo_result_file_paths(results_dir, model_key, max_turns, engine, defense))
    return paths
