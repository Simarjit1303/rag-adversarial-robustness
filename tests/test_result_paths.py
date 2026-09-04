"""
baseline_raw.jsonl and baseline_summary.csv were written to the same fixed
path regardless of which model, corpus, or engine produced them -- a vLLM
run silently overwrote a prior HF-path run's results with no error. These
tests cover evaluation/result_paths.py, the single source of truth for the
per-(model, corpus, engine) naming that fixes it.
"""

import config
import evaluation.result_paths as rp


def test_result_file_paths_are_namespaced_by_model_corpus_and_engine(tmp_path):
    raw_a, summary_a = rp.result_file_paths(tmp_path, "phi-4-mini", "nq_open", "hf")
    raw_b, summary_b = rp.result_file_paths(tmp_path, "phi-4-mini", "nq_open", "vllm")

    # same model/corpus, different engine -- must be distinct files, not
    # the same fixed name a second run would silently overwrite
    assert raw_a != raw_b
    assert summary_a != summary_b
    assert raw_a.name == "baseline_raw_phi-4-mini_nq_open_hf.jsonl"
    assert summary_a.name == "baseline_summary_phi-4-mini_nq_open_hf.csv"


def test_resolve_sweep_selection_defaults_to_full_matrix(monkeypatch):
    monkeypatch.delenv("RAG_MODELS", raising=False)
    monkeypatch.delenv("RAG_CORPORA", raising=False)
    monkeypatch.delenv("INFERENCE_ENGINE", raising=False)

    model_keys, corpus_names, engine = rp.resolve_sweep_selection()

    assert model_keys == list(config.MODELS)
    assert corpus_names == list(config.CORPORA)
    assert engine == "hf"


def test_resolve_sweep_selection_honors_env_vars(monkeypatch):
    monkeypatch.setenv("RAG_MODELS", "phi-4-mini, qwen3-8b")
    monkeypatch.setenv("RAG_CORPORA", "nq_open")
    monkeypatch.setenv("INFERENCE_ENGINE", "vllm")

    model_keys, corpus_names, engine = rp.resolve_sweep_selection()

    assert model_keys == ["phi-4-mini", "qwen3-8b"]
    assert corpus_names == ["nq_open"]
    assert engine == "vllm"


def test_expected_result_files_covers_every_cell_in_the_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MODELS", "phi-4-mini,qwen3-8b")
    monkeypatch.setenv("RAG_CORPORA", "nq_open,ms_marco")
    monkeypatch.setenv("INFERENCE_ENGINE", "hf")

    paths = rp.expected_result_files(tmp_path)

    # 2 models x 2 corpora x 2 files (raw + summary) each = 8
    assert len(paths) == 8
    names = {p.name for p in paths}
    assert "baseline_raw_phi-4-mini_nq_open_hf.jsonl" in names
    assert "baseline_summary_qwen3-8b_ms_marco_hf.csv" in names


# --------------------------------------------------------------------------
# Attack sweep (Phase 2, indirect prompt injection) -- extends the naming
# above with an injection_template axis. nq_open is excluded by default:
# see nq_open_leakage_finding.md and config.py's CORPORA comment.
# --------------------------------------------------------------------------

def test_attack_eligible_corpora_excludes_nq_open():
    assert "nq_open" not in rp.ATTACK_ELIGIBLE_CORPORA
    assert set(rp.ATTACK_ELIGIBLE_CORPORA) == set(config.CORPORA) - {"nq_open"}


def test_attack_result_file_paths_are_namespaced_by_all_four_axes(tmp_path):
    raw_a, summary_a = rp.attack_result_file_paths(tmp_path, "phi-4-mini", "hotpot_qa", "naive", "hf")
    raw_b, summary_b = rp.attack_result_file_paths(tmp_path, "phi-4-mini", "hotpot_qa", "combined", "hf")

    assert raw_a != raw_b  # different template -- must not collide
    assert raw_a.name == "attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl"
    assert summary_a.name == "attack_summary_phi-4-mini_hotpot_qa_naive_hf.csv"


def test_resolve_attack_sweep_selection_defaults_exclude_nq_open_and_include_all_templates(monkeypatch):
    monkeypatch.delenv("RAG_MODELS", raising=False)
    monkeypatch.delenv("RAG_CORPORA", raising=False)
    monkeypatch.delenv("RAG_INJECTION_TEMPLATES", raising=False)
    monkeypatch.delenv("INFERENCE_ENGINE", raising=False)

    model_keys, corpus_names, injection_templates, engine = rp.resolve_attack_sweep_selection()

    assert model_keys == list(config.MODELS)
    assert corpus_names == rp.ATTACK_ELIGIBLE_CORPORA
    assert "nq_open" not in corpus_names
    assert set(injection_templates) == {"naive", "escape_char", "ignore", "fake_completion", "combined"}
    assert engine == "hf"


def test_resolve_attack_sweep_selection_honors_env_vars(monkeypatch):
    monkeypatch.setenv("RAG_MODELS", "qwen3-8b")
    monkeypatch.setenv("RAG_CORPORA", "ms_marco")
    monkeypatch.setenv("RAG_INJECTION_TEMPLATES", "naive, combined")
    monkeypatch.setenv("INFERENCE_ENGINE", "vllm")

    model_keys, corpus_names, injection_templates, engine = rp.resolve_attack_sweep_selection()

    assert model_keys == ["qwen3-8b"]
    assert corpus_names == ["ms_marco"]
    assert injection_templates == ["naive", "combined"]
    assert engine == "vllm"


def test_resolve_attack_sweep_selection_explicit_nq_open_still_works(monkeypatch):
    # a default that excludes nq_open, not a hard block -- an explicit
    # RAG_CORPORA=nq_open is the caller's informed choice
    monkeypatch.setenv("RAG_CORPORA", "nq_open")
    _, corpus_names, _, _ = rp.resolve_attack_sweep_selection()
    assert corpus_names == ["nq_open"]


def test_expected_attack_result_files_covers_every_cell(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MODELS", "phi-4-mini")
    monkeypatch.setenv("RAG_CORPORA", "hotpot_qa,ms_marco")
    monkeypatch.setenv("RAG_INJECTION_TEMPLATES", "naive,combined")
    monkeypatch.setenv("INFERENCE_ENGINE", "hf")

    paths = rp.expected_attack_result_files(tmp_path)

    # 1 model x 2 corpora x 2 templates x 2 files (raw + summary) = 8
    assert len(paths) == 8
    names = {p.name for p in paths}
    assert "attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl" in names
    assert "attack_summary_phi-4-mini_ms_marco_combined_hf.csv" in names
