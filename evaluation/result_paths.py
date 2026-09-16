import os
from pathlib import Path
from config import CORPORA, MODELS
ATTACK_ELIGIBLE_CORPORA = [c for c in CORPORA if c != 'nq_open']
DEFENSE_OPTIONS = ('none', 'instruction_detection', 'spotlighting', 'output_filter')

def _defense_suffix(defense: str) -> str:
    if defense not in DEFENSE_OPTIONS:
        raise ValueError(f"Unknown defense '{defense}'. Options: {list(DEFENSE_OPTIONS)}")
    return f'_defense-{defense}' if defense != 'none' else ''

def resolve_defense():
    defense = os.environ.get('RAG_DEFENSE', 'none')
    if defense not in DEFENSE_OPTIONS:
        raise ValueError(f"RAG_DEFENSE must be one of {list(DEFENSE_OPTIONS)}, got '{defense}'")
    return defense

def resolve_sweep_selection():
    model_keys_env = os.environ.get('RAG_MODELS')
    model_keys = [m.strip() for m in model_keys_env.split(',') if m.strip()] if model_keys_env else list(MODELS)
    corpus_names_env = os.environ.get('RAG_CORPORA')
    corpus_names = [c.strip() for c in corpus_names_env.split(',') if c.strip()] if corpus_names_env else list(CORPORA)
    engine = os.environ.get('INFERENCE_ENGINE', 'hf')
    return (model_keys, corpus_names, engine)

def result_file_paths(results_dir, model_key: str, corpus_name: str, engine: str):
    results_dir = Path(results_dir)
    raw_path = results_dir / f'baseline_raw_{model_key}_{corpus_name}_{engine}.jsonl'
    summary_path = results_dir / f'baseline_summary_{model_key}_{corpus_name}_{engine}.csv'
    return (raw_path, summary_path)

def expected_result_files(results_dir):
    model_keys, corpus_names, engine = resolve_sweep_selection()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            paths.extend(result_file_paths(results_dir, model_key, corpus_name, engine))
    return paths

def resolve_attack_sweep_selection():
    model_keys_env = os.environ.get('RAG_MODELS')
    model_keys = [m.strip() for m in model_keys_env.split(',') if m.strip()] if model_keys_env else list(MODELS)
    corpus_names_env = os.environ.get('RAG_CORPORA')
    corpus_names = [c.strip() for c in corpus_names_env.split(',') if c.strip()] if corpus_names_env else list(ATTACK_ELIGIBLE_CORPORA)
    templates_env = os.environ.get('RAG_INJECTION_TEMPLATES')
    if templates_env:
        injection_templates = [t.strip() for t in templates_env.split(',') if t.strip()]
    else:
        from attacks.injection_templates import TEMPLATES
        injection_templates = list(TEMPLATES)
    engine = os.environ.get('INFERENCE_ENGINE', 'hf')
    return (model_keys, corpus_names, injection_templates, engine)

def attack_result_file_paths(results_dir, model_key: str, corpus_name: str, injection_template: str, engine: str, defense: str='none'):
    results_dir = Path(results_dir)
    suffix = _defense_suffix(defense)
    raw_path = results_dir / f'attack_raw_{model_key}_{corpus_name}_{injection_template}_{engine}{suffix}.jsonl'
    summary_path = results_dir / f'attack_summary_{model_key}_{corpus_name}_{injection_template}_{engine}{suffix}.csv'
    return (raw_path, summary_path)

def expected_attack_result_files(results_dir, defense: str=None):
    model_keys, corpus_names, injection_templates, engine = resolve_attack_sweep_selection()
    if defense is None:
        defense = resolve_defense()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            for injection_template in injection_templates:
                paths.extend(attack_result_file_paths(results_dir, model_key, corpus_name, injection_template, engine, defense))
    return paths
DEFAULT_POISON_CONFIG = 'adv5'

def resolve_poison_sweep_selection():
    model_keys_env = os.environ.get('RAG_MODELS')
    model_keys = [m.strip() for m in model_keys_env.split(',') if m.strip()] if model_keys_env else list(MODELS)
    corpus_names_env = os.environ.get('RAG_CORPORA')
    corpus_names = [c.strip() for c in corpus_names_env.split(',') if c.strip()] if corpus_names_env else list(ATTACK_ELIGIBLE_CORPORA)
    poison_configs_env = os.environ.get('RAG_POISON_CONFIGS')
    poison_configs = [p.strip() for p in poison_configs_env.split(',') if p.strip()] if poison_configs_env else [DEFAULT_POISON_CONFIG]
    engine = os.environ.get('INFERENCE_ENGINE', 'hf')
    return (model_keys, corpus_names, poison_configs, engine)

def poison_result_file_paths(results_dir, model_key: str, corpus_name: str, poison_config: str, engine: str, defense: str='none'):
    results_dir = Path(results_dir)
    suffix = _defense_suffix(defense)
    raw_path = results_dir / f'poison_raw_{model_key}_{corpus_name}_{poison_config}_{engine}{suffix}.jsonl'
    summary_path = results_dir / f'poison_summary_{model_key}_{corpus_name}_{poison_config}_{engine}{suffix}.csv'
    return (raw_path, summary_path)

def expected_poison_result_files(results_dir, defense: str=None):
    model_keys, corpus_names, poison_configs, engine = resolve_poison_sweep_selection()
    if defense is None:
        defense = resolve_defense()
    paths = []
    for model_key in model_keys:
        for corpus_name in corpus_names:
            for poison_config in poison_configs:
                paths.extend(poison_result_file_paths(results_dir, model_key, corpus_name, poison_config, engine, defense))
    return paths
DEFAULT_MAX_TURNS = 5

def resolve_crescendo_sweep_selection():
    model_keys_env = os.environ.get('RAG_MODELS')
    model_keys = [m.strip() for m in model_keys_env.split(',') if m.strip()] if model_keys_env else list(MODELS)
    max_turns_env = os.environ.get('RAG_MAX_TURNS')
    max_turns = int(max_turns_env) if max_turns_env else DEFAULT_MAX_TURNS
    engine = os.environ.get('INFERENCE_ENGINE', 'hf')
    return (model_keys, max_turns, engine)

def crescendo_result_file_paths(results_dir, model_key: str, max_turns: int, engine: str, defense: str='none'):
    results_dir = Path(results_dir)
    suffix = _defense_suffix(defense)
    raw_path = results_dir / f'crescendo_raw_{model_key}_{max_turns}turn_{engine}{suffix}.jsonl'
    summary_path = results_dir / f'crescendo_summary_{model_key}_{max_turns}turn_{engine}{suffix}.csv'
    return (raw_path, summary_path)

def expected_crescendo_result_files(results_dir, defense: str=None):
    model_keys, max_turns, engine = resolve_crescendo_sweep_selection()
    if defense is None:
        defense = resolve_defense()
    paths = []
    for model_key in model_keys:
        paths.extend(crescendo_result_file_paths(results_dir, model_key, max_turns, engine, defense))
    return paths
