import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
GUARD_MODEL_ID = 'meta-llama/Llama-Guard-4-12B'
GUARD_MODEL_REVISION = '87acb4b94e930c3d679e6e7ee9d57e2feab9ea71'
REFUSAL_MARKER = '[OUTPUT_FILTER_BLOCKED] This response was withheld by the Llama-Guard output filter.'
_SAFE_UNSAFE_RE = re.compile('^\\s*(safe|unsafe)', re.IGNORECASE)
_guard_cache = {}

def _disable_llama4_chunked_attention(model_config):
    text_config = model_config.get_text_config(decoder=True)
    layer_types = getattr(text_config, 'layer_types', None)
    if getattr(text_config, 'attention_chunk_size', None) is None and layer_types:
        text_config.layer_types = ['full_attention' if lt == 'chunked_attention' else lt for lt in layer_types]
        text_config.attention_chunk_size = text_config.max_position_embeddings

def load_guard_model(device_map='auto', dtype=torch.bfloat16):
    if 'model' not in _guard_cache:
        tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL_ID, revision=GUARD_MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(GUARD_MODEL_ID, revision=GUARD_MODEL_REVISION, device_map=device_map, dtype=dtype)
        model.eval()
        _disable_llama4_chunked_attention(model.config)
        _guard_cache['model'] = model
        _guard_cache['tokenizer'] = tokenizer
    return (_guard_cache['model'], _guard_cache['tokenizer'])

def unload_guard_model():
    _guard_cache.pop('model', None)
    _guard_cache.pop('tokenizer', None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def classify_response(response_text: str, model=None, tokenizer=None, max_new_tokens: int=20):
    if model is None or tokenizer is None:
        model, tokenizer = load_guard_model()
    conversation = [{'role': 'user', 'content': [{'type': 'text', 'text': response_text}]}]
    inputs = tokenizer.apply_chat_template(conversation, return_tensors='pt', add_generation_prompt=True, return_dict=True)
    inputs = inputs.to(model.device) if hasattr(inputs, 'to') else inputs
    past_key_values = None
    config = getattr(model, 'config', None)
    if config is not None:
        past_key_values = DynamicCache(config=config.get_text_config(decoder=True))
    generate_kwargs = {'max_new_tokens': max_new_tokens, 'pad_token_id': tokenizer.eos_token_id}
    if past_key_values is not None:
        generate_kwargs['past_key_values'] = past_key_values
        generate_kwargs['cache_implementation'] = None
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
    generated_ids = output[0][inputs['input_ids'].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    match = _SAFE_UNSAFE_RE.match(raw_output)
    if match is None:
        return (False, 'unparseable', raw_output)
    label = match.group(1).lower()
    return (label == 'unsafe', label, raw_output)

def apply_output_filter(response_text: str, model=None, tokenizer=None):
    flagged, label, raw_output = classify_response(response_text, model=model, tokenizer=tokenizer)
    final_response = REFUSAL_MARKER if flagged else response_text
    record = {'flagged': flagged, 'guard_label': label, 'guard_raw_output': raw_output}
    return (final_response, record)

def log_filter_event(log_path, record: dict):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def run_output_filter(response_text: str, log_path, row_id: dict, model=None, tokenizer=None):
    final_response, record = apply_output_filter(response_text, model=model, tokenizer=tokenizer)
    log_filter_event(log_path, {**row_id, **record})
    return final_response
