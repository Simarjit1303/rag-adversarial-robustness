import torch
from harness.model_loader import load_model, build_chat_prompt
_vision_call_count = 0

def _hook(module, inputs, output):
    global _vision_call_count
    _vision_call_count += 1

def verify_text_only(n_prompts: int=20):
    model, tokenizer = load_model('ministral-3-8b')
    vision_module = getattr(model, 'vision_tower', None) or getattr(model, 'vision_model', None)
    if vision_module is None:
        raise AttributeError('Could not find a vision submodule on the loaded model. Run `print(model)` and update this script with the correct attribute path.')
    handle = vision_module.register_forward_hook(_hook)
    sample_questions = ['What is the capital of France?', 'Summarize the following passage.'] * (n_prompts // 2)
    for q in sample_questions[:n_prompts]:
        prompt = build_chat_prompt('ministral-3-8b', tokenizer, 'You are a helpful assistant.', q)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
    handle.remove()
    if _vision_call_count == 0:
        print(f'PASS: vision tower was never invoked across {n_prompts} text-only prompts.')
    else:
        print(f'FAIL: vision tower fired {_vision_call_count} times on text-only input. The text-only caveat in the dissertation needs to be revisited.')
if __name__ == '__main__':
    verify_text_only()
