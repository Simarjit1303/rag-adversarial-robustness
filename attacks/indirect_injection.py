import torch
from attacks.injection_templates import TEMPLATES
from data.build_index import build_index, retrieve
from data.normalize import extract_passage_text
from harness.model_loader import build_chat_prompt
from harness.pipeline import SYSTEM_PROMPT, clean_generation

def build_attack_user_prompt(index, records, question: str, corpus_name: str, injection_template: str, top_k: int=5):
    if injection_template not in TEMPLATES:
        raise ValueError(f"Unknown injection template '{injection_template}'. Options: {list(TEMPLATES)}")
    template = TEMPLATES[injection_template]
    retrieved = retrieve(index, records, question, k=top_k)
    if not retrieved:
        raise ValueError('No documents retrieved -- cannot inject into an empty top-k.')
    lines = []
    for i, (doc, score) in enumerate(retrieved):
        text = extract_passage_text(corpus_name, doc)
        if i == 0:
            text = template.render(text)
        lines.append(f'[{i + 1}] {text}')
    context = '\n\n'.join(lines)
    user_prompt = f'Context:\n{context}\n\nQuestion: {question}'
    return (user_prompt, retrieved, template.target_string, template.hijack_type)

def run_attack_query(model, tokenizer, model_key: str, corpus_name: str, question: str, injection_template: str, index=None, records=None, top_k: int=5, max_new_tokens: int=256):
    if index is None or records is None:
        index, records = build_index(corpus_name, split='dev')
    user_prompt, retrieved, target_string, hijack_type = build_attack_user_prompt(index, records, question, corpus_name, injection_template, top_k=top_k)
    prompt = build_chat_prompt(model_key, tokenizer, SYSTEM_PROMPT, user_prompt)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    generated = generated.strip()
    return {'question': question, 'retrieved_doc_ids': [idx for idx, _ in enumerate(retrieved)], 'retrieval_scores': [score for _, score in retrieved], 'generated_answer': generated, 'generated_answer_clean': clean_generation(generated), 'injection_template': injection_template, 'hijack_type': hijack_type, 'target_string': target_string}
