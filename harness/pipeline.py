import ast
import re
import torch
from data.build_index import build_index, retrieve
from data.normalize import extract_passage_text
from harness.model_loader import build_chat_prompt
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question using only the information in the provided context. If the context does not contain the answer, say you don't know. Answer with only the exact answer span: no full sentence, no explanation, no formatting."
PREAMBLE_PATTERNS = ['^the answer is[:\\s]*', '^the correct answer is[:\\s]*', '^according to (the )?(context|passage|document)[,:]?\\s*', '^the context (provided |states|indicates|says)[^,\\.]*(states|indicates|says|that)[,:]?\\s*']

def _try_extract_leaked_answer(text: str):
    dict_match = re.search("\\{.*'answer':\\s*\\[.*?\\].*\\}", text)
    if dict_match:
        try:
            parsed = ast.literal_eval(dict_match.group(0))
            answer = parsed.get('answer')
            if isinstance(answer, list) and answer:
                return str(answer[0])
        except (ValueError, SyntaxError):
            pass
    stripped = text.strip()
    if re.match('^\\[\\s*[\'\\"]', stripped):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                return parsed[0]
        except (ValueError, SyntaxError):
            pass
    return None

def clean_generation(text):
    extracted = _try_extract_leaked_answer(text)
    if extracted is not None:
        text = extracted
    text = re.sub('\\*\\*(.*?)\\*\\*', '\\1', text)
    text = re.sub('\\*(.*?)\\*', '\\1', text)
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub('</?think>', '', text)
    t = text.strip()
    for pat in PREAMBLE_PATTERNS:
        new_t = re.sub(pat, '', t, flags=re.IGNORECASE)
        if new_t != t:
            t = new_t.strip()
            break
    t = t.rstrip('.').strip()
    return t

def build_rag_user_prompt(index, records, question: str, corpus_name: str, top_k: int=5):
    retrieved = retrieve(index, records, question, k=top_k)
    context = '\n\n'.join((f'[{i + 1}] {extract_passage_text(corpus_name, doc)}' for i, (doc, score) in enumerate(retrieved)))
    return (f'Context:\n{context}\n\nQuestion: {question}', retrieved)

def run_query(model, tokenizer, model_key: str, corpus_name: str, question: str, index=None, records=None, top_k: int=5, max_new_tokens: int=256):
    if index is None or records is None:
        index, records = build_index(corpus_name, split='dev')
    user_prompt, retrieved = build_rag_user_prompt(index, records, question, corpus_name, top_k=top_k)
    prompt = build_chat_prompt(model_key, tokenizer, SYSTEM_PROMPT, user_prompt)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    generated = generated.strip()
    return {'question': question, 'retrieved_doc_ids': [idx for idx, _ in enumerate(retrieved)], 'retrieval_scores': [score for _, score in retrieved], 'generated_answer': generated, 'generated_answer_clean': clean_generation(generated)}
