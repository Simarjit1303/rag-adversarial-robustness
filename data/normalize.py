def extract_passage_text(corpus_name: str, record: dict) -> str:
    if corpus_name == 'nq_open':
        return record.get('question', '') + ' ' + ' '.join(record.get('answer', []))
    if corpus_name == 'hotpot_qa':
        context = record.get('context', {})
        titles = context.get('title', [])
        sentences = context.get('sentences', [])
        flat = []
        for t, s in zip(titles, sentences):
            flat.append(t + ': ' + ' '.join(s))
        return ' '.join(flat) if flat else record.get('question', '')
    if corpus_name == 'ms_marco':
        passages = record.get('passages', {})
        texts = passages.get('passage_text', [])
        return ' '.join(texts) if texts else record.get('query', '')
    raise ValueError(f"No normalization rule for corpus '{corpus_name}'")

def extract_question(corpus_name: str, record: dict) -> str:
    if corpus_name == 'nq_open':
        return record.get('question', '')
    if corpus_name == 'hotpot_qa':
        return record.get('question', '')
    if corpus_name == 'ms_marco':
        return record.get('query', '')
    raise ValueError(f"No normalization rule for corpus '{corpus_name}'")

def extract_gold_answers(corpus_name: str, record: dict) -> list[str]:
    if corpus_name == 'nq_open':
        return record.get('answer', [])
    if corpus_name == 'hotpot_qa':
        ans = record.get('answer')
        return [ans] if ans else []
    if corpus_name == 'ms_marco':
        answers = record.get('answers', [])
        return answers if isinstance(answers, list) else [answers]
    raise ValueError(f"No normalization rule for corpus '{corpus_name}'")
