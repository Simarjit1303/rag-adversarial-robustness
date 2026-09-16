import harness.pipeline as pipeline

def _render(monkeypatch, corpus_name, record, question='irrelevant question'):
    monkeypatch.setattr(pipeline, 'retrieve', lambda index, records, q, k=5: [(record, 0.9)])
    user_prompt, _ = pipeline.build_rag_user_prompt(index=None, records=None, question=question, corpus_name=corpus_name)
    return user_prompt

def test_hotpot_qa_context_has_no_raw_dict_repr_or_answer_leak(monkeypatch):
    record = {'question': 'what season is pepper in american horror story', 'answer': 'the second and fourth seasons', 'context': {'title': ['American Horror Story'], 'sentences': [['Pepper appears in seasons two and four.']]}}
    prompt = _render(monkeypatch, 'hotpot_qa', record)
    assert "{'question'" not in prompt
    assert "'answer'" not in prompt
    assert 'the second and fourth seasons' not in prompt
    assert 'Pepper appears in seasons two and four' in prompt

def test_ms_marco_context_has_no_raw_dict_repr_or_answer_leak(monkeypatch):
    record = {'query': 'what is the incarceration rate in the united states', 'answers': ['0.71% of the population'], 'passages': {'passage_text': ['The US incarceration rate is a widely studied statistic.']}}
    prompt = _render(monkeypatch, 'ms_marco', record)
    assert "{'query'" not in prompt
    assert "'answers'" not in prompt
    assert '0.71% of the population' not in prompt
    assert 'widely studied statistic' in prompt

def test_nq_open_still_leaks_the_gold_answer_known_unfixed_limitation(monkeypatch):
    record = {'question': "where's the chick-fil-a kickoff game being played", 'answer': ['Mercedes-Benz Stadium']}
    prompt = _render(monkeypatch, 'nq_open', record)
    assert 'Mercedes-Benz Stadium' in prompt
    assert "{'question'" not in prompt
