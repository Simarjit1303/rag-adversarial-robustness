import json
from attacks.injection_templates import TEMPLATES
from defenses.instruction_detection import detect_injection, filter_retrieved_passages
CLEAN_PASSAGES = ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel's company.", 'Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose.', 'The Amazon River in South America discharges more water than the next several largest rivers combined.']
INJECTED_PASSAGES = [TEMPLATES['ignore'].render('Mount Everest is the tallest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.'), TEMPLATES['fake_completion'].render('The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.')]
_INJECTION_MARKERS = ('Ignore my previous instructions', 'Answer: task complete')

def _fake_classifier(text, **kw):
    flagged = any((marker in text for marker in _INJECTION_MARKERS))
    return [{'label': 'INJECTION' if flagged else 'SAFE', 'score': 0.97 if flagged else 0.02}]

def test_clean_passages_not_flagged():
    for text in CLEAN_PASSAGES:
        result = detect_injection(text, classifier=_fake_classifier)
        assert not result.flagged, f'false positive on clean passage: {text!r} ({result})'

def test_ignore_template_flagged():
    result = detect_injection(INJECTED_PASSAGES[0], classifier=_fake_classifier)
    assert result.flagged, f"missed 'ignore' injection: {result}"

def test_fake_completion_template_flagged():
    result = detect_injection(INJECTED_PASSAGES[1], classifier=_fake_classifier)
    assert result.flagged, f"missed 'fake_completion' injection: {result}"

def test_detect_injection_respects_threshold():
    low_confidence = lambda text, **kw: [{'label': 'INJECTION', 'score': 0.3}]
    result = detect_injection('anything', classifier=low_confidence, threshold=0.5)
    assert not result.flagged
    assert result.score == 0.3
    assert result.label == 'INJECTION'

def test_detect_injection_label_case_insensitive():
    mixed_case = lambda text, **kw: [{'label': 'injection', 'score': 0.9}]
    result = detect_injection('anything', classifier=mixed_case)
    assert result.flagged

def test_detect_injection_passes_explicit_max_length():
    captured_kwargs = {}

    def _recording_classifier(text, **kw):
        captured_kwargs.update(kw)
        return [{'label': 'SAFE', 'score': 0.02}]
    detect_injection('anything', classifier=_recording_classifier)
    assert captured_kwargs.get('truncation') is True
    assert captured_kwargs.get('max_length') == 512

def test_load_default_classifier_caps_torch_threads(monkeypatch):
    import defenses.instruction_detection as idmod
    monkeypatch.setattr(idmod, '_pipeline', None)
    captured = {}

    class _FakePipeline:

        def __call__(self, *a, **kw):
            return [{'label': 'SAFE', 'score': 0.01}]

    def _fake_set_num_threads(n):
        captured['n'] = n

    def _fake_pipeline_factory(*a, **kw):
        captured['threads_capped_before_pipeline_load'] = 'n' in captured
        return _FakePipeline()
    monkeypatch.setattr('torch.set_num_threads', _fake_set_num_threads)
    monkeypatch.setattr('transformers.pipelines.pipeline', _fake_pipeline_factory)
    idmod._load_default_classifier()
    assert captured.get('n') == 1
    assert captured.get('threads_capped_before_pipeline_load') is True

def test_filter_retrieved_passages_drops_flagged_and_keeps_clean(monkeypatch):
    retrieved = [({'text': CLEAN_PASSAGES[0]}, 0.8), ({'text': INJECTED_PASSAGES[0]}, 0.7), ({'text': CLEAN_PASSAGES[1]}, 0.6)]
    import defenses.instruction_detection as idmod
    monkeypatch.setattr(idmod, 'extract_passage_text', lambda corpus_name, doc: doc['text'])
    context, log = filter_retrieved_passages(retrieved, 'fake_corpus', classifier=_fake_classifier)
    assert CLEAN_PASSAGES[0] in context
    assert CLEAN_PASSAGES[1] in context
    assert INJECTED_PASSAGES[0] not in context
    assert len(log) == 3
    assert [entry.passage_id for entry in log] == [0, 1, 2]
    assert [entry.flagged for entry in log] == [False, True, False]

def test_log_passage_detection_event_appends_and_is_durable_across_calls(tmp_path):
    from defenses.instruction_detection import log_passage_detection_event
    log_path = tmp_path / 'instruction_detection_log.jsonl'
    log_passage_detection_event(log_path, {'model': 'qwen3-8b', 'corpus': 'hotpot_qa', 'injection_template': 'naive', 'question': 'q1?', 'passages': [{'passage_id': 0, 'flagged': True, 'score': 0.99, 'label': 'INJECTION'}, {'passage_id': 1, 'flagged': False, 'score': 0.02, 'label': 'SAFE'}]})
    log_passage_detection_event(log_path, {'model': 'qwen3-8b', 'corpus': 'hotpot_qa', 'injection_template': 'naive', 'question': 'q2?', 'passages': [{'passage_id': 0, 'flagged': False, 'score': 0.01, 'label': 'SAFE'}]})
    lines = log_path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2
    row1 = json.loads(lines[0])
    row2 = json.loads(lines[1])
    assert row1['question'] == 'q1?'
    assert row1['passages'][0] == {'passage_id': 0, 'flagged': True, 'score': 0.99, 'label': 'INJECTION'}
    assert row1['passages'][1]['flagged'] is False
    assert row2['question'] == 'q2?'
    assert len(row2['passages']) == 1
