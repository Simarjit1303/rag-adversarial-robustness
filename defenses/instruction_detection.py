import json
import os
from dataclasses import dataclass
from typing import Callable, Optional
from data.normalize import extract_passage_text
_MODEL_NAME = 'protectai/deberta-v3-base-prompt-injection-v2'
_FLAG_LABEL = 'INJECTION'
_DEFAULT_THRESHOLD = 0.5
_pipeline = None

def _load_default_classifier():
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import pipeline
        torch.set_num_threads(1)
        _pipeline = pipeline('text-classification', model=_MODEL_NAME, tokenizer=_MODEL_NAME, device=-1)
    return _pipeline

@dataclass
class DetectionResult:
    flagged: bool
    score: float
    label: str

def detect_injection(text: str, classifier: Optional[Callable[..., list]]=None, threshold: float=_DEFAULT_THRESHOLD) -> DetectionResult:
    clf = classifier or _load_default_classifier()
    result = clf(text, truncation=True, max_length=512)[0]
    label = result['label']
    score = float(result['score'])
    flagged = label.upper() == _FLAG_LABEL and score >= threshold
    return DetectionResult(flagged=flagged, score=score, label=label)

@dataclass
class PassageLog:
    passage_id: int
    flagged: bool
    score: float
    label: str

def filter_retrieved_passages(retrieved, corpus_name: str, classifier=None):
    lines = []
    log = []
    for i, (doc, score) in enumerate(retrieved):
        text = extract_passage_text(corpus_name, doc)
        result = detect_injection(text, classifier=classifier)
        log.append(PassageLog(passage_id=i, flagged=result.flagged, score=result.score, label=result.label))
        if result.flagged:
            continue
        lines.append(f'[{i + 1}] {text}')
    context = '\n\n'.join(lines)
    return (context, log)

def log_passage_detection_event(log_path, record: dict):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
