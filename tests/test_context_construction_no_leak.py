"""
build_rag_user_prompt() rendered every retrieved document as
`doc.get('text', doc)`. None of the three corpora's raw records carry a
'text' key, so this always fell through to `doc` itself -- the model's
"Context:" block was the raw Python dict repr of the dataset row, gold
answer included (confirmed against real cached nq_open records; see
harness/pipeline.py's build_rag_user_prompt docstring and
nq_open_leakage_finding.md).

These tests pin the fixed behavior: hotpot_qa/ms_marco now render real
passage text via data.normalize.extract_passage_text, with no raw dict
repr and no answer key leaked into context. nq_open is asserted to still
leak -- not a regression, but data/normalize.py's nq_open branch
concatenating the gold answer into the "passage" text by construction
(nq_open has no independent supporting passage at all). That's a known,
flagged, NOT-fixed-here limitation -- this test exists so it can't
silently regress into "fixed" without anyone noticing the assumption
changed.
"""

import harness.pipeline as pipeline


def _render(monkeypatch, corpus_name, record, question="irrelevant question"):
    monkeypatch.setattr(
        pipeline, "retrieve", lambda index, records, q, k=5: [(record, 0.9)]
    )
    user_prompt, _ = pipeline.build_rag_user_prompt(
        index=None, records=None, question=question, corpus_name=corpus_name
    )
    return user_prompt


def test_hotpot_qa_context_has_no_raw_dict_repr_or_answer_leak(monkeypatch):
    record = {
        "question": "what season is pepper in american horror story",
        "answer": "the second and fourth seasons",
        "context": {
            "title": ["American Horror Story"],
            "sentences": [["Pepper appears in seasons two and four."]],
        },
    }
    prompt = _render(monkeypatch, "hotpot_qa", record)

    assert "{'question'" not in prompt  # no raw dict repr
    assert "'answer'" not in prompt  # answer key itself never appears
    assert "the second and fourth seasons" not in prompt  # gold answer not leaked
    assert "Pepper appears in seasons two and four" in prompt  # real passage text present


def test_ms_marco_context_has_no_raw_dict_repr_or_answer_leak(monkeypatch):
    record = {
        "query": "what is the incarceration rate in the united states",
        "answers": ["0.71% of the population"],
        "passages": {"passage_text": ["The US incarceration rate is a widely studied statistic."]},
    }
    prompt = _render(monkeypatch, "ms_marco", record)

    assert "{'query'" not in prompt
    assert "'answers'" not in prompt
    assert "0.71% of the population" not in prompt
    assert "widely studied statistic" in prompt


def test_nq_open_still_leaks_the_gold_answer_known_unfixed_limitation(monkeypatch):
    # nq_open has no independent supporting passage in the raw dataset --
    # extract_passage_text's nq_open branch concatenates the gold answer
    # into the "passage" text because there's nothing else to embed/show.
    # This is intentional-and-flagged, not something this fix resolves.
    record = {
        "question": "where's the chick-fil-a kickoff game being played",
        "answer": ["Mercedes-Benz Stadium"],
    }
    prompt = _render(monkeypatch, "nq_open", record)

    assert "Mercedes-Benz Stadium" in prompt  # gold answer IS still present in context
    assert "{'question'" not in prompt  # but at least no raw dict-repr formatting anymore
