# Finding: gold-answer leakage in the RAG context, and why nq_open can't be fixed the same way as the other two corpora

## Summary

While preparing the Phase 2 indirect prompt injection attack, I found that
the RAG pipeline's "retrieved context" — the block of text every model
actually reads before answering — was not real passage text. For all three
corpora, it was the raw Python dict representation of the underlying
dataset row, gold answer included. For `nq_open` specifically, this isn't
fixable at the pipeline level: the corpus itself has no independent
supporting passage to show the model, so any usable "document" text for it
necessarily contains the answer.

Practical effect: `nq_open` Phase 1 baseline numbers (F1 ≈ 0.96, near-ceiling
and nearly identical across all four models) most likely reflect the model
copying an answer that was handed to it, not retrieval-augmented QA. I'm
treating every existing `nq_open` result file as provisional pending your
decision on how to handle it, and I'm not running Phase 2's injection
attack against `nq_open` in its current form.

## The bug

`harness/pipeline.py`'s `build_rag_user_prompt()` built each document's
context line as:

```python
f"[{i + 1}] {doc.get('text', doc)}"
```

None of the three corpora's raw HuggingFace records have a `'text'` key
(`nq_open` → `question`/`answer`; `hotpot_qa` → `question`/`context`/`answer`;
`ms_marco` → `query`/`passages`/`answers` — confirmed against
`data/normalize.py`'s own field mappings). So this line always fell through
to `doc` itself, and the model was shown the entire record's dict repr.

Concretely, for a real cached `nq_open` record:

```
OLD: [1] {'question': "where's the chick-fil-a kickoff game being played", 'answer': ['Mercedes-Benz Stadium']}
```

This is also the direct explanation for a previously-landed fix
(`fix_leaked_answer_pattern.md` / `_try_extract_leaked_answer()` in
`clean_generation()`): that fix noticed the model sometimes echoes this
exact dict-repr shape back in its answer, and added logic to extract the
answer already sitting inside it. It treated the echo as an output-cleanup
quirk (~1-2% of generations) rather than root-causing that the *input*
context contained the gold answer on 100% of questions — the other ~98-99%
of the time, the model was simply copying the answer without echoing the
surrounding dict formatting, which is invisible to that fix and to every
metric downstream of it.

## What I fixed, on `fix-rag-context-answer-leak`

`data/normalize.py` already has `extract_passage_text()`, a per-corpus
function that returns real passage/context text — it was already used for
FAISS embedding text in `data/build_index.py`, just never reused for
context rendering in `pipeline.py`. I swapped `build_rag_user_prompt()` to
use it.

For `hotpot_qa` and `ms_marco`, this is a genuine fix: their
`extract_passage_text()` branches return real supporting text (HotpotQA's
distractor context sentences; MS MARCO's passage text) with no answer key
mixed in. Verified with an explicit regression test
(`tests/test_context_construction_no_leak.py`) asserting no dict-repr
formatting and no gold-answer substring in the rendered context for either
corpus.

## What I did NOT fix: `nq_open`

`data/normalize.py`'s `extract_passage_text()` for `nq_open`:

```python
if corpus_name == "nq_open":
    return record.get("question", "") + " " + " ".join(record.get("answer", []))
```

This concatenates the gold answer into the "passage" text **by
construction** — there is nothing else to show. `nq_open` (Google's
open-domain NQ variant) ships with no supporting passage at all; this
codebase substituted question+answer text as a stand-in "document" so a
FAISS index could be built over something. That's a corpus-construction
choice made before this project, not a bug in how context gets rendered —
no change to `pipeline.py` can fix it, because there is no non-leaking text
to substitute in.

After the fix, `nq_open`'s context still contains the gold answer, just as
a plain sentence instead of a dict repr:

```
NEW: [1] where's the chick-fil-a kickoff game being played Mercedes-Benz Stadium
```

## Supporting evidence: the nq_open-vs-hotpot_qa/ms_marco F1 gap

Real Phase 1 results on disk (`phase1_results_complete/`, all 12
model×corpus cells, `INFERENCE_ENGINE=vllm`). **`hotpot_qa`/`ms_marco`
columns updated 2026-09-06** after the `pipeline.py` fix below was actually
applied and the baseline rerun (`nq_open` is untouched by that fix — still
the original, unfixable-by-construction leaked numbers, as it always will
be). **All three columns updated again 2026-09-08** after a second, smaller
fix: `evaluation/metrics.py`'s `normalize_text()` was deleting punctuation
outright instead of replacing it with a space, so answer pairs like
`"28-32"`/`"28 - 32"` normalized to different strings and silently
undercounted EM/F1 by a formatting artifact, not a real answer difference
(see `PHASE2_INJECTION_INSIGHTS.md`'s equivalent banner for the full
writeup). Recomputed from already-stored raw JSONL, no re-generation:

| Corpus | Mean F1(clean) across 4 models | Spread across models |
|---|---|---|
| `nq_open` | 0.963 | 0.005 (0.960–0.966) |
| `hotpot_qa` | 0.585 | 0.179 (0.460–0.639) |
| `ms_marco` | 0.255 | 0.042 (0.234–0.276) |

(Previous, pre-normalize-fix values: `nq_open` 0.962/0.005, `hotpot_qa`
0.583/0.178, `ms_marco` 0.246/0.046 — all three moved by ≤0.01, `ms_marco`
moved the most since its gold answers contain the most numeric-range/
punctuation formatting. None of this section's conclusions change: `nq_open`
still sits far closer to ceiling than either real corpus, before or after
either fix.)

`nq_open` is both near-ceiling and nearly invariant across four models of
meaningfully different capability — the pattern you'd expect from answer-
copying, not from a task where retrieval and reasoning quality matter. That
core conclusion is untouched by the fix (nq_open's leak isn't caused by the
bug the fix addresses; it's caused by the corpus having no real passage to
substitute). What the fix *did* change: pre-fix, this table's `ms_marco` row
showed a large spread (0.29, since retracted — see below) that this
document originally read as "tracking the models' known relative strength."
With the leak removed, `ms_marco`'s spread across models actually shrank to
0.046 while `hotpot_qa`'s grew to 0.178 — the opposite of that original
sub-claim, which is retracted here. The core point stands regardless:
`nq_open` sits far closer to its ceiling (0.96, spread 0.005) than either
real corpus does, before or after the fix, and it's telling you `nq_open`,
as currently constructed in this project, is not measuring
retrieval-augmented QA.

## What this means for existing results, and what I need from you

- Every `nq_open` cell already collected in `phase1_results_complete/` and
  `phase1_results_partial/` should be treated as **provisional, not
  settled** — I have not deleted or modified those files.
- I have **not** re-run any part of the Phase 1 sweep. This fix only
  changes code; it does not retroactively correct already-written result
  files.
- I am **not** running the Phase 2 indirect-injection attack against
  `nq_open` in its current form — Phase 2 for now proceeds on `hotpot_qa`
  and `ms_marco` only.
- Options for `nq_open` going forward, for you to decide: (a) source real
  supporting passages for it (the original Natural Questions release does
  have long/short answer spans over real Wikipedia pages — worth checking
  whether that's obtainable for this HF variant), (b) drop `nq_open` from
  the corpus set entirely, or (c) keep it, but explicitly reframe what it's
  measuring (closed-book knowledge recovery via a retrieval-shaped
  scaffold, not RAG) and caveat every conclusion drawn from it accordingly.

## Where the code lives

Branch `fix-rag-context-answer-leak` (based on `phase2-injection-templates`'s
tip, not `main` — `main` predates the `clean_generation`/
`build_rag_user_prompt` refactor entirely and does not reflect current
pipeline behavior). Touches `harness/pipeline.py`,
`evaluation/run_baseline.py`, `scripts/compute_max_model_len.py` (all three
callers of `build_rag_user_prompt`, updated to pass `corpus_name` through),
plus the new regression test. All 67 pre-existing tests still pass.
