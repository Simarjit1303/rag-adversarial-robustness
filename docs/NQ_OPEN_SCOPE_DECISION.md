# Scope Adjustment: Corpus Selection

**Status: DRAFT — pending supervisor acknowledgment.** This is not a
unilateral decision being reported after the fact. It is written to be
discussed and signed off at the upcoming supervisor meeting; until then,
the project proceeds on the reduced 2-corpus design provisionally, exactly
as `nq_open_leakage_finding.md` itself frames it ("I'm treating every
existing `nq_open` result file as provisional pending your decision on how
to handle it").

Source: this note is drawn entirely from `nq_open_leakage_finding.md` (full
file, read in its entirety for this note — nothing below is reconstructed
from memory or general familiarity with the project) and `config.py`'s
`CORPORA` dict, which records the resulting scope decision directly in
code comments.

---

## 1. What `nq_open` was intended to contribute

The project's original design was **4 models × 3 corpora**: `hotpot_qa`,
`ms_marco`, and `nq_open`, evaluated identically. All three corpora were
registered in `config.py`'s `CORPORA` dict and all three were swept in
Phase 1's baseline (`phase1_results_complete/` contains all 12 model×corpus
cells, including `nq_open`). `nq_open` was meant to add a third,
distinct retrieval difficulty/style profile to the corpus dimension — used,
like the other two, wherever an attack or defense sweep varies by corpus
(the baseline, and the two corpus-dependent attacks — see Section 4 below
for why Crescendo is the exception).

## 2. The exact technical finding

Quoting `nq_open_leakage_finding.md` directly, not paraphrasing:

> "While preparing the Phase 2 indirect prompt injection attack, I found
> that the RAG pipeline's 'retrieved context' — the block of text every
> model actually reads before answering — was not real passage text. For
> all three corpora, it was the raw Python dict representation of the
> underlying dataset row, gold answer included."

The root cause was `harness/pipeline.py`'s `build_rag_user_prompt()`
rendering each document's context line as:

```python
f"[{i + 1}] {doc.get('text', doc)}"
```

None of the three corpora's raw HuggingFace records have a `'text'` key
(confirmed against `data/normalize.py`'s own field mappings — `nq_open` →
`question`/`answer`; `hotpot_qa` → `question`/`context`/`answer`;
`ms_marco` → `query`/`passages`/`answers`), so this line always fell
through to `doc` itself. A real cached `nq_open` record was shown to the
model verbatim as:

```
OLD: [1] {'question': "where's the chick-fil-a kickoff game being played", 'answer': ['Mercedes-Benz Stadium']}
```

`hotpot_qa` and `ms_marco` were fixed by routing context rendering through
`data/normalize.py`'s existing `extract_passage_text()`, which returns real
supporting passage text for both with no answer key mixed in (verified by
a regression test, `tests/test_context_construction_no_leak.py`).

**`nq_open` could not be fixed the same way**, for a reason specific to how
the corpus is constructed, not a further code bug. `extract_passage_text()`'s
`nq_open` branch reads:

```python
if corpus_name == "nq_open":
    return record.get("question", "") + " " + " ".join(record.get("answer", []))
```

Per the finding document: "This concatenates the gold answer into the
'passage' text by construction — there is nothing else to show. `nq_open`
(Google's open-domain NQ variant) ships with no supporting passage at all;
this codebase substituted question+answer text as a stand-in 'document' so
a FAISS index could be built over something. That's a corpus-construction
choice made before this project, not a bug in how context gets rendered —
no change to `pipeline.py` can fix it, because there is no non-leaking text
to substitute in." After the fix, the rendered context still contains the
gold answer, just as a plain sentence instead of a dict repr:

```
NEW: [1] where's the chick-fil-a kickoff game being played Mercedes-Benz Stadium
```

## 3. Why this makes `nq_open`'s results unsuitable for inclusion

The leakage is not a matter of degree or a fixable formatting artifact for
`nq_open` — the gold answer is present in what the model reads on every
single question, by construction of the corpus itself. The finding
document's own supporting evidence table (recomputed directly from stored
raw JSONL, all 12 real Phase 1 cells, no re-generation needed):

| Corpus | Mean F1(clean) across 4 models | Spread across models |
|---|---|---|
| `nq_open` | 0.962 | 0.005 (0.960–0.965) |
| `hotpot_qa` | 0.585 | 0.179 (0.460–0.639) |
| `ms_marco` | 0.255 | 0.042 (0.234–0.276) |

*Values recomputed on 2026-09-19 from the committed Phase 1 summary CSVs (`THESIS_MASTER_RECORD.md` Section 9.1). The `nq_open` row is 0.962 / 0.960–0.965; `nq_open_leakage_finding.md`, preserved unmodified, still shows 0.963 / 0.960–0.966. The other two rows and every spread are identical in both.*

As the finding states: "`nq_open` is both near-ceiling and nearly invariant
across four models of meaningfully different capability — the pattern
you'd expect from answer-copying, not from a task where retrieval and
reasoning quality matter." Any attack or defense comparison run against
`nq_open` would be measuring resistance to manipulating a prompt that
already contains the correct answer, not retrieval-augmented question
answering — making ASR, utility-drop, and defense-effectiveness numbers
computed on it **not comparable** to the equivalent `hotpot_qa`/`ms_marco`
numbers, and not meaningful as a stand-alone RAG robustness result either.
This is why the finding document states plainly it is "not running Phase
2's injection attack against `nq_open` in its current form," and why
`config.py`'s `CORPORA` entry for `nq_open` carries the comment: "Real
matrix going forward: 4 models x 2 corpora (hotpot_qa, ms_marco) x 3
attacks. `nq_open`'s Phase 1 result is kept only as a documented
limitation." The `nq_open` entry itself was left registered and pinned in
`config.py` — not deleted — specifically so the already-collected Phase 1
baseline stays reproducible as a documented limitation, not so it could be
used in further comparisons.

**The decision record, found outside this git repo:** the finding document
proposes three options for how to handle `nq_open` going forward — (a)
source real supporting passages for it from the original Natural Questions
release, (b) drop it from the corpus set entirely, or (c) keep it but
explicitly reframe what it measures (closed-book knowledge recovery, not
RAG) with every conclusion caveated accordingly. `nq_open_leakage_finding.md`
itself does not record which option was chosen or by whom. That record
exists in `Drafts/PHASE2_ROADMAP.md` (parent project folder, outside this
git repo), under a section header dated identically to `config.py`'s own
comment. Quoting it directly, not paraphrasing:

> "## nq_open Exclusion (decided 2026-09-04)
>
> A context-construction bug caused gold-answer leakage into the RAG
> context for all three corpora (see `nq_open_leakage_finding.md`,
> committed on branch `fix-rag-context-answer-leak`). The bug is fixed
> there for `hotpot_qa` and `ms_marco`. It **cannot** be fixed the same way
> for `nq_open`: `nq_open` has no independent supporting passage at all, so
> its 'passage' text is the gold answer by construction. There is no
> non-leaking passage to fall back to.
>
> **Decision: `nq_open` is excluded from all real Phase 2/3 sweeps going
> forward.** Its Phase 1 result is retained only as a documented
> limitation, not as a valid baseline to build on. The real matrix going
> forward is **4 models × 2 corpora (hotpot_qa, ms_marco) × 3 attacks**,
> not 4×3×3. Every 'three corpora' reference elsewhere in this document
> predates this decision and should be read with that correction in mind."

*Provenance note:* `PHASE2_ROADMAP.md` does not exist in this git
repository — confirmed via repo-wide search. It exists only in the parent
project folder (`Drafts/PHASE2_ROADMAP.md`, outside this repo), so the
quote above is the only place its content is preserved in anything
committed here. Several task briefs (e.g. `phase2_poisonedrag_task.md:25`)
also cite it by name as the source for poison-count and dataset-size
decisions "confirmed in Meeting 3."

This confirms option (b) is what was chosen, on 2026-09-04 — the same date
`config.py`'s `CORPORA` comment records, consistent with a single decision
made once and reflected in both places. `PHASE2_ROADMAP.md` states the
decision itself but, like `config.py`'s comment, does not separately name
who made it or record a supervisor sign-off — that remains the open item
Section 6 below asks to close at the upcoming meeting.

## 4. Crescendo is corpus-independent — the scope impact is narrower than "one of three corpora lost"

`nq_open`'s exclusion has **zero effect on the Crescendo attack family**.
Crescendo is a direct multi-turn conversational attack on the target
model with no retrieval step and no corpus axis at all — per
`phase2_crescendo_task.md`'s own scope section: "No corpus dimension here
(Crescendo isn't a RAG-corpus attack — it's a direct conversational attack
on the target model, RAG context is not the injection surface)." Crescendo
was never run against any corpus, leaked or otherwise, and its 4-model
sweep and results are entirely unaffected by this scope adjustment.

This means the practical scope reduction lands on **2 of 3 attack
families** — indirect prompt injection and PoisonedRAG, both of which
depend on retrieved context — plus the Phase 1 baseline itself. Crescendo's
full 4-model, 100-behavior-per-model design and results stand exactly as
originally planned.

## 5. Why a substitute third corpus was not pursued

A natural alternative would be to swap in a different third corpus rather
than drop the corpus dimension to 2. This was considered and set aside for
two honest reasons, not silently dropped:

- **Timing.** The leakage was found while preparing Phase 2 (per the
  finding document's own opening line), with two further attacks and all
  three defenses still ahead. Sourcing, integrating, and validating a
  wholly new corpus at that point — new field-mapping logic in
  `data/normalize.py`, a new FAISS index, a full Phase 1 baseline re-run,
  then re-running every subsequent attack/defense sweep against it — was
  not a cost the remaining project timeline could absorb without delaying
  every phase behind it.
- **A new corpus carries the same category of undiscovered-until-executed
  risk that this exact finding is an instance of.** This project's own
  documented pattern (`THESIS_MASTER_RECORD.md` Section 7's eight
  real-hardware bugs, and Section 8's backend confound) is that this class
  of data/infrastructure issue is not reliably caught by code review or
  small-scale testing — it surfaces only once the real pipeline is run
  against real data at real scale. `nq_open`'s leak itself fits that exact
  pattern: it was invisible until a real attack was being prepared against
  real retrieved context, not caught by design review beforehand. A freshly
  substituted corpus would carry the same risk of hiding its own
  construction quirk, discoverable only after the same sequence of
  integration work had already been spent — with no guarantee of a better
  outcome and a confirmed cost in time this late in the project.

Both reasons are named explicitly here so this is legible as a considered
trade-off, not an oversight.

## 6. This is a scope adjustment for discussion, not a closed decision

To restate plainly: this document formalizes, for the Methodology/
Limitations chapter, a reduction already reflected in the code
(`config.py`'s `CORPORA` comment, dated 2026-09-04) and already applied
throughout Phase 2 and Phase 3's real sweeps. It is written up now so it
can be acknowledged explicitly at the next supervisor meeting — matching
`nq_open_leakage_finding.md`'s own original framing that every `nq_open`
result is "provisional pending your decision on how to handle it." Nothing
here should be read as a final, unilaterally-settled position ahead of that
conversation.

**For discussion:** confirm the 4-models × 2-corpora × 3-attacks scope
(`hotpot_qa`, `ms_marco`; `nq_open` retained only as a documented Phase 1
limitation) as the final thesis scope, or direct a different path from the
three options `nq_open_leakage_finding.md` originally proposed.

---

*Companion to [`nq_open_leakage_finding.md`](nq_open_leakage_finding.md)
(the original technical finding, preserved unmodified) and
[`THESIS_MASTER_RECORD.md`](THESIS_MASTER_RECORD.md) Section 4 (the
consolidated project-wide summary of this exclusion).*
