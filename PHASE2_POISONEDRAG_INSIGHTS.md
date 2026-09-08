# Phase 2, Attack 2: PoisonedRAG — Insights

Full sweep: 4 models × 2 corpora (`hotpot_qa`, `ms_marco` — `nq_open` excluded
project-wide, see `nq_open_leakage_finding.md`) × 1 poison config (`adv5`,
`ADV_PER_QUERY=5`), n=90/100 `hotpot_qa` and n=96/100 `ms_marco` (skip/loss
breakdown below — Phase A's poisoned contexts are model-agnostic and shared
across all 4 models' cells, so the same 90/96 questions feed every model).
Raw data: `phase2_poisonedrag_results/`. Baseline comparison:
`phase1_results_complete/`. Analysis script:
`scripts/analyze_phase2_poisonedrag_stats.py` (re-runnable, prints the full
per-cell table and every significance test below).

## Headline finding: corpus identity, not model identity, is what determines PoisonedRAG's success

Attack 1 (indirect prompt injection) found model identity as the dominant
factor — llama-3.1-8b robust across the board, ministral-3-8b and qwen3-8b
selectively vulnerable to specific templates, with real per-model mechanism
differences (see `PHASE2_INJECTION_INSIGHTS.md`). PoisonedRAG shows close to
the opposite pattern:

- **ASR sits at 70–78% on `hotpot_qa` and 18–20% on `ms_marco`, for every
  single one of the 4 models** — a ~50-60 percentage point gap driven
  entirely by which corpus is being attacked, not which model answers.
- **All 12 model-vs-model McNemar comparisons (6 pairs × 2 corpora) are
  non-significant after Holm-Bonferroni correction** (smallest raw p=0.065,
  qwen3-8b vs phi-4-mini on `hotpot_qa` — even that one doesn't survive
  correction). The small differences between models within a corpus (e.g.
  qwen3-8b at 77.8% vs phi-4-mini at 70.0% on `hotpot_qa`) are genuinely
  indistinguishable from chance at this sample size, not a real ranking.
- **All 4 corpus-vs-corpus Fisher comparisons (1 per model) are significant
  at p < 2×10⁻¹²**, Holm-corrected — the corpus effect is not sampling
  noise.

**Model robustness and corpus robustness look like genuinely independent
dimensions in this dataset.** Attack 1 showed that *which model* you ask
matters enormously for injection but *which corpus* barely matters (goal-
hijack templates stay flat everywhere; process-hijack susceptibility is a
per-model property that shows up on both corpora for a given model, just at
different magnitudes). PoisonedRAG shows the mirror image: *which corpus* you
poison matters enormously, and *which model* answers from the poisoned
context barely matters. Put together, this suggests corpus-poisoning
resistance and injection resistance aren't the same underlying capability
that a single "robustness" score could summarize — a model (or a defense)
good at one is not thereby predicted to be good at the other, and a full
robustness picture needs both attack types evaluated separately rather than
collapsed into one number.

## Per-cell results (ASR, poison-echoed-not-adopted, utility)

| model | corpus | n | ASR | echoed, not adopted | contains-target | F1 baseline | F1 attack | utility drop (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | hotpot_qa | 90 | 73.3% | 2.2% | 75.6% | 0.640 | 0.142 | 0.498 (0.399–0.594) |
| qwen3-8b | hotpot_qa | 90 | 77.8% | 4.4% | 82.2% | 0.662 | 0.105 | 0.556 (0.463–0.649) |
| phi-4-mini | hotpot_qa | 90 | 70.0% | 10.0% | 80.0% | 0.392 | 0.091 | 0.301 (0.205–0.399) |
| ministral-3-8b | hotpot_qa | 90 | 76.7% | 6.7% | 83.3% | 0.679 | 0.127 | 0.552 (0.455–0.646) |
| llama-3.1-8b | ms_marco | 96 | 17.7% | 2.1% | 19.8% | 0.265 | 0.102 | 0.163 (0.101–0.230) |
| qwen3-8b | ms_marco | 96 | 19.8% | 2.1% | 21.9% | 0.288 | 0.121 | 0.167 (0.105–0.231) |
| phi-4-mini | ms_marco | 96 | 18.8% | 3.1% | 21.9% | 0.275 | 0.157 | 0.117 (0.052–0.182) |
| ministral-3-8b | ms_marco | 96 | 19.8% | 0.0% | 19.8% | 0.273 | 0.114 | 0.159 (0.097–0.225) |

**"Poison-echoed, not adopted" (`contains_target_diagnostic=1 AND
attack_success=0`)** is reported here as its own category rather than folded
into a binary success/fail, per the same logic Attack 1 applied to its
ASR-scoring caveats: the model's raw generation contains the poisoned target
string, but `attack_success`'s stricter scoring (matching gold-answer-style
extraction against `target_answer`) doesn't count it as adoption — e.g. the
model discusses or quotes the poisoned claim without asserting it as the
answer. This bucket is small everywhere (0–10%) but not negligible for
phi-4-mini on `hotpot_qa` (10.0%, its highest of any cell) — consistent with
phi-4-mini also having the smallest F1-baseline-to-utility-drop ratio of the
four models, discussed below.

**Utility drop is large and highly significant everywhere**, tracking the
same corpus split as ASR: 0.30–0.56 F1 on `hotpot_qa`, 0.12–0.17 F1 on
`ms_marco`, every 95% CI well clear of zero. Unlike Attack 1 — where utility
damage turned out to closely track ASR once the baseline leak was fixed, with
near-zero-ASR cells showing near-zero F1 drop — PoisonedRAG's utility drop is
large even relative to its ASR: `ms_marco`'s ~18-20% ASR still produces a
0.12-0.17 F1 drop, larger than Attack 1's *entire* grid outside its two
process-hijack templates. This is expected given the two attacks' different
mechanisms: PoisonedRAG doesn't just redirect the model to a wrong answer on
"success," it also replaces 5 of the passages in context with poison text
that crowds out real supporting evidence even on `attack_success=0` rows,
depressing gold-answer F1 independently of whether the target answer was
actually adopted.

## Why the corpus effect is this large: two contributing, not competing, mechanisms

Both hypotheses named in the request line up with real, measured numbers
from this same data — not just plausible stories:

**1. Retrieval competition dilutes poison presence on `ms_marco`.** Mean
retrieval precision at k (the fraction of the top-5 retrieved documents that
are actually the 5 planted poison passages, from `poisoned_contexts_*.json`,
computed once at Phase-A build time and shared across all 4 models) is
**99.3% on `hotpot_qa` vs 88.96% on `ms_marco`**. On `hotpot_qa`, the
retriever returns essentially only poison for a poisoned question — the real
corpus barely competes. On `ms_marco`, roughly 1 in 9 of the retrieved slots
goes to a real, unpoisoned passage instead, diluting the poison's presence in
context on top of any question-answering effect. This matches the
0.8-precision-ballpark finding flagged from the earlier smoke test and now
confirmed at full scale.

**2. `hotpot_qa`'s multi-hop structure makes a single poisoned passage
easier to make persuasive.** Mean question length (word count, same 90/96
poisoned questions) is **18.17 words on `hotpot_qa` vs 6.14 words on
`ms_marco`** — `hotpot_qa` questions are long, multi-clause, multi-entity
descriptions ("Which club [...] Romanian international footballer [...]
under coach Cesare Prandelli [...] did join in late August 2002?"), while
`ms_marco` questions are short, keyword-style web queries. The PoisonedRAG
generator's ~100-word crafted passage has substantially more surface to work
with when it can weave a fabricated answer into a long, specific,
multi-clue question than when it has to invent supporting "evidence" for a
short, generic query — and `ms_marco`'s real corpus passages are themselves
short, informal snippets (see `PHASE2_INJECTION_INSIGHTS.md`'s phi-4-mini
discussion of this same corpus property), giving genuine competing passages
less distance to close against a fabricated one.

These two mechanisms are not mutually exclusive and both point the same
direction — retrieval competition and question/passage complexity both favor
`hotpot_qa` poisoning, and the observed effect size (a 50+ point ASR gap) is
plausibly the product of both operating together rather than either alone
fully explaining it. Disentangling their relative contribution (e.g. via a
controlled experiment holding retrieval precision fixed across corpora)
would need a follow-up study; this document reports both as real,
data-supported contributing factors rather than claiming to have isolated
one as dominant.

## Skip/loss rate: 10/100 `hotpot_qa`, 4/100 `ms_marco`

The cached `poisoned_contexts_*.json` files retain only successfully-built
poison contexts (90 and 96 entries respectively) — Phase A's generation
pipeline never persists raw content for a question it ultimately skips (see
`attacks/poisonedrag.py`'s `generate_poison_texts`, which only prints failed
raw responses to stderr, confirmed while fixing the parser bug below). The
breakdown below is therefore reconstructed from real-time monitoring notes
taken during the live sweep, not from a retained log file or a rescoreable
cache — it should be read as an honest field account of what was observed
as it happened, not a re-derivable statistic.

Three distinct, confirmed failure modes contributed to the 14 total skips:

1. **Genuine content-policy refusals — rare (2 confirmed cases, both
   `hotpot_qa`).** The generator explicitly declined to fabricate
   misinformation about a real, named, identifiable subject: a real
   political scandal (Bridgegate) and a real named actor (Skyler Gisondo).
   Early in the sweep (first ~10 questions) these looked like they might be
   a systematic bias against real-world-entity questions, prompting close
   monitoring through the rest of the run — but no further refusals of this
   kind appeared in the remaining ~90 questions. At n=2 out of 100, this is
   too small a sample to characterize as a systematic entity bias with any
   confidence; it's reported here as a real, observed, rare phenomenon, not
   a resolved pattern. A dedicated study varying entity type and prominence
   would be needed to determine whether this generalizes.
2. **Reasoning-token-exhaustion truncations — the majority of skips.**
   `finish_reason="length"` before all 5 corpus sections were produced,
   confirmed as the dominant cause once the sweep moved past its first
   small, unlucky cluster (an early ~25-30% short-window skip rate that did
   not hold — the settled rate was close to 10% overall). This is the same
   underlying failure class documented in `attacks/poisonedrag.py`'s module
   docstring (nemotron's internal reasoning stage competing with the
   requested output for the same token budget) — the `"detailed thinking
   off"` system message reduces but does not eliminate it, since reasoning
   can still consume enough of the budget on longer, more complex questions
   to truncate the final corpus sections even without leaking into
   `content`. Topic sensitivity does not explain this bucket: truncations
   were observed across ordinary, unremarkable topics (Olympic results,
   film trivia, TV casting, awards) with no apparent thematic pattern,
   consistent with non-deterministic reasoning-token variance rather than
   content-based avoidance.
3. **Parser formatting gap — 2 lost, now fixed for future runs.** Two
   `ms_marco` questions produced complete, well-formed, high-quality
   5-corpus generations (`finish_reason="stop"`, nothing truncated) that
   used a markdown-heading label (`### Incorrect Answer`) instead of the
   expected bold-inline label (`**Incorrect Answer:**`), which the parser's
   regex did not yet accept — a real code bug discarding an otherwise
   fully successful generation, not a model limitation. Fixed in
   `attacks/poisonedrag.py` (commit `facd3f2`, this branch) with a
   narrowly-scoped fallback regex plus two real fixtures captured from the
   exact failing responses. The fix is forward-looking only — these two
   specific `ms_marco` questions remain skipped in this sweep's n=96, since
   no raw-response cache exists to rescore from; regenerating them would
   need 2 fresh API calls, not yet done.

The remaining skips (10 total minus 2 refusals = 8 on `hotpot_qa`; 4 total
minus 2 parser-bug = 2 on `ms_marco`) are attributed to truncation by
elimination, consistent with what was directly observed during monitoring,
though not each individually confirmed via a retained per-question log.

**Net assessment:** the dominant loss mechanism is non-deterministic
reasoning-budget variance, not a systematic bias against sensitive content —
consistent with the "revised hypothesis" tracked during live monitoring.
Genuine refusals are real and worth documenting as a citable limitation of
using a safety-aligned model as a poison generator (a different generator,
or a jailbroken/fine-tuned one, would presumably show a different, likely
lower, refusal rate on this axis) — but at 2/100, they are a minor
contributor to the overall skip rate compared to token-budget truncation.
This is reported here as a genuine finding worth citing, not buried in a
footnote, per the standard this project holds ASR-scoring caveats to
elsewhere (see `PHASE2_INJECTION_INSIGHTS.md`'s "ASR scoring conflates three
phenomena" section for the parallel precedent).

## Methodology notes (for the dissertation's methodology chapter)

**McNemar vs Fisher's exact, and why both are used** — identical reasoning
to Attack 1 (see `PHASE2_INJECTION_INSIGHTS.md`'s own methodology section):
model-vs-model comparisons within one corpus are genuinely paired (all 4
models are evaluated against the same shared, cached `poisoned_contexts_*
.json` question set for that corpus), so McNemar's exact test applies.
Corpus-vs-corpus comparisons are not paired — `hotpot_qa` and `ms_marco`'s
100 target questions are independently sampled from different underlying
corpora with no item-to-item correspondence — so these use
`evaluation.stats.fisher_exact_asr_comparison` on independent proportions
instead. Both are exact (hypergeometric/binomial), both feed one shared
Holm-Bonferroni family.

**One family, not one per poison config.** Attack 1 corrected each of its 5
injection templates as an independently Holm-Bonferroni-corrected family,
since each template is a mechanistically distinct delivery method.
PoisonedRAG has only one poison configuration (`adv5`, `ADV_PER_QUERY=5`
fixed for this study — see `attacks/poisonedrag.py`'s module docstring), so
all 16 comparisons (12 McNemar + 4 Fisher) belong to a single family, matching
the same "one family per genuinely distinct condition" principle Attack 1
established.

**Utility metric: `f1_clean` (gold-answer F1), not `f1_target`.** PoisonedRAG's
raw rows carry two separate F1 fields: `f1_target` scores similarity to the
*poisoned* target answer (used as part of `attack_success` scoring) and
`f1_clean` scores similarity to the real gold answer, computed identically to
Phase 1's baseline `f1_clean` field. The utility-under-attack analysis here
uses `f1_clean` exclusively, paired against the same field in
`phase1_results_complete/`'s baseline rows by question text — the correct
substitution for what this document's Statistics section calls "the utility
drop," since utility is about answering the real question correctly, not
about how closely the model matched the attacker's target.

**Generator journey — a real methodological finding, not incidental
setup.** Three poison-generator choices were tried in sequence before
settling on the one used for this sweep, each abandoned for a concrete,
verified reason rather than a preference, documented in full in
`attacks/poisonedrag.py`'s module docstring and reproduced here as a citable
methods note:

1. **`kimi-k3` via NVIDIA NIM** (~2.8T parameters, the largest catalog entry
   found on either platform checked) — reachable, confirmed working at a
   measured 480 seconds for a two-word test reply. Excluded on
   cost/practicality grounds: 480s for a trivial reply implies the real
   poison-generation prompt (5 corpora + an incorrect answer, per call)
   would plausibly take minutes per call across ~200 real target-question
   calls (100 per corpus × 2 corpora) — not a viable sweep runtime.
2. **`moonshotai/Kimi-K2-Instruct-0905` via the HuggingFace Inference API
   router** (1T total / 32B active) — worked, including a real
   reasoning-token-exhaustion bug found and fixed
   (`chat_template_kwargs.thinking=false`) — but HF's Inference API free
   tier turned out to be a $0.10/month cap in practice, not a genuinely
   free research tier, and paid HF credits weren't judged the right fix
   given this project also needs real API volume later for the Crescendo
   attack (Phase 2, Attack 3). Abandoned for cost/sustainability, not
   quality or access — the reasoning-disable fix that made it work remains
   valid, just moot for this generator choice.
3. **`nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM** — the generator
   actually used for this sweep. NIM's hosted developer tier is free for
   prototyping/research/development/evaluation (not a dollar-credit cap),
   rate-limited at ~40 requests/minute, confirmed via NVIDIA Developer
   Program documentation and independent forum reports, not just asserted.
   This model also has an internal reasoning stage; on at least one real
   call during earlier development it leaked that reasoning (visible
   word-counting, self-correction) directly into the response's `content`
   field rather than a separate `reasoning_content` field, then was cut off
   mid-reasoning by `finish_reason="length"` — the same underlying cause as
   Kimi's exhaustion bug (reasoning competing with the requested output for
   token budget), a different failure shape. Fix: a
   `{"role": "system", "content": "detailed thinking off"}` message
   (NVIDIA's documented Nemotron convention, distinct from Kimi's
   OpenAI-style `chat_template_kwargs` mechanism), confirmed on the
   question that had just failed plus 3 further fresh questions before
   being trusted for the full sweep.
4. **`minimax/minimax-m3:free` via OpenRouter — verified as a backup, not
   used for this sweep's data.** Evaluated during the same development pass
   as an alternative: genuinely clean by default with no thinking-disable
   flag needed, but produces a structurally different response shape (a
   collective `**Corpora:**` header plus a numbered list, rather than
   per-item `**Corpus N:**` headers) that required its own parser fallback
   path (`_COLLECTIVE_CORPORA_HEADER_RE` + `_NUMBERED_LIST_ITEM_RE` in
   `attacks/poisonedrag.py`, covered by real fixtures in
   `tests/test_poisonedrag.py`). Kept as a documented, tested fallback
   option, not deployed for this sweep.

This progression — three real transport/model choices tried, two abandoned
for verified engineering reasons (latency, billing model) rather than
quality, one kept as a tested-but-unused backup — is reported here in full
as a legitimate part of the methodology, following the same standard this
project applies to reasoning-leak and parser-format findings elsewhere: real
engineering friction encountered while building an attack pipeline is a
citable part of how the results were produced, not something to omit as
implementation trivia.

## What this means for Phase 3

- A defense evaluated against PoisonedRAG needs to be judged primarily by
  corpus, not by model — a defense that closes the gap on `hotpot_qa`-style
  content (long, multi-clause, multi-entity questions where poison
  dominates retrieval) should not be assumed to generalize to `ms_marco`-
  style content, and vice versa, given how large and consistent this
  document's corpus split is across every one of the 4 models tested.
- Because model identity was *not* a significant factor here (unlike Attack
  1), a single defense configuration is a reasonable choice to test across
  all 4 models for PoisonedRAG specifically — there is no model-specific
  tuning signal in this data to design around, unlike Attack 1's
  ministral-3-8b/qwen3-8b-specific process-hijack gaps.
- The retrieval-precision finding (99.3% vs 88.96%) suggests a
  retrieval-side defense (e.g. down-weighting near-duplicate or
  suspiciously question-similar passages, since PoisonedRAG's
  `retrieval_piece = question + "."` construction makes poisoned passages
  unusually close to the query itself) is a promising angle worth testing
  specifically on `hotpot_qa`, where poison already crowds out nearly all
  competing real content.
- The genuine-refusal finding (2/100, real but rare) is worth keeping in
  mind for any future PoisonedRAG-style sweep using a safety-aligned
  generator model: expect a small, non-zero rate of content genuinely
  impossible to generate for real, sensitive, named entities, independent
  of any pipeline bug.

## Limitations

- Skip/loss breakdown (refusals vs. truncations vs. parser bug) is
  reconstructed from real-time monitoring notes taken during the live
  sweep, not from a retained per-question log or a rescoreable cache — the
  exact count attributed to truncation-by-elimination (8 of 10 on
  `hotpot_qa`, 2 of 4 on `ms_marco`) was not individually confirmed
  per-question.
- Genuine content-policy refusals (n=2) are too small a sample to
  characterize as a systematic bias against real-world-entity questions
  with statistical confidence — reported as a real, rare, observed
  phenomenon, not a resolved pattern requiring a specific mitigation.
- The retrieval-competition and question-complexity explanations for the
  corpus effect are both real, data-supported contributing factors, not
  disentangled from each other — this document does not claim to know
  their relative weight, only that both point the same direction and both
  are measurable in this data.
- Two `ms_marco` poison contexts lost to the (now-fixed) parser bug were
  not regenerated for this document's n=96 — a rescore including them
  would need 2 fresh API calls, not performed here.
- `nq_open` excluded entirely from this attack, same project-wide
  constraint as Attack 1 (see `nq_open_leakage_finding.md`) — these
  findings describe `hotpot_qa` and `ms_marco` only.
- One poison configuration (`ADV_PER_QUERY=5`) and one generator model
  (`nemotron-3-ultra-550b-a55b`) — this document cannot distinguish
  "PoisonedRAG doesn't work well on `ms_marco`" from "this specific
  generator's poison passages don't work well on `ms_marco`"; a different
  generator or a larger `ADV_PER_QUERY` might narrow the corpus gap.
