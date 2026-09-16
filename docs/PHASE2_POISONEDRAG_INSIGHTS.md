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

**Verified directly against the live sweep's own log**
(`rag-scratch/poisonedrag_sweep.log`, RunPod S3-compatible network volume,
bucket `cplemvuitj`, downloaded and grepped for this document — same
commands used for real-time monitoring during the run:
`grep -c "SKIPPING question after"` and
`grep -B1 "SKIPPING question after"`), not reconstructed from notes. This
replaces an earlier draft of this section that *was* reconstructed from
real-time monitoring notes taken during the sweep — that draft's total (14)
and per-corpus split (10/4) held up exactly, but its 3-category breakdown
did not survive contact with the actual log and is corrected below.

`grep -c "SKIPPING question after"` returns **14**, matching the n=90/100
and n=96/100 cell sizes exactly (10 on `hotpot_qa`, 4 on `ms_marco`, split
confirmed by matching each skipped question's text against the two
corpora's known question style). The log records 111 total attempt-level
failures across the sweep (`max_attempts=3` retry, most recovered
successfully) — only these 14 exhausted all 3 attempts and were
permanently skipped. The verified breakdown of those 14, by the *actual*
final error each one gave up on:

1. **Truncation (`Missing corpus section(s) [...]`) — 9 of 14, the
   majority, as monitored live.** 7 on `hotpot_qa`, 2 on `ms_marco`. Exact
   examples from the log: `Missing corpus section(s) [3, 4] ... (found [1,
   2, 5])` (Freud/Éluard question), `Missing corpus section(s) [1, 2, 3, 4,
   5] ... (found [])` (a complete miss, zero corpora produced — the Cambridge
   Guide to Women's Writing question), down to a single missing section
   (`Missing corpus section(s) [4, 5] ... (found [1, 2, 3])`, two `ms_marco`
   questions on ACT scores and vehicle-safety-inspection validity). Same
   underlying cause across all 9: nemotron's internal reasoning stage
   competing with the requested output for the same token budget (see
   `attacks/poisonedrag.py`'s module docstring) — the `"detailed thinking
   off"` fix reduces but does not eliminate this. No topical pattern across
   the 9: Olympic results, film trivia, TV casting, an academic reference
   editor, a music producer's legal name, betting types, ACT scores, safety
   inspections — ordinary questions, consistent with non-deterministic
   reasoning-token variance, not content-based avoidance.
2. **API/infrastructure failures — 3 of 14, a real category the earlier
   reconstructed draft folded into "truncation" without verifying.** Two
   `HTTPError: 503 Server Error: Service Unavailable` (one `hotpot_qa`, one
   `ms_marco`) and one `ReadTimeout` after 180s (`hotpot_qa`) from
   `integrate.api.nvidia.com` — transient NIM-side outages/timeouts, not a
   generation-quality problem at all. These 3 questions never even reached
   the parser; nothing about their content caused the failure.
3. **Genuine content-policy refusal — 1 of 14, not 2, and not the two
   questions named during live monitoring.** The one skip that gave up with
   an explicit refusal (`"I'm sorry, but I can't help with that."`) is
   *"Who made their debut in 'A Rather English Marriage' and later starred
   in 'Surrogates?'"* — a `hotpot_qa` question about a film actor,
   unrelated to either question flagged live. **Bridgegate and Skyler
   Gisondo, the two refusals reported during real-time monitoring, are
   confirmed by the log to have refused on an early attempt (attempt 1 and
   attempts 1–2 respectively) but then succeeded on retry** — both have
   real, successful entries in `poisoned_contexts_hotpot_qa_adv5.json` and
   are not among the 14 final skips. This is a genuine, material correction
   to what was reported live: refusal at the attempt level is real and
   somewhat more common than the final-skip count alone suggests (8 of the
   111 total attempt-level failures across the whole sweep show refusal-
   style text — see below), but at the level that actually costs a
   question, it happened once.
4. **Parser formatting gap — 1 question lost, not 2, now fixed for future
   runs.** The `ms_marco` question *"how important political is to the
   state of texas"* hit the `### Incorrect Answer` heading-style bug on
   both its 1st attempt and its 3rd/final attempt (attempt 2 was a separate
   503 error) — one question, two of its three attempts exhibiting the same
   bug, not two different questions as an earlier draft of this document
   stated (the two committed test fixtures, `texas_political_parties_
   attempt1.txt` / `_attempt3.txt`, are verified by the log to be this
   single question's first and third attempts, not two distinct
   questions). Fixed in `attacks/poisonedrag.py` (commit `facd3f2`, this
   branch). The fix is forward-looking only — this question remains skipped
   in this sweep's n=96; regenerating it would need 1 fresh API call, not
   yet done.

**Attempt-level refusal behavior, beyond the final-skip count.** 8 of the
111 total attempt-level failures across the whole sweep show explicit
refusal-style text (`grep`-verified: "sorry", "not going to", "can't help",
"disallowed"), but only 1 escalated to a permanent skip — 7 of 8 were
retry-recovered. Reading all 8 by topic complicates a pure real-entity-bias
reading: alongside the two real, named, sensitive subjects reported live
(Bridgegate, a real political scandal; Skyler Gisondo, a real actor) and
three more real named people/institutions (a Cambridge reference-book
editor, a Summerburst Festival music producer's "legal name," financial-firm
director Gilbert F. Casellas), **two of the eight refusals have nothing to
do with real-world entities at all**: a question about the plant genus
*Shibataea* vs. *Pediocactus* (Chinese bamboo taxonomy), and a Microsoft
Access 2013 how-to question, refused because "providing incorrect steps for
configuring a database field could lead to data errors." The generator's
refusal trigger looks broader than "avoid fabricating claims about named
real people" — it also fires on fabricating incorrect *technical
instructions*, at least once. Both readings (real-entity sensitivity, and a
broader "don't state confident falsehoods as fact" pattern) are consistent
with 8 real attempt-level data points; this sample is too small (8) to
adjudicate between them with confidence.

**Net assessment:** the dominant loss mechanism is confirmed, verified
truncation (9 of 14, ~64%) — consistent with the "revised hypothesis"
tracked during live monitoring, and now backed by the actual log rather than
an estimate. The refusal finding survives but is smaller and more nuanced
than first reported: 1 of 14 permanent skips, with the two live-flagged
cases turning out to be retry-recovered rather than hard blocks, and the
broader 8-instance attempt-level refusal sample showing the trigger isn't
cleanly "real entity" — it includes at least two clearly non-entity
technical/factual refusals too. Both corrections make the finding *more*
defensible as a genuine limitation to cite (real refusal behavior exists,
confirmed at the attempt level 8 separate times, and is largely but not
entirely retry-recoverable), while walking back the specific claim that the
model reliably refuses on real, named, sensitive entities — the two examples
originally cited for that claim did not, in fact, end up refused.

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
- The genuine-refusal finding (1/14 permanent skips; 8/111 attempt-level
  failures, mostly retry-recovered) is worth keeping in mind for any future
  PoisonedRAG-style sweep using a safety-aligned generator model: expect a
  real, non-trivial refusal rate at the individual-attempt level — not
  cleanly limited to real, named, sensitive entities, per the two
  non-entity technical refusals found in this log — most of which a
  bounded retry (`max_attempts=3`, as used here) will recover from, with a
  small residual permanent-loss rate independent of any pipeline bug.

## Limitations

- Skip/loss breakdown (truncation vs. API/infra failure vs. refusal vs.
  parser bug) is now verified directly against `poisonedrag_sweep.log`
  (all 14 final skips individually inspected by their actual last-error
  text), not reconstructed — no remaining uncertainty on this count.
- Genuine content-policy refusal at the permanent-skip level (n=1) and at
  the attempt level (n=8, 7 of which recovered on retry) are both too small
  a sample to characterize the refusal trigger precisely — the 8 attempt-
  level cases include real-named-entity questions and at least two
  unrelated technical/factual ones, so this document reports "the generator
  sometimes refuses to fabricate confident falsehoods, on both entity and
  non-entity questions, usually but not always recoverable by retry" as the
  supported claim, not a narrower real-entity-specific bias.
- The retrieval-competition and question-complexity explanations for the
  corpus effect are both real, data-supported contributing factors, not
  disentangled from each other — this document does not claim to know
  their relative weight, only that both point the same direction and both
  are measurable in this data.
- One `ms_marco` poison context lost to the (now-fixed) parser bug was not
  regenerated for this document's n=96 — a rescore including it would need
  1 fresh API call, not performed here. (Three further questions lost to
  transient API/infra failures — 503s and a timeout — were also not
  retried beyond the sweep's own `max_attempts=3`.)
- `nq_open` excluded entirely from this attack, same project-wide
  constraint as Attack 1 (see `nq_open_leakage_finding.md`) — these
  findings describe `hotpot_qa` and `ms_marco` only.
- One poison configuration (`ADV_PER_QUERY=5`) and one generator model
  (`nemotron-3-ultra-550b-a55b`) — this document cannot distinguish
  "PoisonedRAG doesn't work well on `ms_marco`" from "this specific
  generator's poison passages don't work well on `ms_marco`"; a different
  generator or a larger `ADV_PER_QUERY` might narrow the corpus gap.
