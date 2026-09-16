# Adversarial Robustness of Open-Source RAG Pipelines — Master Record

Single, self-contained record of the entire thesis project: research design, all
three project phases, every real number currently committed to this repository,
the engineering narrative behind them, and what's still open. Built for a
supervisor meeting and as the backbone for the thesis write-up — every number
below traces to a specific file in this repository, cited inline. Where source
material was genuinely unclear, contradictory, or still open, that is stated
explicitly rather than resolved by guessing.

**Compiled:** 2026-09-16, from `PHASE2_INJECTION_INSIGHTS.md`,
`PHASE2_POISONEDRAG_INSIGHTS.md`, `PHASE2_CRESCENDO_INSIGHTS.md`,
`PHASE3_DEFENSE_INSIGHTS.md` (commit `ce66b5a`, the final corrected version),
`nq_open_leakage_finding.md`, `README.md`, `phase1_results_complete/`'s raw
summary CSVs, every task-brief `.md` file in the repo root, and this
repository's own git history for `defenses/output_filter.py` and
`defenses/instruction_detection.py`. **No `CITATIONS.md` exists anywhere in
this repository** (confirmed via repo-wide search) — Section 12's reference
list was reconstructed from the "Grounding" sections of the three Phase 2
task briefs and the model/dataset identifiers used directly in code and
config; it is not a copy of a pre-existing citation file, because none exists.

---

## 1. Executive Summary

**Research question.** Under the EU AI Act's Article 15 robustness
requirements, how vulnerable are small-to-mid-size (4–12B parameter)
open-source LLMs to adversarial manipulation when deployed in a
retrieval-augmented generation (RAG) pipeline, and how effective are
practical, low-cost defenses at closing that gap without destroying the
system's underlying utility? (`README.md:3-4`.)

**What was built.** A four-model × three-corpus RAG benchmark harness (Phase
1), three structurally distinct attack surfaces implemented against it —
indirect prompt injection, corpus poisoning (PoisonedRAG), and multi-turn
conversational jailbreaking (Crescendo) (Phase 2) — and three defenses
evaluated against those attacks: a retrieval-stage injection classifier
(`instruction_detection`), a prompt-structuring technique
(`spotlighting`), and a post-generation safety filter (`output_filter`)
(Phase 3). Every phase's numbers are backed by real generations on real
hardware (RunPod A100, HuggingFace `transformers` and vLLM), not simulated
or estimated.

**Headline results across all three attacks.** The three attacks turn out to
decompose along **three different dominant axes**, which is itself one of
this project's central findings: indirect injection's success is dominated
by *which model* answers (model identity, with large, model-specific
mechanism differences — Section 5.1); PoisonedRAG's success is dominated by
*which corpus* is poisoned (corpus identity, with all four models
statistically indistinguishable — Section 5.2); Crescendo's outcome (ASR)
doesn't significantly separate the four models at all, but the *mechanism*
each model uses to reach that outcome is a strong, per-model signature —
refuse-constantly-but-fold vs. barely-refuse-at-all vs. refuse-and-often-hold
(Section 5.3). No single "robustness score" summarizes all three; a model or
defense good at one axis is not thereby predicted to be good at another.

**Headline results across all three defenses.** Once a real inference-backend
confound (`hf` vs `vllm`) discovered mid-analysis was fully resolved (Section
8), the corrected picture is: `spotlighting` is the only defense with a
real, backend-isolated effect on injection ASR (9 of 16 real cells remain
significant), but at a severe utility cost (mean ΔF1 = −0.283, as low as
−0.44 in some cells). `output_filter` and `instruction_detection` show **no
significant backend-isolated effect on injection ASR** (0/19 and 0/16
cells) — their apparent effect in the uncorrected numbers was almost
entirely the `hf`/`vllm` backend switch, not the defense. Against
PoisonedRAG, `spotlighting` is again the only defense with any significant
cells (4/24, all backend-clean from the start — no confound ever existed for
PoisonedRAG); `output_filter` is a **structural mismatch** against
PoisonedRAG — a safety classifier flags 0.0000 of 160 scored poisoned
responses, because factually-wrong-but-safe-sounding answers are simply
outside what a safety classifier screens for (Section 6.3). All three
defenses cost roughly the same large utility hit against PoisonedRAG
(mean ΔF1 ≈ −0.29 to −0.31) because none of them repair the underlying
retrieval corruption.

**Meta-finding: real-hardware verification was not optional.** Eight
distinct, genuinely blocking bugs in the two production defense modules
(`output_filter.py`, `instruction_detection.py`) were invisible to this
project's own test suite and were only found by running on the real
RunPod A100 pod against real model weights — a chat-template alternation
crash, a tensor/`BatchEncoding` shape mismatch, three rounds of Llama4
chunked-attention/cache-construction crashes, a `generation_config`-baked
cache conflict, a silent truncation no-op that produced a multi-hour
near-hang, and an uncapped CPU thread pool causing a 12.6× user/real time
ratio (Section 7). Every one of these bugs passed the existing mocked test
suite cleanly, because the mocks were built before the bug existed and
never exercised the real code path that broke. This is treated in this
project as a citable methodological finding in its own right, not
incidental engineering trivia: **for this class of system — real model
weights, real tokenizer/chat-template behavior, real hardware-dependent
performance characteristics — a green test suite at small scale is not
evidence of correctness at real scale**, and a defense evaluation that never
executed on real hardware could easily have reported the *wrong* passing
numbers rather than no numbers at all.

**A second meta-finding, about intellectual honesty as method.** A real
inference-backend inconsistency (Phase 2 injection's baseline ran on `vllm`;
every Phase 3 defended run ran on `hf`) was discovered mid-analysis, not
before it. Rather than leave it as a caveat, it was investigated directly —
a backend-matched, no-defense baseline was built and compared three ways
per cell — which fully overturned two of the three defenses' headline
results (Section 8). The same discipline applies to this document: **one
finding remains genuinely open and is not resolved here** — `ministral-3-8b`'s
prompt-injection ASR figures (`ignore`/`fake_completion` templates) are
flagged provisional pending a chat-template round-trip diagnostic that was
never run; see Section 5.1 and Section 10.

---

## 2. Project Scope & Research Design

**Models (4):** `llama-3.1-8b` (Meta Llama-3.1-8B-Instruct, gated, requires
license acceptance), `ministral-3-8b` (Mistral3ForConditionalGeneration,
Ministral-3-8B-Instruct-2512 — internally routes through a vision tower that
is confirmed unused for this project's text-only inputs), `phi-4-mini`
(Microsoft Phi-4-mini-instruct), `qwen3-8b` (Alibaba Qwen3-8B, thinking mode
disabled project-wide via `config.QWEN3_ENABLE_THINKING=False`). Every
model is pinned to an exact HuggingFace commit hash in `config.py`
("every phase must load byte-identical weights") — the same discipline was
later extended to corpora (`corpus_revision_pinning_task.md`, Section 7.5).

**Corpora (3, one excluded from all comparative analysis):** `hotpot_qa`,
`ms_marco`, and `nq_open`. **`nq_open` is excluded from every attack,
defense, and cross-corpus comparison in this project**, by construction, not
by choice of convenience — see Section 4's full explanation. Its Phase 1
baseline numbers are reported once, as a documented limitation, and nowhere
else.

**Three attacks, three structurally distinct surfaces (Phase 2):**
indirect prompt injection (plants a wrong *instruction*), PoisonedRAG
(plants a wrong *fact*), Crescendo (exploits multi-turn *conversational
state*, no RAG-corpus axis at all). Each was deliberately chosen to test a
genuinely different attack surface rather than three variations on one
mechanism (`phase2_indirect_injection_task.md:15`,
`phase2_poisonedrag_task.md:13`, `phase2_crescendo_task.md:15`).

**Three defenses, three structurally distinct mechanisms (Phase 3):**
`instruction_detection` (a `protectai/deberta-v3-base-prompt-injection-v2`
classifier run per retrieved passage, pre-generation, dropping flagged
passages), `spotlighting` (base64-encoding the retrieved context so
injected natural-language instructions inside it can't be parsed as
instructions), `output_filter` (a `meta-llama/Llama-Guard-4-12B` safety
classifier run post-generation on the model's final response, replacing
flagged output with a fixed marker). Each targets a different point in the
pipeline (retrieval-stage filtering, prompt structuring, output-stage
filtering) — again a deliberate design choice to sample genuinely different
defense mechanisms, not three parameterizations of one idea.

**Why this matrix.** The 4×3(×3 attacks ×3 defenses) design was chosen so
that "which model," "which corpus," and "which mechanism" could each be
isolated as independent variables. This paid off directly: Section 5 shows
the three attacks are dominated by three *different* factors (model
identity, corpus identity, and mechanism-without-outcome-separation
respectively) — a design that couldn't distinguish "attacks differ" from
"a single attack happens to look different across conditions" would have
missed this.

**EU AI Act Article 15 framing.** Per `README.md`'s own framing (the only
place in this repository's currently committed material that states the
project's regulatory motivation explicitly): this is a "cross-model
benchmark of prompt injection, knowledge poisoning, and multi-turn attacks
under EU AI Act Article 15" — Article 15 requires high-risk AI systems to
achieve "an appropriate level of accuracy, robustness, and cybersecurity"
and to be resilient against attempts to exploit system vulnerabilities. No
separate scope/overview document elaborating this framing further (a
dedicated Article 15 mapping, specific sub-requirements addressed, etc.) was
found in this repository as of this compilation — see Section 11.

---

## 3. Infrastructure & Methodology

### 3.1 Tech stack

- **Inference:** two parallel engines behind an `INFERENCE_ENGINE` env
  var — HuggingFace `transformers` (`hf`, the default, used for every Phase
  3 defended run and for PoisonedRAG's and Crescendo's Phase 2 baselines)
  and vLLM (`vllm`, added in Stage 2, used for Phase 1's baseline sweep and
  Phase 2 injection's baseline sweep — the source of the backend confound
  investigated in Section 8). vLLM auto-selects bf16 on the A100 (compute
  capability 8.0) and auto-detects `ministral-3-8b`'s native fp8 checkpoint
  via its Marlin kernel path — a memory-footprint difference from the plain
  HF path, which dequantizes that same checkpoint to bf16 on any GPU below
  compute capability 8.9 (`stage2_vllm_task.md:30-36`).
- **Hardware:** RunPod A100 GPU pods (migrated from an earlier Azure
  Container Apps deployment — see Section 3.3).
- **Build/deploy:** Docker image built and pushed to GHCR (GitHub Container
  Registry) on every push to `main`; the running of a paid GPU sweep is a
  separate, explicit, human-triggered action, never automatic on push (see
  Section 3.3 for why this separation exists and what it replaced).
- **Retrieval:** FAISS dense index per corpus, `sentence-transformers`
  embedder, built once and cached with atomic-write + marker-file integrity
  guarantees (Section 3.4).

### 3.2 Statistical methods, and why each was chosen where it was

- **McNemar's exact test** (paired, hypergeometric) — used for every
  model-vs-model comparison within a fixed corpus/behavior-set, because all
  models in this project are evaluated on the identical seeded item sample
  for a given corpus — a genuinely paired design.
- **Fisher's exact test** (unpaired, on independent proportions) — used
  wherever two conditions have no item-to-item correspondence, most notably
  corpus-vs-corpus comparisons (`hotpot_qa` vs `ms_marco` are different
  question sets), where treating them as paired would violate McNemar's
  core assumption.
- **Holm-Bonferroni correction**, applied within one family per genuinely
  distinct condition (e.g. one family per injection template, since each is
  a mechanistically distinct delivery method; pooling would dilute power
  between hypotheses that aren't testing the same thing).
- **Paired bootstrap 95% CI** (10,000 resamples) — used for utility-drop
  (F1) comparisons, kept structurally separate from the binary-outcome
  significance tests (ASR via McNemar/Fisher) so "which models/corpora
  differ in vulnerability" and "how much utility does this attack cost"
  stay two cleanly separated statistical questions rather than one
  conflated metric.
- **Pairwise McNemar with Holm-Bonferroni, applied within each cell's own
  3-comparison family** — the method chosen for Phase 3's backend-confound
  isolation (Section 8), over Cochran's Q, because the actual question is
  *which* of three pairwise gaps differs (attributing a reduction to
  backend vs. defense), not merely whether the three conditions differ
  somewhere — Cochran's Q would still need a post-hoc pairwise test to
  answer that, making pairwise McNemar the more direct fit, not just cheaper
  reuse of existing helpers (`PHASE3_DEFENSE_INSIGHTS.md:308-312`).

All tests are exact (hypergeometric/binomial, never chi-square-approximated)
and implemented without a `scipy`/`statsmodels` dependency
(`evaluation/stats.py`).

### 3.3 The Azure → RunPod infrastructure migration, briefly

Phase 1/2 development began on Azure Container Apps. That platform's
revision-update behavior meant **pushing to `main` automatically started a
paid GPU deployment** — the mechanism behind an actual overnight
restart-loop billing scare early in the project. The fix was not a
manual-trigger flag on the same resource type but switching to a **Container
Apps Job**, a structurally different resource type that does not
auto-execute on image update (`stage3_partA_claude_code.md:18-23`) — Task A1
of that stage explicitly forbids ever adding a `job start` call to CI,
making "running a sweep" a permanent, separate, human action. `PHASE2_PREP_LOG.md`'s
Phase A audit (2026-08-31) later found this fix had **not yet reached
`main`** at that point — `main`'s live workflow was still the old
auto-deploying Container App — confirmed directly by reading the merged
workflow file rather than trusting the PR history, and closed only once
that file was verified, post-merge, to contain zero `containerapp`/
`azure/CLI`/`azure/login` references.

The project subsequently moved to RunPod, which has no CI/CD trigger of any
kind — pod launches are a manual "Start command" override, confirmed via a
full audit that RunPod is referenced nowhere in any workflow file on any
branch (`PHASE2_PREP_LOG.md:49`). This introduced its own new failure mode —
a pod that finishes and never terminates bills indefinitely — addressed by
`scripts/run_and_terminate.py`, whose own history is a small, real
engineering narrative:

1. **Verify actual success, not just exit code 0** — `verify_success()`
   checks that expected JSONL/CSV output files exist and are non-empty
   before ever allowing termination, given this project's own history of
   silent-looking failures elsewhere (`runpod_self_termination_task.md`).
2. **Never terminate on failure** — a crashed run stays alive for log
   inspection rather than erasing its own evidence (same file).
3. **A hang never exits, so it never terminates either** — a real smoke
   test found `phi-4-mini` load complete, then the pod idling at 0%
   utilization for 12+ minutes with zero telemetry movement; the pipeline
   subprocess is now wrapped in a hard timeout (`RAG_PIPELINE_TIMEOUT_SECONDS`,
   default 6h) that kills the whole process group, not just the direct
   child, on expiry (`fix_hang_and_corpora_filter.md`, Bug 1).
4. **`RAG_CORPORA` silently didn't filter anything** — a pod launched with
   `RAG_CORPORA=nq_open` still built indices for `ms_marco` too; fixed in
   the same commit (`fix_hang_and_corpora_filter.md`, Bug 2).
5. **The `runpod` SDK conflicts with `vllm`** — `pip install -r
   requirements.txt` failed outright on a real build: `vllm==0.25.1`
   requires `fastapi<0.137.0`, `runpod==1.11.0` requires `fastapi>=0.139.0`,
   non-overlapping ranges. Fixed by dropping the SDK entirely and calling
   RunPod's terminate endpoint as a plain `requests.delete()` HTTP call,
   confirmed with an actual local Docker build (`fix_runpod_fastapi_conflict.md`)
   — a direct lesson that mocking a dependency for unit tests does not
   substitute for confirming it's actually installable alongside everything
   else.
6. **A failed *termination* call must never crash the script** — observed
   twice on real pods: pipeline succeeds, results are safely written, then
   the termination REST call itself fails (403, a separate unresolved
   credential issue), the unhandled `HTTPError` crashes the script, and
   RunPod's own orchestration reads the crash-exit as "needs retrying,"
   triggering a full, expensive, from-scratch re-run
   (`fix_termination_failure_safety.md`). Fixed by never letting a
   termination failure exit the process — it idles instead, with an
   explicit log message, since idling (bounded, visible) is strictly better
   than an unbounded re-run loop.
7. **Generalized to every failure path, not just the one observed** — a
   follow-up smoke test proved the previous fix only covered the
   termination-call site; a pipeline-stage failure still exited normally
   and still triggered a restart. Refactored into one required choke
   point, `_fail_and_idle()`, that every failure path in the script must
   route through, confirmed via a line-by-line audit of every `sys.exit`/
   early `return`/uncaught `raise` in the file
   (`fix_universal_idle_on_failure.md`).

This sequence — five real, independently-discovered infrastructure bugs,
each found on a real pod, each fixed and tested before the next one
surfaced — is reported here as methodology, not filler: it is the same
"real hardware surfaces bugs mocks don't" pattern documented at length for
the defense modules in Section 7, just at the platform-orchestration layer
instead of the model-inference layer.

### 3.4 Cache and corpus integrity

- **Atomic writes.** FAISS index, pickle metadata, and JSONL result writes
  all use a temp-file-in-the-same-directory + `os.replace()` pattern
  (`cache_integrity_fix_task.md`) — `os.replace()`'s atomicity guarantee
  only holds within a single filesystem, so the temp file must never be
  written to `/tmp` or a different mount.
- **Two-file consistency, closed with a marker file, not file ordering.**
  A first fix reordered which of (index, metadata) was written last on the
  theory that made the crash-window harmless; a follow-up correctly
  identified this doesn't actually close the window, only relocates which
  file is stale on a crash — the real fix is a `.build_complete` marker
  file, written last, atomically, after both the index and metadata are
  already safely in place. The cache-validity check now checks for the
  marker's existence, not the index file's — a leftover index or metadata
  file from an interrupted build is no longer evidence of anything valid
  (`pr9_marker_file_followup.md`).
- **Corpus revisions are pinned**, the same way model revisions already
  were, closing a gap the cache-integrity audit itself surfaced: "does old
  cached metadata still match a freshly regenerated one" was conditional on
  the upstream HF dataset snapshot not changing, which was never guaranteed.
  The `.build_complete` marker now carries the revision hash it was built
  with, so a revision change in `config.py` automatically invalidates every
  stale cache without a manual `force_rebuild=True`
  (`corpus_revision_pinning_task.md`).
- **Explicitly flagged as still open:** `os.replace()`'s atomicity guarantee
  has only been verified on local/POSIX filesystems — it was never
  confirmed to hold the same way over the Azure Files SMB mount originally
  planned for shared scratch storage, since that infrastructure path was
  superseded by the RunPod migration before that verification step was
  reached (`cache_integrity_fix_task.md`, "What stays open").

---

## 4. The `nq_open` Exclusion — Why One of Three Corpora Never Appears in a Real Comparison

Found while preparing Phase 2's injection attack: the RAG pipeline's
"retrieved context" — the block of text every model actually read before
answering — was not real passage text for any of the three corpora. It was
the raw Python dict representation of the underlying dataset row, **gold
answer included** (`harness/pipeline.py`'s `build_rag_user_prompt()` fell
through to `f"[{i+1}] {doc}"` whenever `doc` had no `'text'` key, which none
of the three corpora's raw records do).

For `hotpot_qa` and `ms_marco`, this was a genuine, fixable bug — both
corpora have real supporting passage text available via
`data/normalize.py`'s existing `extract_passage_text()`, already used for
FAISS embedding but never reused for context rendering. Fixed on
`fix-rag-context-answer-leak`, verified by a regression test asserting no
gold-answer substring appears in rendered context for either corpus.

**For `nq_open`, this is not fixable at the pipeline level.** The corpus, as
constructed for this project, has no independent supporting passage at all
— `extract_passage_text()`'s `nq_open` branch concatenates the question and
the gold answer together as a stand-in "document," because there is nothing
else to substitute. After the fix, `nq_open`'s context still contains the
gold answer, just as a plain sentence instead of a dict repr — this is a
corpus-construction choice made before this project, not a bug reachable
from pipeline code.

**Evidence this matters, not just in principle:** recomputed from real
Phase 1 data (all 12 model×corpus cells, `phase1_results_complete/`), mean
F1(clean) across the 4 models is 0.963 (spread 0.005) on `nq_open` versus
0.585 (spread 0.179) on `hotpot_qa` and 0.255 (spread 0.042) on `ms_marco`
(`nq_open_leakage_finding.md`, Table). `nq_open` is both near-ceiling and
nearly invariant across four models of meaningfully different capability —
the signature of answer-copying, not of a task where retrieval and
reasoning quality matter.

**Consequence, applied consistently for the rest of the project:**
`nq_open` is excluded from every attack sweep (Phase 2), every defense
sweep (Phase 3), and every cross-corpus comparison anywhere in this
project — not just from the injection attack it was discovered while
preparing. Its Phase 1 numbers are reported once, in Section 9.1, labeled
as a documented limitation (closed-book knowledge recovery via a
retrieval-shaped scaffold, not genuine RAG), and are not a data point
comparable to anything else in this document.

---

## 5. Phase 2 — Attacks

### 5.1 Indirect Prompt Injection

**Design.** 4 models × 2 corpora (`hotpot_qa`, `ms_marco`) × 5 injection
templates, n=1000 per cell, 40 cells. Five strategies adapted from Liu,
Jia, Geng, Jia & Gong, *Formalizing and Benchmarking Prompt Injection
Attacks and Defenses* (USENIX Security 2024; reference implementation
`liu00222/Open-Prompt-Injection`), mapped onto a **goal-hijacking**
(redirect to a different question) vs. **process-hijacking** (make the
model believe it's at a different stage of its own task) taxonomy — this
mapping was a real design decision made for this project, not a lookup
from the source paper (`phase2_indirect_injection_task.md:24`). Templates:
`naive`, `escape_char`, `combined` (goal-hijack); `ignore`,
`fake_completion` (process-hijack). Injection point: organic placement —
appended to an organically retrieved document, not forced into top-k, to
keep this attack's surface distinct from PoisonedRAG's forced-retrieval
mechanism.

**Two corrections to the headline finding, both already applied to the
numbers below.** (1) The Phase 1 baseline used for utility-drop comparisons
was originally inflated 0.12–0.40 F1 points by the `nq_open`-style
dict-repr leak (Section 4) before it was fixed for `hotpot_qa`/`ms_marco`;
recomputing against the corrected baseline **retracted** the original claim
of a universal, compliance-independent F1 collapse. (2) `evaluation/metrics.py`'s
`normalize_text()` was deleting punctuation instead of replacing it with a
space, silently undercounting matches on answers like `"28-32"` vs
`"28 - 32"` — recomputed directly from stored raw JSONL, affecting 8.9% of
combined Phase 1 + this attack's rows, concentrated on `ms_marco`. Neither
correction changed any ASR number or significance verdict — ASR is a raw
substring check untouched by either fix.

**Headline finding: utility damage tracks compliance, not a separate
context-pollution effect.** On the three goal-hijack templates, where ASR
sits at 0.1–1.1% almost everywhere, the F1 "drop" against the corrected
baseline is tiny and inconsistent in sign (every one of 24 goal-hijack
cells in [−0.038, +0.025]). F1 damage is concentrated in exactly the cells
that also show elevated ASR — the two process-hijack templates, and within
those, specifically the models that show real compliance. One small,
genuine exception survives: `phi-4-mini`/`hotpot_qa`/`fake_completion`
shows a real F1 drop (0.024, 95% CI 0.010–0.037) alongside near-zero ASR
(0.2%) — the one surviving data point for a compliance-independent utility
cost, an order of magnitude smaller than originally reported.

**ASR by model × corpus × template (%):**

| model | corpus | naive | escape_char | ignore | fake_completion | combined |
|---|---|---|---|---|---|---|
| llama-3.1-8b | hotpot_qa | 0.20 | 0.20 | 0.00 | 0.00 | 0.20 |
| llama-3.1-8b | ms_marco | 0.40 | 0.40 | 0.30 | 0.60 | 0.40 |
| qwen3-8b | hotpot_qa | 0.20 | 0.40 | 0.30 | **11.70** | 0.10 |
| qwen3-8b | ms_marco | 0.40 | 0.90 | 3.20 | **14.10** | 0.50 |
| phi-4-mini | hotpot_qa | 0.20 | 0.30 | 0.00 | 0.20 | 0.20 |
| phi-4-mini | ms_marco | 0.30 | 0.10 | 3.20 | **10.10** | 0.10 |
| ministral-3-8b | hotpot_qa | 0.70 | 0.50 | **35.30** ⚠ | **60.80** ⚠ | 0.50 |
| ministral-3-8b | ms_marco | 0.80 | 0.70 | **30.50** ⚠ | **51.50** ⚠ | 1.10 |

⚠ **PROVISIONAL, STILL UNRESOLVED AS OF THIS COMPILATION.** Direct
inspection of all 607 of `ministral-3-8b`'s `marker+extra` rows across all
four `ignore`/`fake_completion` cells found **0 of 607 echo the injected
text and 100% of 607 are the model still attempting the real question** —
never once abandoning the real task. This looks far more like a mechanical
prepend/leak (plausibly tied to the `tokenize=False`
render-then-retokenize pattern shared by `harness/model_loader.py`'s
`build_chat_prompt`, though `ministral-3-8b` is the only one of the four
models on the distinct `mistral3` loader/tokenizer class, so a shared-code
explanation doesn't cleanly account for why only this model shows it at
scale) than genuine instruction compliance. The document that reported this
number explicitly states: **"Do not cite ministral-3-8b's 30.5–60.8%
figures as confirmed vulnerability"** until a chat-template round-trip
diagnostic (tokenize=False → re-tokenize → decode, compared byte-for-byte
against the original rendered prompt) has been run
(`PHASE2_INJECTION_INSIGHTS.md:1-13`). **This diagnostic was never run** —
confirmed by a repository-wide search for any round-trip/diagnostic
follow-up finding no resolution anywhere in this repository as of this
compilation. This is the one number in this entire consolidated document
that remains genuinely open; see Section 10.

**Mechanism, for the numbers that are not under this suspicion.**
`qwen3-8b` is vulnerable specifically to `fake_completion` (11.7–14.1%) but
not `ignore` (0.3–3.2%) — hypothesized (not proven) to relate to Qwen3's
explicit thinking-mode chat-template machinery being more susceptible to a
forged *completion* signal than a blunt instruction override.
`phi-4-mini` is vulnerable to `fake_completion` on `ms_marco` (10.1%) but
not `hotpot_qa` (0.2%) — Fisher's exact confirms this corpus split is real
(p = 8.18×10⁻²⁹), hypothesized to relate to `ms_marco`'s shorter,
less-clearly-bounded passage fragments giving a fake completion signal more
room to blend in. `llama-3.1-8b` is near-zero everywhere (0.0–0.6%), the
most robust of the four models against this attack.

**ASR scoring conflates three distinct phenomena** under one substring-match
number, confirmed by direct manual inspection of every model's flagged
rows: (1) verbatim echo of the injected text (not compliance — the model
regurgitates what it just read); (2) the marker glued onto an otherwise
normal, unaffected answer attempt (the pattern behind the ministral-3-8b
flag above); (3) explicit reasoning leading to a deliberate, genuinely
compliant choice (the rarest pattern, found twice for `phi-4-mini`). This is
reported as a real limitation of rule-based substring-match ASR scoring for
this class of attack, not a footnote specific to one model.

### 5.2 PoisonedRAG

**Design.** 4 models × 2 corpora × 1 poison configuration (`adv5`,
`ADV_PER_QUERY=5`), n=90/100 (`hotpot_qa`) and n=96/100 (`ms_marco`) —
Phase A's poisoned contexts are model-agnostic and shared across all 4
models' cells. Adapted from Zou et al., *PoisonedRAG: Knowledge Corruption
Attacks to Retrieval-Augmented Generation of Large Language Models* (USENIX
Security 2025; reference implementation `sleeepeer/PoisonedRAG`).

**Headline finding: corpus identity, not model identity, determines
success — the mirror image of the injection attack.** ASR sits at 70–78%
on `hotpot_qa` and 18–20% on `ms_marco` for every single one of the 4
models. All 12 model-vs-model McNemar comparisons are non-significant after
Holm-Bonferroni correction (smallest raw p=0.065, doesn't survive
correction). All 4 corpus-vs-corpus Fisher comparisons are significant at
p < 2×10⁻¹². Attack 1 showed model identity dominant and corpus identity
nearly irrelevant; PoisonedRAG shows the reverse — suggesting corpus-
poisoning resistance and injection resistance are genuinely independent
capabilities, not summarizable by one "robustness" score.

**Per-cell results:**

| model | corpus | n | ASR | echoed, not adopted | F1 baseline | F1 attack | utility drop (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | hotpot_qa | 90 | 73.3% | 2.2% | 0.640 | 0.142 | 0.498 (0.399–0.594) |
| qwen3-8b | hotpot_qa | 90 | 77.8% | 4.4% | 0.662 | 0.105 | 0.556 (0.463–0.649) |
| phi-4-mini | hotpot_qa | 90 | 70.0% | 10.0% | 0.392 | 0.091 | 0.301 (0.205–0.399) |
| ministral-3-8b | hotpot_qa | 90 | 76.7% | 6.7% | 0.679 | 0.127 | 0.552 (0.455–0.646) |
| llama-3.1-8b | ms_marco | 96 | 17.7% | 2.1% | 0.265 | 0.102 | 0.163 (0.101–0.230) |
| qwen3-8b | ms_marco | 96 | 19.8% | 2.1% | 0.288 | 0.121 | 0.167 (0.105–0.231) |
| phi-4-mini | ms_marco | 96 | 18.8% | 3.1% | 0.275 | 0.157 | 0.117 (0.052–0.182) |
| ministral-3-8b | ms_marco | 96 | 19.8% | 0.0% | 0.273 | 0.114 | 0.159 (0.097–0.225) |

**Why the corpus effect is this large — two contributing, not competing,
mechanisms, both measured directly.** (1) Retrieval precision (fraction of
top-5 retrieved documents that are the planted poison) is 99.3% on
`hotpot_qa` vs 88.96% on `ms_marco` — on `ms_marco`, real content still
competes for roughly 1 in 9 retrieved slots. (2) Mean question length is
18.17 words on `hotpot_qa` vs 6.14 words on `ms_marco` — `hotpot_qa`'s
long, multi-clause, multi-entity questions give a crafted ~100-word poison
passage far more surface to weave a persuasive fabricated answer into than
a short, generic web-query-style question does. Both mechanisms point the
same direction; this document does not claim to know their relative weight.

**Skip/loss rate: 14/1,472 target questions (10 `hotpot_qa`, 4 `ms_marco`),
verified directly against the live sweep log, not reconstructed from
memory.** Root causes, by actual final error: 9/14 truncation (nemotron's
internal reasoning competing with output for token budget), 3/14
API/infrastructure failures (503s, one timeout — unrelated to content), 1/14
genuine content-policy refusal, 1/14 a since-fixed parser formatting gap.
8 of 111 total attempt-level failures across the sweep showed explicit
refusal-style text; only 1 escalated to a permanent skip (7/8
retry-recovered) — and the refusal trigger is not cleanly "avoid real named
entities": 2 of the 8 refusals concerned non-entity content (a plant
taxonomy question, a database-configuration how-to).

**Poison-generator selection, a real methodological finding.** Three
generator/transport combinations were tried and abandoned for concrete,
verified engineering reasons before settling on
`nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM: `kimi-k3` (measured
480s for a trivial reply — not viable at sweep scale), Kimi-K2 via
HuggingFace's Inference API (worked, but HF's "free" tier turned out to be
a $0.10/month cap in practice). A fourth (`minimax/minimax-m3:free` via
OpenRouter) was verified as a working, tested backup and documented but not
used.

### 5.3 Crescendo

**Design.** 4 target models × 1 multi-turn escalation strategy × 5 turns,
n=100 behaviors per model, seeded stratified sample from a pooled 100
JBB-Behaviors + 400 HarmBench (500 total) behavior set. Adapted from
Russinovich, Salem & Eldan, *Great, Now Write an Article About That: The
Crescendo Multi-Turn LLM Jailbreak Attack* (2024; reference implementation
Microsoft PyRIT, `Azure/PyRIT`). **DeepSeek V4 Pro is the attacker and judge
in every conversation, never a target** — this role asymmetry must be read
into every result below: "which model resists Crescendo" describes only
the four `config.MODELS` targets, never DeepSeek, whose high compliance as
attacker (verified pre-flight, 10/10 real 5-turn escalation calls, no
refusal) is expected generator behavior, not a susceptibility finding.

**Headline finding: outcome looks like PoisonedRAG (no significant model
ranking), but the *mechanism* is a strong per-model signature like the
injection attack — a third, distinct pattern.** ASR ranges 59.2–70.2% across
the 4 models; none of the 6 pairwise McNemar comparisons survive
Holm-Bonferroni correction (closest: llama-3.1-8b vs phi-4-mini, raw
p=0.0428, needs p≤0.0083 to survive). But the refusal/backtrack behavior
underlying that outcome is wildly different per model: `llama-3.1-8b`
refuses constantly (0.91 refusals/conversation, highest) and backtracks
constantly (44% of behaviors) — yet still lands at the single highest ASR
(70.2%), because backtracking reliably breaks through. `ministral-3-8b`
almost never refuses at all (0.02 refusals/conversation, 2% backtrack rate)
and lands at essentially the same ASR (69.5%) — it doesn't need to be
talked out of resistance because there's barely any resistance to begin
with. `qwen3-8b` and `phi-4-mini` sit in between on refusal rate but
diverge on outcome: `qwen3-8b`'s backtracks convert to success about as
readily as `llama-3.1-8b`'s (68.4% ASR); `phi-4-mini`'s same amount of
resistance more often *holds* (59.2% ASR, lowest of the four; bootstrap 95%
CI on its gap to `llama-3.1-8b` is 0.129 [0.011, 0.237], excluding zero even
though the Holm-corrected McNemar test on the same pair does not reach
significance).

**Per-model results:**

| model | n | n_scored | ASR | backtrack rate | mean backtracks | mean refusals |
|---|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b | 100 | 94 | 70.21% | 44.0% | 0.90 | 0.91 |
| ministral-3-8b | 100 | 95 | 69.47% | 2.0% | 0.02 | 0.02 |
| qwen3-8b | 100 | 95 | 68.42% | 23.0% | 0.28 | 0.28 |
| phi-4-mini | 100 | 98 | 59.18% | 23.0% | 0.29 | 0.29 |

**`is_refusal()` is a substring scan with a confirmed false-positive mode.**
Both of `ministral-3-8b`'s two backtrack rows in the entire sweep were
individually audited and are false positives — the target's own reply was
already substantively compliant in both cases; the match landed inside
fabricated dialogue the model itself generated, not the model's own
refusal. `refusal_count`/`backtrack_rate` should therefore be read as an
upper bound on genuine refusal, confirmed for `ministral-3-8b` specifically
(the only model where an exhaustive per-row audit was tractable — 2 rows);
the other three models' 23–44 flagged rows each were not individually
audited.

**Engineering findings from this attack's build, reported as methodology:**
five real, verified generator/transport obstacles across two model-
selection rounds and two transport rounds (Kimi-K2/kimi-k3/Nemotron all
"looked fine small and broke at scale" via reasoning-token exhaustion at
real 5-turn scale; DeepSeek via NIM verified clean at scale; a mid-sweep
NIM account-level rate-limit exhaustion forced a config-driven switch to
OpenRouter; OpenRouter then independently surfaced its own reasoning-leak
failure under harder adversarial content, fixed via `reasoning.enabled=false`).
A separate durability-vs-atomicity engineering lesson: the raw JSONL writer
originally used the same whole-file atomic-write pattern as the other two
attacks, which gives atomicity at the cost of durability — an overnight
sweep interruption lost an entire model's 100 completed behaviors because
the file under its real name never existed until the whole loop finished.
Fixed by trading atomicity for `flush()`+`fsync()` after every row, so a
future interruption only costs behaviors not yet completed.

### 5.4 Three attacks, three different dominant axes — direct comparison

| | Injection | PoisonedRAG | Crescendo |
|---|---|---|---|
| Dominant outcome factor | Model identity | Corpus identity | Neither — no model pair survives correction |
| Model-vs-model significance | Real, per-template | All 12 comparisons non-significant | All 6 comparisons non-significant |
| Per-model mechanism differences | Real (goal-hijack flat everywhere; process-hijack model-specific) | None found — models converge via the same corpus-driven mechanism | Real and large (refuse-then-bypass vs. barely-refuse vs. refuse-and-hold) |

This is a real, load-bearing finding for how Phase 3 defenses should be
evaluated (Section 6): a defense judged only on aggregate ASR reduction
against Crescendo would treat all 4 models as an equally good or bad
starting point, when in fact `llama-3.1-8b` and `ministral-3-8b` need
opposite interventions to improve — one needs its refusals to survive
rephrasing, the other needs to refuse in the first place.

---

## 6. Phase 3 — Defenses

All numbers in this section are the **final, backend-corrected** figures —
see Section 8 for the full backend-confound investigation and resolution
that produced them. The real cell inventory is **79 cells** (51 injection +
24 PoisonedRAG + 4 Crescendo/`output_filter`, diagnostic-only), not the 52
originally assumed — injection has 3× more real cells because coverage is
asymmetric across the three defenses (`instruction_detection` and
`spotlighting` each cover 16 real cells — 2 templates × 2 corpora × 4
models; `output_filter` covers 19 — the same 16 plus 3 extra templates that
completed only for one model×corpus). Six additional files at exactly n=3
are confirmed infrastructure smoke-test fragments, excluded from every
table below.

### 6.1 `instruction_detection`

**Design.** `protectai/deberta-v3-base-prompt-injection-v2` run against
each retrieved passage before context assembly; flagged passages are
dropped (partial context loss measured, not a blanket refusal). Capped at
n=40 per cell (not the full n=1000) because a real-hardware timing
measurement found a single item took 4m27s wall-clock even after fixing a
CPU thread-oversubscription bug (Section 7) — the full n=1000 sweep was
impractical within this project's compute budget.

**Final, backend-corrected result: 0 of 16 cells significant** (down from
3/16 under the original, confound-contaminated `vllm`-baseline comparator
— all 3 originally-significant cells were `ministral-3-8b`, and all 3 are
confirmed `backend_confound`, the same pattern as `output_filter`, not
`spotlighting`; see Section 8).

**Mechanism attribution, backend-corrected.** Of items genuinely blocked by
the classifier (defined against the backend-matched baseline, not `vllm`):
**10 of the original 55 "blocked" items survive the backend correction, and
all 10 (100%) have at least one passage actually flagged** — the other 45
were never really blocked by the classifier at all; the model's answer had
simply changed between `vllm` and the undefended `hf` backend, independent
of any defense. The confound inflated the denominator roughly 4.5×; it did
not obscure a real classifier failure. Every genuine block in this dataset
has the classifier's fingerprint behind it.

### 6.2 `spotlighting`

**Design.** Base64-encodes the retrieved context, defeating literal-text
injection templates that depend on the model parsing natural-language
instructions inside that context. Full n=1000 per cell (no per-item
classifier cost).

**Final, backend-corrected result: 9 of 16 cells still significant** — the
**only** defense in this project with a confirmed, backend-isolated real
effect on injection ASR:

| Model | Corpus | Template | hf-nodef ASR | hf-def ASR | Reduction | p (Holm) |
|---|---|---|---:|---:|---:|---:|
| qwen3-8b | hotpot_qa | fake_completion | 0.117 | 0.000 | 0.117 | 1.57e-34 |
| qwen3-8b | ms_marco | ignore | 0.034 | 0.000 | 0.034 | 1.16e-09 |
| qwen3-8b | ms_marco | fake_completion | 0.140 | 0.000 | 0.140 | 2.01e-41 |
| phi-4-mini | ms_marco | ignore | 0.016 | 0.000 | 0.016 | 2.44e-04 |
| phi-4-mini | ms_marco | fake_completion | 0.027 | 0.000 | 0.027 | 1.34e-07 |
| ministral-3-8b | hotpot_qa | ignore | 0.052 | 0.000 | 0.052 | 4.89e-15 |
| ministral-3-8b | hotpot_qa | fake_completion | 0.085 | 0.000 | 0.085 | 6.20e-25 |
| ministral-3-8b | ms_marco | ignore | 0.143 | 0.000 | 0.143 | 2.87e-42 |
| ministral-3-8b | ms_marco | fake_completion | 0.143 | 0.000 | 0.143 | 2.87e-42 |

The backend switch still contributes substantially to 6 of these 9 cells
(all `ministral-3-8b` and `phi-4-mini`/`ms_marco` — 50–86% of the
*original* `vllm`-vs-defended reduction, `partial_split` verdict) — so the
uncorrected headline "100% ASR reduction" overstated `spotlighting`'s own
unique contribution for those specific cells. Only `qwen3-8b`'s 3
significant cells show essentially zero backend contribution
(`backend_frac` ≈ 0) — a real, model-specific, backend-independent effect.

**But the utility cost is severe and the headline "100% ASR reduction"
alone hides it.** 12 of 16 cells hit exactly 0% defended ASR, but mean F1
drops **−0.283** vs. the Phase 1 no-attack baseline (as low as −0.44 in
some cells) — the model frequently cannot answer the question at all once
its context is base64-encoded.

### 6.3 `output_filter`

**Design.** `meta-llama/Llama-Guard-4-12B` run post-generation on the
model's final response; flagged output is replaced with a fixed
`REFUSAL_MARKER` (never generated text) so scoring can distinguish
"defense fired" from "model refused on its own."

**Final, backend-corrected result against injection: 0 of 19 cells
significant** (down from 6/19 under the original comparator). Direct
mechanism evidence for why: only **12 guard-catches out of 1,511 blocked
items (0.8%)** — and the cells with the largest, most significant
*uncorrected* ASR reductions (`ministral-3-8b`: 303–525 blocked items each)
have the *lowest* guard-catch fractions (0.2–1.1%), the exact opposite of
what "the guard is doing the defending" would predict. The confound
explains this directly: nearly all of the apparent defense effect was the
backend switch, not guard action.

**Against PoisonedRAG: a structural architectural mismatch, not a matter of
degree.** The guard's flag rate is **exactly 0.0000 across all 160 scored
PoisonedRAG responses**, every model, every corpus. A safety-content
classifier structurally cannot see factual poisoning — a poisoned answer
isn't unsafe, it's just wrong, and that is orthogonal to what a Llama-Guard-
style classifier screens for. Every apparent ASR change in `output_filter`'s
PoisonedRAG cells is attributable to something other than the guard.

**Against Crescendo: not a defended-condition measurement at all, and
correctly excluded from every ASR-reduction table.** By design
(`run_crescendo.py:184-257`), the guard's verdict is logged but never acted
on — the judge always scores the real, unfiltered conversation, so there is
no live intervention to measure a reduction from. What was measured
instead, purely diagnostically: of conversations that succeeded
unfiltered, **67–82% had at least one turn the guard would have flagged**
before the conversation concluded (`llama-3.1-8b` 72.7%, `qwen3-8b` 81.8%,
`phi-4-mini` 66.7%, `ministral-3-8b` 73.3%) — a genuinely encouraging
number for a *future, actually-wired* version of this defense, since
Crescendo's attack surface (multi-turn escalation toward unsafe content) is
exactly what a safety classifier is built to catch, unlike PoisonedRAG's.
This remains a well-motivated, unexplored next step, not a hidden gap.

### 6.4 Utility preservation — mean ΔF1 by (attack, defense)

Averaged across the 8 model×corpus cells, Phase 3 defended F1 vs. the
matching Phase 1 no-attack, no-defense baseline:

| Attack | Defense | Mean ΔF1 |
|---|---|---:|
| injection | instruction_detection | −0.0014 (negligible) |
| injection | output_filter | −0.0161 (small) |
| injection | spotlighting | **−0.2831 (severe)** |
| poisonedrag | instruction_detection | −0.3011 (severe) |
| poisonedrag | output_filter | −0.2873 (severe) |
| poisonedrag | spotlighting | −0.3067 (severe) |

For injection, `instruction_detection` is essentially free on utility (it
only drops passages an off-the-shelf classifier actually flags, leaving
most legitimate context intact) while `spotlighting`'s encoding devastates
it. For PoisonedRAG, all three defenses cost roughly the same large utility
hit — none of them repair the underlying retrieval corruption; they only
change what the model does with an already-poisoned context (refuse,
hedge, or fail to extract an answer), which reads as "wrong answer" either
way against gold-answer F1.

### 6.5 Verdict summary

**9 of 51 real injection cells significant** (0 `output_filter` + 9
`spotlighting` + 0 `instruction_detection`), all backend-corrected. **4 of
24 PoisonedRAG cells significant**, all `spotlighting` — needing no
correction, since PoisonedRAG was never backend-confounded (Section 8).
Crescendo/`output_filter`'s 4 cells are diagnostic-only and are not members
of either significance family.

---

## 7. The Real-Hardware Engineering Story

Eight distinct, genuinely blocking bugs were found in the two production
defense modules — six in `defenses/output_filter.py`, two in
`defenses/instruction_detection.py` — every one of them **invisible to this
project's own mocked test suite**, and every one found only by running the
real code against real model weights on the real RunPod pod. Chronological
order, from git history (`git log -- defenses/output_filter.py
defenses/instruction_detection.py`):

**1. Chat-template alternation crash (`output_filter`, commit `9dec207`,
2026-09-12).** `classify_response()` built a single-turn conversation as
`[{"role": "assistant", "content": response_text}]` — a bare string,
single "assistant" turn. Llama-Guard-4-12B's own `chat_template.jinja`
unconditionally enforces `messages[0]["role"] == "user"` and requires
content to be a list of typed `{"type": "text", ...}` dicts, not a bare
string. This failed deterministically on every call — not something
specific to the response text — raising
`jinja2.exceptions.TemplateError("Conversation roles must alternate...")`
the first time this code path ever ran against the real tokenizer.
**Why the mocks missed it:** the test suite's stub tokenizer's
`apply_chat_template` just echoed the conversation back via a plain
assertion — it never executed real Jinja alternation logic at all. Fixed
by matching the model card's own "Getting Started" example exactly;
the regression test added afterward renders the verbatim template snippet
through an *actual* `jinja2.Environment`, not an echoing stub — proving
this test would have caught the original bug, which no prior test could.

**2. `BatchEncoding` vs. tensor crash (`output_filter`, commit `5d036a0`,
2026-09-12).** `apply_chat_template`'s `return_dict` default flipped to
`True` in the installed `transformers` (5.7.0), so `tokenize=True +
return_tensors="pt"` alone returns a `BatchEncoding` (dict-like), not a
bare tensor. The old code passed that dict straight into
`model.generate(input_ids=<dict>, ...)`, crashing on real hardware with an
`AttributeError` on `.shape` deep inside `generate()`. **Why the mocks
missed it:** both existing test stubs returned a bare tensor, silently
masking the exact shape mismatch that broke on real hardware. Fixed by
requesting `return_dict=True` explicitly and unpacking via `**inputs`;
both stubs were updated to return a real `transformers.BatchEncoding`
instead of a hand-rolled lookalike.

**3–4–6. The Llama4 chunked-attention/cache-construction saga — three
separate real-pod crashes at three different call sites, same underlying
config mismatch (`output_filter`, commits `31081cf`, `4dd974c`, `a26fc372`,
2026-09-12/13).** Root cause, confirmed by reading the installed
`transformers` source directly: Llama-Guard-4-12B's published `config.json`
sets `attention_chunk_size=None` (a deliberate Llama4 "Scout" long-context
design choice, not a Meta oversight), but `Llama4TextConfig.__post_init__`
computes `layer_types=["chunked_attention", ...]` for every layer
independent of that field — leaving every Cache class and the model's own
`forward()` mask-building code to crash on the `None` value at three
distinct points:
  - **Attempt 1** (`31081cf`): `StaticCache` crashed on
    `min(sliding_window, max_cache_len)` with `sliding_window=None`.
    `cache_implementation="dynamic_full"` fixed it in the local dev
    venv's `transformers==5.7.0` — but the floating `transformers>=5.13`
    pin in `requirements.txt` meant the real Docker build installed a
    newer version (5.17.0) where `"dynamic_full"` no longer existed at
    all, so this fix passed locally and failed on the real pod.
  - **Attempt 2** (`4dd974c`): replaced the version-fragile
    `cache_implementation` string entirely with a direct fix — relabel
    every `"chunked_attention"` layer to `"full_attention"` on a copy of
    the config and build a `DynamicCache` from that copy directly, a
    mechanism that doesn't depend on any particular enum surviving a
    version bump.
  - **Attempt 3** (`a26fc372`, "the 6th attempt on this same architectural
    issue"): a *different* real-pod crash, this time from inside
    `Llama4TextModel.forward()` itself, not Cache construction — because
    Attempt 2's fix patched only a *copy* of the config used to build the
    Cache, never the model's own live `model.config`, which `forward()`
    reads independently on every call. Fixed by patching the model's real
    config exactly once, right after load (confirmed via the source that
    `self.config` is a shared reference through the whole submodule tree,
    never copied) — and confirming a second, unconditional mask-building
    path (`create_chunked_causal_mask`) also needed
    `attention_chunk_size` set to a real int, not just relabeled
    `layer_types`, or the crash persists regardless of layer type.

**5. `generation_config`'s baked-in `cache_implementation` conflict
(`output_filter`, commit `9c610e0`, 2026-09-13).** A fourth real-pod crash,
after Attempt 2 above: `ValueError: Passing both cache_implementation ...
and past_key_values ... is unsupported`. Root cause, confirmed via
`transformers.generation.utils` source: the guard model's own
`generation_config.json` bakes in `cache_implementation="static"`, and
`generate()`'s config-preparation step seeds its working config from that
model default *before* the module's own kwargs are applied — so passing
`past_key_values=` alone was never enough to override it. Fixed by
explicitly passing `cache_implementation=None`, confirmed via
`GenerationConfig.update()`'s own source to be the only way that actually
clears a model-baked default from Python.

**7. Silent truncation no-op, near-hang (`instruction_detection`, commit
`9fff0c2`, 2026-09-13).** A real n=1000 sweep against `llama-3.1-8b` +
`hotpot_qa` appeared to hang — 50+ minutes, zero log output, CPU pinned at
100%, confirmed genuinely running (state R, not I/O-blocked). Root cause,
confirmed via the model's own published files, no GPU needed:
`protectai/deberta-v3-base-prompt-injection-v2`'s tokenizer sets
`model_max_length` to HF's "no limit configured" sentinel (~1e30), and
`transformers`' own tokenization internals silently downgrade
`truncation=True` to no-op whenever `max_length` is omitted and
`model_max_length` exceeds that sentinel — so `truncation=True` alone,
with no explicit `max_length`, did nothing. `hotpot_qa`'s real passages
(concatenating an entire ~10-document distractor context per record) ran
through DeBERTa's O(n²) self-attention at full, unbounded length on CPU.
**Why the mocks missed it:** the module's own smoke-test fixtures were
short strings that never exercised the real classifier or real corpus text
at all, by design. Fixed by passing `max_length=512` explicitly alongside
`truncation=True`.

**8. Uncapped CPU thread pool, 12.6× user/real time ratio
(`instruction_detection`, commit `c4ab250`, 2026-09-13).** After fix 7,
a single `RAG_SAMPLE_N=1` question still took 4m27s wall-clock but 56m5s
of aggregate CPU time. Confirmed by elimination, not guessed: exactly 5
`detect_injection` calls per question (one per retrieved passage, `TOP_K=5`
— no hidden loop), the classifier model correctly loaded once and reused
(the singleton pattern worked as designed), and a `cProfile` run on a real
hotpot_qa-shaped passage confirmed 100% of wall-time was genuine DeBERTa
forward-pass compute, not a stray loop or reinitialization. Root cause:
`torch.set_num_threads()` was never capped anywhere in this module or its
callers (confirmed via a repo-wide grep for zero hits) — on a
high-vCPU-count or cgroup-quota-mismatched cloud host, PyTorch's default
uncapped intra-op thread pool spawns and synchronizes far more threads
than a ~184M-parameter model's small forward pass can actually use, the
textbook cause of exactly this symptom. Fixed with
`torch.set_num_threads(1)`, called once before the classifier pipeline is
constructed.

**What this reveals about testing methodology for this class of system.**
Every one of these 8 bugs passed the existing test suite cleanly before it
was ever found on real hardware, and in most cases the reason is
structurally the same: the test suite's stubs and fixtures were built to
exercise the *shape* of the code path (does the function get called, with
roughly the right arguments) without ever running the *real* tokenizer,
*real* chat template, *real* model config, or *real* corpus text through
it — precisely the layer where every one of these 8 bugs actually lived.
This is reported here as a citable methodological finding about testing
LLM-adjacent infrastructure, not as incidental engineering trivia: **a
green mocked test suite is evidence that the code's control flow is
correct, not evidence that it will run correctly against the real
artifacts (weights, tokenizers, chat templates, hardware) it was written
for.** A defense evaluation run entirely on results produced before these
fixes landed would not have failed loudly — several of these bugs (the
truncation no-op, the thread-pool oversubscription) would have produced a
*slow* run rather than a *wrong* one, but the Llama4 cache crashes and the
chat-template/BatchEncoding bugs would have prevented `output_filter` from
producing any real defended-condition data at all, silently invalidating
every number in Section 6.3 had they not been caught before the real
sweep.

---

## 8. The Backend-Confound Discovery

### 8.1 How it was found

Phase 2's injection attack baseline was generated on the `vllm` inference
engine (`phase2_injection_results/attack_raw_*_vllm.jsonl`); every Phase 3
defended run — for every one of the three defenses — ran on the `hf`
(`transformers`) engine instead. Same model weights, different inference
stack. This was first flagged as a methodology footnote, then strengthened
from "possible" to "likely" once `output_filter`'s guard-catch fraction
came back at 0.8% (Section 6.3) — a number too low to explain the large ASR
reductions the master table was reporting, which raised the direct
question of what *else* could be producing them.

### 8.2 What it affected, resolved with a real third condition rather than
left as inference

Rather than continue treating this as a suspected confound, a genuine
third condition was built and measured directly: a backend-matched,
no-defense `hf` baseline — same items, same `RAG_DEFENSE=none`, verified
item-for-item identical to each defense's real item set (`scripts/
verify_phase3_backend_baseline_items.py` for `output_filter`/`spotlighting`
at n=1000; `scripts/verify_phase3_instruction_detection_baseline_items.py`
for `instruction_detection` at n=40, verified as an exact 40-row prefix of
the same n=1000 file — meaning **no new GPU sweep was needed** for this
defense's correction at all). This gives three points per cell: `vllm`
baseline → `hf` no-defense baseline → `hf` defended. Pairwise McNemar with
Holm-Bonferroni, applied within each cell's own 3-comparison family,
attributes each cell's total ASR reduction to one of four verdicts:
`backend_confound` (backend switch alone explains it), `defense_works`
(backend switch changes nothing, defense does all the work),
`partial_split` (both pairwise gaps significant — both contribute), or
`inconclusive` (neither gap reached significance, usually a near-zero-ASR
cell with no room to show an effect).

- **`output_filter`/injection: CONFIRMED, fully explained by the backend
  switch.** Every one of its 6 non-inconclusive cells is `backend_confound`
  — `hf`-no-defense-vs-defended never reaches significance (p_holm = 1.0 in
  5/6, 0.06 in the sixth), with 76–100% of the total reduction attributable
  to the backend switch alone. 0/19 cells remain significant once
  corrected (Section 6.3).
- **`spotlighting`/injection: PARTIALLY confirmed — a real effect survives.**
  All 9 of its non-inconclusive cells have a significant
  `hf`-no-defense-vs-defended gap — the defense genuinely does something —
  but 6 of those 9 are `partial_split`: the backend switch already explains
  50–86% of the *original* reduction before `spotlighting`'s own base64
  encoding contributes anything further. 9/16 cells remain significant
  (Section 6.2), with meaningfully smaller effect sizes than the
  uncorrected numbers implied.
- **`instruction_detection`/injection: CONFIRMED, same pattern as
  `output_filter`.** Its only 3 non-inconclusive cells (all `ministral-3-8b`)
  are all `backend_confound`, 85–90% of the reduction attributable to the
  backend switch. 0/16 cells remain significant once corrected (Section
  6.1).
- **PoisonedRAG and Crescendo: CONFIRMED never confounded, resolved by git
  history, not a new sweep.** `git log --all --diff-filter=A --name-only`
  on `phase2_poisonedrag_results/` and `phase2_crescendo_results/` shows
  every file ever committed to either directory was generated with
  `INFERENCE_ENGINE=hf` — never `vllm` — because
  `evaluation/result_paths.py`'s file-naming functions bake the actual
  engine used directly into the committed filename, making this a direct
  read of history, not an inference. Both attacks' Phase 2 baselines
  already ran on the same backend as their Phase 3 defended runs, from the
  start.

### 8.3 What this means methodologically for anyone doing before/after
comparisons across inference backends

Two independent inference stacks that load the *same pinned model weights*
are not guaranteed to produce the *same generation behavior* — `hf` and
`vllm` differ enough in this project's own data to fully explain a
0.302–0.524 absolute ASR reduction on their own, with zero defense
involved. A before/after comparison that changes the inference backend
between the "before" and "after" conditions cannot distinguish a real
intervention effect from a backend artifact, no matter how large or
statistically significant the raw difference looks — the significance test
was answering "did something change," not "did the thing I changed cause
it." The general lesson, stated for reuse beyond this project: **when
introducing a new stage into a pipeline (a defense, a filter, any
intervention), verify the inference backend is held constant across the
comparison before trusting the result, and if it wasn't, build the missing
matched condition rather than argue from indirect evidence** — exactly what
this section did, and what let two of three defenses' headline numbers be
corrected before being written into this document rather than after.

---

## 9. Master Results Tables

### 9.1 Phase 1 — Baseline (all 4 models × 3 corpora, `vllm`, n=1000 each)

| model | corpus | f1_clean | exact_match | contains_answer (diagnostic) |
|---|---|---:|---:|---:|
| llama-3.1-8b | hotpot_qa | 0.6058 | 0.531 | 0.562 |
| llama-3.1-8b | ms_marco | 0.2340 | 0.092 | 0.108 |
| llama-3.1-8b | nq_open † | 0.9608 | 0.909 | 0.989 |
| ministral-3-8b | hotpot_qa | 0.6392 | 0.560 | 0.601 |
| ministral-3-8b | ms_marco | 0.2572 | 0.088 | 0.123 |
| ministral-3-8b | nq_open † | 0.9599 | 0.910 | 0.993 |
| phi-4-mini | hotpot_qa | 0.4600 | 0.371 | 0.471 |
| phi-4-mini | ms_marco | 0.2539 | 0.074 | 0.146 |
| phi-4-mini | nq_open † | 0.9649 | 0.923 | 0.984 |
| qwen3-8b | hotpot_qa | 0.6363 | 0.557 | 0.599 |
| qwen3-8b | ms_marco | 0.2755 | 0.112 | 0.166 |
| qwen3-8b | nq_open † | 0.9623 | 0.921 | 0.991 |

† `nq_open` rows: documented limitation, not a valid RAG baseline — see
Section 4. Not used in any comparison anywhere else in this document.
(Source: `phase1_results_complete/baseline_summary_{model}_{corpus}_vllm.csv`.)

### 9.2 Phase 2 — Attack ASR summaries

Injection: Section 5.1's table. PoisonedRAG: Section 5.2's table.
Crescendo: Section 5.3's table. (Not repeated here — see those sections for
the complete, final per-cell numbers; reproducing them a third time would
be pure duplication, not new information.)

### 9.3 Phase 3 — Defense results (backend-corrected, final)

`instruction_detection`: 0/16 significant (Section 6.1). `spotlighting`:
9/16 significant, full per-cell table in Section 6.2. `output_filter`:
0/19 significant against injection, structural 0.0000 flag rate against
PoisonedRAG (Section 6.3). PoisonedRAG master verdict: 4/24 significant,
all `spotlighting` (Section 6.5). Utility preservation: Section 6.4.

### 9.4 Mechanism attribution table

| Defense | Attack | Mechanism metric | Result |
|---|---|---|---|
| output_filter | injection | guard-caught fraction of blocked items | 12/1,511 (0.8%) |
| output_filter | poisonedrag | guard flag rate, all scored responses | 0/160 (0.0000) |
| output_filter | crescendo | frac. of unfiltered successes guard would've flagged | 67–82% (counterfactual only) |
| instruction_detection | injection | frac. of backend-corrected blocks with ≥1 passage flagged | 10/10 (100%) |

---

## 10. Limitations

**Consolidated across all three phases — every item below is documented
somewhere in this repository's source material, not newly asserted here.**

- **`nq_open` is excluded from every attack, defense, and cross-corpus
  comparison in this project**, by construction (Section 4) — its Phase 1
  numbers describe closed-book answer-copying via a retrieval-shaped
  scaffold, not genuine RAG, and are reported once, labeled, in Section 9.1
  only.
- **`ministral-3-8b`'s injection ASR figures (`ignore`/`fake_completion`)
  remain genuinely open, unresolved as of this compilation.** This is the
  one number in this entire document not brought to a confirmed or closed
  state — see Section 5.1's full explanation and the flag there. Do not
  cite the 30.5–60.8% figures as confirmed vulnerability without first
  running the chat-template round-trip diagnostic that was specified but
  never executed.
- **The n asymmetry within injection (40 vs. 1000).**
  `instruction_detection` runs a real transformer classifier per retrieved
  passage on CPU; real-hardware timing made the full n=1000 sweep
  impractical, capping it at n=40 (Section 6.1). `spotlighting` and
  `output_filter` have no comparable per-item cost and ran the full n=1000.
- **PoisonedRAG's matched-subset reduction.** Phase 2's baseline sampled
  n=90–96 per model×corpus; Phase 3's defended cells drew a smaller n=15–20
  subset, with 1–3 genuinely new items per cell (absent from the Phase 2
  baseline) excluded from each cell's paired test. Every cell still had a
  non-empty matched subset (n=14–18); no cell required the unpaired
  Fisher's-exact fallback.
- **Crescendo's Phase 3 behavior-pool mismatch.** Only 9 of Phase 3's 20
  behaviors per model appear in Phase 2's n=100 baseline (same 9 matched,
  same 11 missing, consistent across all 4 models) — usable paired n for
  every Crescendo cell is 9, not 20, materially widening confidence
  intervals on the diagnostic-only numbers reported for this cell.
- **HarmBench dataset source.** Two HarmBench variants exist on the HF Hub:
  `walledai/HarmBench` (gated, the one actually used — access requested and
  granted same-day) and `AlignmentResearch/HarmBench` (ungated), documented
  in `config.py` as an unused fallback, never exercised for any real sweep
  (`phase2_crescendo_task.md:10`). The originally-assumed HarmBench pool
  size (510) was wrong and was corrected to the real, live-verified total
  of 400 (`standard` 200 + `contextual` 100 + `copyright` 100) before any
  real sweep ran.
- **ASR scoring is rule-based substring matching for all three attacks**,
  confirmed (Section 5.1, 5.3) to conflate genuinely different phenomena
  under one binary flag — verbatim echo, marker-glued-to-real-answer, and
  genuine reasoned compliance for injection; a substring refusal-phrase
  scan with a confirmed false-positive mode for Crescendo's backtrack
  trigger.
- **`is_refusal()`'s false-positive rate is confirmed for `ministral-3-8b`
  only** (both of its 2 backtrack rows individually audited) — plausibly
  affects some unquantified fraction of the other 3 models' 23–44 flagged
  rows each, not individually audited for this project.
- **One attacker/judge model for Crescendo** (DeepSeek V4 Pro) and one
  behavior-pool composition — this project cannot distinguish "these 4
  target models resist Crescendo-style escalation similarly" from "this
  specific attacker model's escalation strategy happens to land similarly
  on these 4 targets."
- **One poison configuration** (`ADV_PER_QUERY=5`) and one poison-generator
  model for PoisonedRAG — cannot distinguish "PoisonedRAG doesn't work well
  on `ms_marco`" from "this specific generator's poison passages don't work
  well on `ms_marco`."
- **`os.replace()`'s atomic-write guarantee was verified only on local/POSIX
  filesystems**, never confirmed against the Azure Files SMB mount
  originally planned for shared scratch storage — that infrastructure path
  was superseded by the RunPod migration before this verification step was
  reached (Section 3.4).
- **No Fisher's-exact fallback was ever exercised in this dataset** — every
  one of the 79 Phase 3 cells had a non-empty matched-item baseline
  subset, so the unpaired fallback path exists in code but is untested
  against real project data.

---

## 11. Open Items / Remaining Work

Kept brief and factual, per what this repository's material actually
documents as outstanding — this section does not speculate about anything
beyond what is directly stated in source material.

- **The `ministral-3-8b` injection ASR round-trip diagnostic** (Section
  5.1, Section 10) has not been run. This is the single highest-priority
  open item for finalizing Phase 2's injection findings, since it
  determines whether `ministral-3-8b`'s largest reported vulnerability
  (up to 60.8% ASR) is real or a formatting artifact.
- **`output_filter`-for-Crescendo is not currently wired to enforce
  anything live** (Section 6.3) — converting its 67–82% counterfactual
  catch rate into a real, measured ASR reduction (substituting
  `REFUSAL_MARKER` into the live conversation history, the same way
  injection's and PoisonedRAG's runners already do) is a well-motivated,
  unexplored next step, not yet started.
- **A Phase 4 (EU AI Act Article 15 mapping) is referenced only by
  implication** — `README.md` states the project's motivation in terms of
  Article 15 (Section 2), but no dedicated Phase 4 planning, mapping, or
  status document was found anywhere in this repository as of this
  compilation. `PHASE2_ROADMAP.md`, referenced by name in multiple task
  briefs (e.g. `phase2_poisonedrag_task.md:25`) as the source for
  poison-count/dataset-size decisions "confirmed in Meeting 3," **does not
  exist in this repository** (confirmed via repo-wide search) — its
  content is not recoverable from what's committed here.
- **Supervisor meeting count.** Task briefs reference "Meeting 3" and
  "Meeting 4" in passing (`phase2_poisonedrag_task.md:25`,
  `phase2_indirect_injection_task.md:45`) as the source of specific
  sequencing/scope decisions (e.g. the sequential-phase gate — don't start
  attack N+1's real sweep until attack N's insight file is closed out).
  No consolidated meeting log or count of meetings held/remaining was
  found in this repository.
- **Azure Files SMB atomic-write verification** (Section 3.4, Section 10)
  was explicitly deferred pending Part B provisioning of that
  infrastructure — since the project migrated to RunPod before that
  provisioning happened, this verification was never performed and, on
  current infrastructure, may no longer be applicable.
- **Write-up status.** This document is explicitly positioned, per the
  request that produced it, as thesis-chapter backbone material — the
  actual thesis chapters themselves are not part of this repository's
  committed material as of this compilation.

---

## 12. References

**No `CITATIONS.md` or equivalent bibliography file exists anywhere in this
repository** (confirmed via repository-wide search, both by filename and by
content). The list below was reconstructed entirely from the "Grounding"
sections of the three Phase 2 attack task briefs and from model/dataset
identifiers used directly in `config.py` and the codebase — it is a
best-effort compilation from primary project material, not a transcription
of a pre-existing reference list.

**Attack methodology papers:**

- Liu, Y., Jia, Y., Geng, R., Jia, J., & Gong, N. Z. (2024). *Formalizing
  and Benchmarking Prompt Injection Attacks and Defenses.* USENIX Security
  2024. Reference implementation: `liu00222/Open-Prompt-Injection` (GitHub).
- Zou, W., et al. (2025). *PoisonedRAG: Knowledge Corruption Attacks to
  Retrieval-Augmented Generation of Large Language Models.* USENIX Security
  2025. Reference implementation: `sleeepeer/PoisonedRAG` (GitHub).
- Russinovich, M., Salem, A., & Eldan, R. (2024). *Great, Now Write an
  Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack.*
  Reference implementation: Microsoft PyRIT, `Azure/PyRIT` (GitHub).

**Datasets:**

- HotpotQA (`hotpotqa/hotpot_qa`)
- MS MARCO (`microsoft/ms_marco`)
- Natural Questions Open / `nq_open` (`google-research-datasets/nq_open`) —
  excluded from all comparative analysis, see Section 4
- JailbreakBench / JBB-Behaviors (`JailbreakBench/JBB-Behaviors`)
- HarmBench (`walledai/HarmBench`, gated, used; `AlignmentResearch/HarmBench`,
  ungated, documented fallback, unused)
- XSTest (`Paul/XSTest`, migrated from the no-longer-live
  `paul-rottger/xstest`) — used for `instruction_detection` false-positive
  checking
- OR-Bench-Hard-1K — used for `instruction_detection` false-positive
  checking

**Models:**

- Llama-3.1-8B-Instruct (Meta)
- Ministral-3-8B-Instruct-2512 (Mistral AI)
- Phi-4-mini-instruct (Microsoft)
- Qwen3-8B (Alibaba)
- Llama-Guard-4-12B (Meta) — `output_filter`'s safety classifier
- `protectai/deberta-v3-base-prompt-injection-v2` (Protect AI) —
  `instruction_detection`'s injection classifier
- `nvidia/nemotron-3-ultra-550b-a55b` (NVIDIA, via NIM) — PoisonedRAG's
  poison-passage generator
- `deepseek-ai/deepseek-v4-pro-0813` (DeepSeek, via NIM/OpenRouter) —
  Crescendo's attacker and judge model

---

*End of master record. Every number above traces to a file cited inline;
Section 10 and Section 11 name every place a gap, an open question, or an
unverifiable claim was found rather than resolved by assumption.*
