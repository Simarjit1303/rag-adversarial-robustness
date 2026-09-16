# Phase 4: EU AI Act Article 15 Mapping

Synthesis of this project's already-final Phase 1–3 results against the
EU AI Act's Article 15 (Accuracy, robustness and cybersecurity) — no new
experiments, no GPU/pod work. Companion to
[`THESIS_MASTER_RECORD.md`](THESIS_MASTER_RECORD.md), which every result
number below is drawn from directly.

**Compiled:** 2026-09-16, against `THESIS_MASTER_RECORD.md` commit
`128771a` (the final, backend-corrected version). Regulatory context
verified via live web search against official EU sources on the same
date — see Section 1's sourcing notes for exactly what was and wasn't
confirmable through the tools available in this session.

---

## 1. Regulatory context — verified, with one discrepancy found and corrected

Before mapping anything, the assumed regulatory context this task started
from was checked against current, authoritative sources rather than taken
on faith, per the explicit instruction not to assume it was still
correct. **Most of it held up. One material error did not.**

### 1.1 What was wrong in the assumed context

The assumed framing stated "EU AI Act enacted as Regulation (EU)
2026/1744." **This is incorrect and is corrected here.** Confirmed via
EUR-Lex and cross-checked against multiple independent legal-tracker
sources (Cambridge Core's *International Legal Materials*, European
Sources Online, the EU AI Act Explorer):

- **The EU AI Act itself is Regulation (EU) 2024/1689** of the European
  Parliament and of the Council of 13 June 2024, laying down harmonised
  rules on artificial intelligence. Published in the Official Journal
  12 July 2024; entered into force 1 August 2024.
  (`https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng`)
- **Regulation (EU) 2026/1744 is the *Digital Omnibus on AI* — an
  amending regulation to the AI Act, not the AI Act itself.** Published
  in the Official Journal 24 July 2026; entered into force 27 July 2026.
  Confirmed independently by White & Case, K&L Gates, Mayer Brown, the
  Cloud Security Alliance, Hunton, and DLA Piper's tracking pages, all
  agreeing on this number, date, and characterization.

Article 15, the article this whole document is about, is a provision of
Regulation (EU) 2024/1689 — it is not renumbered or replaced by the
Digital Omnibus, which amends the Act's *deadlines and scope* (Section
1.2), not Article 15's substantive text.

### 1.2 What was confirmed accurate in the assumed context

The Digital Omnibus deferral timeline given in the original framing —
provisional agreement 7 May 2026, Council green light 29 June 2026,
Annex III obligations deferred from 2 August 2026 to 2 December 2027,
Annex I to 2 August 2028 — is **confirmed accurate**, sourced from the
Council of the EU's own press releases:

- Provisional political agreement between Council and Parliament:
  **7 May 2026** (Consilium press release,
  `consilium.europa.eu/en/press/press-releases/2026/05/07/...`).
- Council's final green light: **29 June 2026** (Consilium press release,
  `consilium.europa.eu/en/press/press-releases/2026/06/29/...`).
- Digital Omnibus on AI (Regulation (EU) 2026/1744) published in the
  Official Journal 24 July 2026, in force 27 July 2026.
- **Stand-alone Annex III high-risk AI system obligations** (the
  classification a deployed RAG-based system of the kind this project
  studies would most plausibly fall under, if classified high-risk at
  all — see the scoping caveat in Section 3.3) deferred from **2 August
  2026 to 2 December 2027**.
- **Annex I** (AI embedded in regulated products under existing product-
  safety legislation) deferred to **2 August 2028**.
- One correction to a secondary-source result this search surfaced along
  the way: an initial search pass returned "6 May" / "13 May" for the
  provisional-agreement and Council-approval dates from a law-firm blog
  summary — checked directly against the Council of the EU's own press
  release titles and dates, which read **7 May** and **29 June**
  respectively. The primary source is trusted over the conflicting
  secondary summary; flagged here so the discrepancy-check itself is
  auditable, not just its conclusion.

**Article 50 transparency obligations** (AI-disclosure to users,
deepfake labeling) are unaffected by the Omnibus and still apply from
2 August 2026; Article 50(2) watermarking obligations specifically are
separately postponed to 2 December 2026. Neither is directly relevant to
Article 15 or this project's scope, noted here only to avoid conflating
"the whole Act is delayed" with the narrower, accurate claim that
high-risk obligations specifically are deferred.

### 1.3 CEN-CENELEC JTC 21 harmonised standards — confirmed still delayed

The assumed context's claim that no Article-15-relevant harmonised
standard yet exists in the Official Journal is **confirmed accurate** as
of this search. CEN and CENELEC's Joint Technical Committee 21 has
accelerated work following a joint Technical Board decision (14–16
October 2025), but the two standards most directly relevant to Article
15 remain in draft ("pr", i.e. *proposed* European standard) status, not
yet published as final ENs and not yet cited in the Official Journal —
which is the specific act that would give them presumption-of-conformity
legal effect under the AI Act:

- **prEN 18229-2** — AI trustworthiness framework, Part 2: Accuracy and
  robustness.
- **prEN 18282** — Cybersecurity specifications for AI systems.
- **prEN 18281:2026** — Evaluation methods for computer vision systems
  (adjacent, not Article-15-specific).

This gap is returned to as the project's stated novelty hook in
Section 4.

### 1.4 Sourcing honesty on Article 15's text specifically

**Direct EUR-Lex fetches of Article 15's full text failed in this
session** — three attempts at different EUR-Lex URL forms
(`/eli/reg/2024/1689/oj/eng`, `/legal-content/EN/TXT/?uri=CELEX:32024R1689`,
`/eli/reg/2024/1689/oj`) all returned empty content through the fetch
tool available here, most likely because EUR-Lex's page requires
JavaScript rendering this tool doesn't execute. Rather than paper over
that with an unsourced paraphrase, Section 2 below quotes Article 15 from
two independent, mutually consistent secondary legal-text mirrors
(`artificialintelligenceact.eu` and `euaiact.com`) that both explicitly
state EUR-Lex governs in case of any conflict — the correct fallback
position, stated plainly rather than silently presented as a direct
EUR-Lex read.

---

## 2. Article 15's real requirements

Article 15, "Accuracy, robustness and cybersecurity," applies to
high-risk AI systems. Five paragraphs, quoted directly where the source
mirrors gave exact operative wording; summarized (clearly marked) where
they did not:

**Article 15(1):**

> "High-risk AI systems shall be designed and developed in such a way
> that they achieve an appropriate level of accuracy, robustness, and
> cybersecurity, and that they perform consistently in those respects
> throughout their lifecycle."

**Article 15(2)** *(summarized by both mirror sources, not given verbatim):*
the Commission is tasked with encouraging the development of benchmarks
and measurement methodologies for assessing these capabilities, in
cooperation with relevant stakeholders and organisations such as
metrology and benchmarking authorities. This is the provision under
which harmonised standards like the CEN-CENELEC work in Section 1.3
exist.

**Article 15(3):**

> "The levels of accuracy and the relevant accuracy metrics of high-risk
> AI systems shall be declared in the accompanying instructions of use."

**Article 15(4)** *(summarized, key phrases quoted):* high-risk AI
systems must be resilient regarding errors, faults, or inconsistencies
that may occur, through technical and organisational measures, which may
include "technical redundancy solutions, which may include backup or
fail-safe plans." Systems that continue to learn after deployment must
be designed to "eliminate or reduce as far as possible the risk of
possibly biased outputs influencing input for future operations
(feedback loops)."

**Article 15(5):**

> Systems must be "resilient against attempts by unauthorised third
> parties to alter their use, outputs or performance by exploiting
> system vulnerabilities."

Technical solutions addressing this must be "appropriate to the relevant
circumstances and the risks," and must address, where relevant, the
specific vulnerability classes both source mirrors list identically:
**data poisoning, model poisoning, adversarial examples or model evasion,
confidentiality attacks, and model flaws.**

**What is directly relevant to this project, stated plainly:**
paragraph (5) is the operative provision this dissertation's entire
attack/defense evaluation speaks to most directly — its general clause
("resilient against attempts by unauthorised third parties to alter
their use, outputs or performance by exploiting system vulnerabilities")
and its illustrative vulnerability list overlap substantially, though not
perfectly, with this project's three attack families (Section 3).
Paragraphs (1) and (3) speak to this project's utility/accuracy
measurements (Section 3). Paragraph (4)'s fault/redundancy resilience is
**not** something this project tested — stated as a gap, not silently
omitted (Section 3.4).

---

## 3. The mapping — Article 15's requirements against this project's final results

**Every number below is the final, backend-corrected figure from
`THESIS_MASTER_RECORD.md`** — where an earlier, uncorrected number would
have differed, that is noted explicitly so the correction itself is
visible, not just its outcome.

### 3.1 Article 15(1) and (3) — accuracy, declared consistently

| Requirement | This project's evidence | Verdict |
|---|---|---|
| "Appropriate level of accuracy... consistently... throughout lifecycle" | Phase 1 baseline: F1(clean)/EM per model × corpus, `hotpot_qa` 0.460–0.639, `ms_marco` 0.234–0.276 (`THESIS_MASTER_RECORD.md` Section 9.1). Phase 3 utility-preservation table (Section 6.4) tracks the *same* metric under every attack/defense condition, not a one-off number. | **Partially demonstrated.** Accuracy is measured and declared per condition, satisfying the spirit of the requirement for a benchmark project. "Throughout their lifecycle" implies longitudinal/deployment-drift monitoring, which this project — a fixed-point benchmark, not a deployed, monitored system — does not and cannot test. Gap named explicitly, not glossed over. |
| "Levels of accuracy and relevant accuracy metrics... declared in the accompanying instructions of use" | This project declares F1(clean) as primary, EM as secondary, `contains_answer` as diagnostic-only (`THESIS_MASTER_RECORD.md` Section 3.2, methodology), for every reported condition. | **Directly satisfied in spirit, with a scope caveat.** This project is a research benchmark, not a market-placed product issuing "instructions of use" to end users in the regulatory sense — the mapping here is that this project models the *kind* of transparent, hierarchy-declared metric reporting Article 15(3) requires of a real deployed system, not that this project itself is subject to the obligation. |

### 3.2 Article 15(4) — resilience to errors, faults, inconsistencies

**Not tested by this project — a genuine gap, stated as one.** No
fault-injection, redundancy, or backup/fail-safe testing was in scope for
any phase. This project's real-hardware bug-discovery work (Section 7's
eight bugs, `THESIS_MASTER_RECORD.md`) is adjacent — it demonstrates
*engineering* fault-discovery discipline — but is not a test of the
deployed system's own resilience to runtime faults, which is what 15(4)
actually asks for. Named here explicitly as outside this project's
demonstrated scope, not silently folded into a different finding that
only resembles it.

### 3.3 Article 15(5) — resilience against third-party manipulation, the core mapping

**Scoping caveat, stated once here rather than repeated at every row:**
this project does not itself establish that a RAG chatbot of the kind
studied here is legally classified as an Annex III high-risk AI system
under the AI Act — that classification depends on the system's actual
deployment context (e.g. use in employment, education, credit, law
enforcement, critical infrastructure — the categories Annex III lists),
which this project's benchmark scope does not specify. The mapping below
treats Article 15(5)'s *substantive* requirements as the relevant
technical standard to test against, independent of that separate legal
classification question.

| Article 15(5) vulnerability class | This project's attack | Final, backend-corrected result | Verdict |
|---|---|---|---|
| **Data poisoning** | PoisonedRAG (`THESIS_MASTER_RECORD.md` Section 5.2) | ASR 70–78% on `hotpot_qa`, 18–20% on `ms_marco`, all 4 models, corpus-dominated not model-dominated. **Defenses tested against it, final backend-corrected:** `spotlighting` is the only one with any significant cells (4/24, needing no correction — never confounded); `output_filter` flags **0.0000 of 160** scored poisoned responses — a structural architectural mismatch, not a partial failure (Section 6.3); `instruction_detection` shows the same severe utility cost pattern as the others (mean ΔF1 ≈ −0.30, Section 6.4). | **Directly tested, with a real compliance-relevant finding:** a widely-plausible off-the-shelf mitigation (a safety-classifier output filter) is *structurally* unable to address this vulnerability class at all, independent of tuning — a factually-wrong-but-safe-sounding answer is invisible to a safety classifier by design. This is a concrete instance of "appropriate to the relevant circumstances and the risks" (15(5)'s own qualifier) failing for a specific, plausible real-world mitigation choice. |
| **Adversarial examples / model evasion** (closest available category — see note below) | Indirect prompt injection (`THESIS_MASTER_RECORD.md` Section 5.1) | Model-dominated, not corpus-dominated. `ministral-3-8b` confirmed (after a lossless tokenizer round-trip diagnostic ruled out a measurement artifact) genuinely, partially compliant with the injected print-instruction at 30.5–60.8% ASR — real, but partial compliance, not full task hijacking (0/607 flagged rows abandon the real question). Other three models near-zero to modest (`qwen3-8b` 11.7–14.1% on `fake_completion` only; `phi-4-mini` 10.1% on `ms_marco`/`fake_completion` only; `llama-3.1-8b` near-zero everywhere). **Defenses, final backend-corrected:** `spotlighting` real effect, 9/16 cells significant, but severe utility cost (mean ΔF1 −0.2831); `output_filter` **0/19** significant — its apparent effect was almost entirely an inference-backend artifact, not the defense; `instruction_detection` **0/16** significant at n=40, but mechanism attribution confirms **100% (10/10)** of genuinely-blocked items have a classifier flag behind them — real, just statistically underpowered at this sample size, not absent. | **Directly tested, nuanced finding.** No tested defense achieves both a confirmed robustness effect *and* utility preservation simultaneously against this vulnerability class — the one defense with a confirmed effect (`spotlighting`) costs the most utility; the one nearly free on utility (`instruction_detection`) can't statistically confirm its effect at the sample size tested, though its mechanism is real. |
| **Model poisoning** | Not directly tested by any attack in this project — no attack modifies model weights or training data; all three attacks operate at inference time (retrieved context, or conversational turns) | Not applicable / out of scope | **Named gap.** This project's attack surface is entirely inference-time (RAG context, conversation), not training-time. Model poisoning (compromising weights or training data directly) is a distinct threat category this project does not speak to. |
| **Confidentiality attacks** | Not directly tested — no attack in this project attempts to extract training data, retrieved documents outside the intended context, or system-prompt/configuration secrets | Not applicable / out of scope | **Named gap**, same treatment as model poisoning — stated plainly rather than stretched to fit an unrelated finding. |
| **Model flaws** (general) / the opening clause: "attempts by unauthorised third parties to alter... use, outputs or performance by exploiting system vulnerabilities" | Crescendo (`THESIS_MASTER_RECORD.md` Section 5.3) | ASR 59.2–70.2% across all 4 models; no significant model-vs-model ranking survives Holm-Bonferroni correction, but the *mechanism* each model uses is a strong per-model signature (refuse-constantly-but-fold vs. barely-refuse-at-all vs. refuse-and-often-hold). **Defense: `output_filter` only, and diagnostic-only, by design** — its guard verdict is logged but deliberately never fed back into the conversation, so there is no measured ASR reduction to report, only a counterfactual: 67–82% of successful attacks had at least one turn the guard would have flagged, had it been wired to enforce (Section 6.3). | **Fits the general clause of 15(5) directly, even though "multi-turn conversational jailbreak" is not one of the five named illustrative vulnerability types.** This is itself a relevant observation about Article 15(5)'s own drafting (Section 4): its illustrative list is skewed toward classical ML adversarial-robustness categories (poisoning, evasion, confidentiality) and does not name conversational/multi-turn alignment erosion as a category, even though it plainly falls under the article's own general clause. The one defense tested against it was never wired to actually intervene — a genuine, named engineering gap in this project's own defense coverage (`THESIS_MASTER_RECORD.md` Section 11), not glossed over here either. |

### 3.4 Testing methodology and documentation — implicit throughout

Article 15 does not itself specify a testing methodology (that is
Article 15(2)'s benchmark/measurement-methodology mandate, delegated to
standards bodies — Section 1.3). What this project demonstrates,
concretely, as a candidate methodology:

- **Paired, exact statistical significance testing** (McNemar, Fisher's
  exact, Holm-Bonferroni correction within properly scoped families) —
  not the informal "we tried it and it seemed to work" standard common in
  the adjacent attack/defense literature this project's own `CITATIONS.md`
  surveys.
- **A demonstrated, corrected inference-backend confound**
  (`THESIS_MASTER_RECORD.md` Section 8) — proof by direct example that an
  uncontrolled infrastructure variable (`hf` vs. `vllm`) can fully explain
  an apparent defense effect large enough to look decisive (a 0.302–0.524
  absolute ASR reduction, fully attributable to the backend switch alone
  for `output_filter`). This is a directly citable methodological warning
  for anyone building a compliance-testing regime around before/after
  defense comparisons: **the inference stack itself must be held constant
  and verified, or a compliance test could certify a defense that does
  nothing.**
- **Real-hardware verification as a precondition for trustworthy results**
  (`THESIS_MASTER_RECORD.md` Section 7) — eight bugs invisible to a
  mocked test suite, found only by executing on real hardware against
  real model weights. A compliance-testing methodology that accepted
  results from an unexecuted or mock-tested pipeline could certify a
  defense that was never actually run correctly.

---

## 4. Discussion: this project's contribution to the compliance question

**The standards gap is the novelty hook, treated as a contribution, not
downplayed as a limitation.** As of this compilation (Section 1.3), no
harmonised standard for Article 15's accuracy/robustness or cybersecurity
requirements has been cited in the Official Journal — `prEN 18229-2` and
`prEN 18282` remain in draft. This means there is currently **no
technical standard a provider of a RAG-based high-risk AI system could
point to for a presumption of conformity with Article 15**, specifically
for the RAG deployment pattern this project studies. Even once those
general AI-trustworthiness standards are finalised, they are unlikely to
be RAG-specific at the level of detail this project operates at (the
retrieval-stage vs. generation-stage distinction; corpus-poisoning vs.
prompt-injection vs. multi-turn-conversational threat models as genuinely
separate axes, Section 5.4 of `THESIS_MASTER_RECORD.md`) — general AI
robustness standards are being drafted for AI systems broadly, not for
the specific architecture of "an LLM that retrieves external documents at
inference time," which is exactly the attack surface Article 15(5)'s
"exploiting system vulnerabilities" clause has to eventually be
operationalised against for this deployment pattern to be tested at all.

**What this project offers into that gap, concretely and without
overclaiming legal effect:**

1. **A demonstrated three-family threat taxonomy for RAG systems
   specifically** (corpus poisoning, indirect prompt injection, multi-turn
   conversational escalation) that maps — imperfectly but substantially,
   Section 3.3 — onto Article 15(5)'s own illustrative vulnerability list,
   while also surfacing where that list's classical-ML framing doesn't
   cleanly cover a RAG-relevant threat (Crescendo, Section 3.3's last
   row) — a concrete, citable data point for whoever eventually drafts
   the RAG-specific technical content of a future standard.
2. **A demonstrated statistical testing discipline** (Section 3.4) that
   could inform what "adequate testing" means operationally for Article
   15(2)'s benchmark mandate, in a domain where the existing published
   literature (per this project's own `CITATIONS.md` survey) inconsistently
   reports significance testing on ASR differences at all.
3. **A demonstrated, citable warning about a specific class of testing
   error** — the inference-backend confound (Section 3.4) — that a
   compliance-testing regime built without this project's discipline
   could walk directly into: certifying a "defense" whose entire apparent
   effect was an infrastructure artifact.
4. **An honest example of what current, real, off-the-shelf inference-time
   mitigations actually achieve against RAG-specific threats, at real
   effect sizes** — not one, but a full three-defense, three-attack grid
   showing that "add a safety filter" (`output_filter`) can be
   *structurally* the wrong mitigation for a given threat class
   (PoisonedRAG, Section 3.3), that a real, working mitigation
   (`spotlighting`) can come at a utility cost severe enough to raise its
   own Article 15(1)/(3) accuracy-consistency question, and that a
   mitigation with a genuine, mechanism-confirmed effect
   (`instruction_detection`) can still fail to reach statistical
   significance at a defensible sample size — three different, concrete
   failure modes a future compliance-testing standard would need to be
   built carefully enough to actually distinguish between, not just one
   generic "the defense didn't fully work" story.

**What this project does not claim.** This is not a legal compliance
assessment, a conformity assessment under Article 43, or a substitute for
the eventual harmonised standard. It does not establish that any specific
deployed system is or is not Annex III high-risk, and it does not purport
to test every Article 15(5) vulnerability class (model poisoning and
confidentiality attacks are named, explicit gaps, Section 3.3). What it
does claim, and can defend directly against its own final numbers: a
concrete, statistically rigorous, real-hardware-verified demonstration of
what testing RAG-specific adversarial robustness under an Article-15-shaped
lens actually looks like, at a moment when no harmonised standard yet
exists to say what that should look like instead.

---

*Companion to [`THESIS_MASTER_RECORD.md`](THESIS_MASTER_RECORD.md) (every
result number above traces to that document's current, final,
backend-corrected content) and [`CITATIONS.md`](CITATIONS.md) (the
peer-reviewed literature this project's own threat taxonomy and
statistical methodology were built against). Regulatory sourcing per
Section 1 above — re-verify all dates and the Official Journal citation
status before final submission, since both remain live-moving targets.*
