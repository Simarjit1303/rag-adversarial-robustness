# Task: Build PoisonedRAG Corpus Poisoning (Phase 2, Attack 2 of 3)

Branch: `phase2-poisonedrag-templates` first, then `phase2-poisonedrag-sweep` once poison generation is verified. Base off `main` — confirm with `git log --oneline -3` before branching, same check as the injection task.

## Before writing any code: status and one real blocking check

- [x] **The baseline-validity question is already resolved — do not re-investigate this.** `phase1_results_complete/` for hotpot_qa/ms_marco was corrected via the full 8-cell rerun on `rerun-phase1-baseline-context-fix` (now merged into this branch's history, latest commit `978c55e`). The leak inflated F1 by 0.12–0.40 points per cell; corrected values are in place and confirmed via `PHASE2_INJECTION_INSIGHTS.md`'s methodology notes. `fix-rag-context-answer-leak` is folded in. Build directly against the current `phase1_results_complete/` — it's trustworthy.
- [ ] Confirm `PHASE2_INJECTION_INSIGHTS.md`'s methodology section on substring-match ASR limitations before designing PoisonedRAG's own scoring function — PoisonedRAG's success scoring needs the same scrutiny: does "attack succeeded" mean the model gave the poisoned answer, or could echoing/partial-match inflate this the same way it did for the injection marker (confirmed real: verbatim echo, marker-glued-to-real-answer, and genuine reasoned compliance were three different behaviors all collapsing into one binary success flag)? Design PoisonedRAG's scoring function with this failure mode in mind from the start, not discovered after a full sweep.
- [ ] Be aware: the corrected baseline also overturned the injection attack's own headline finding — utility damage (F1 drop) turned out to track ASR/compliance rather than being independent of it, once measured against clean context. Keep this in mind when framing PoisonedRAG's own utility-vs-success comparison; don't assume the two attacks will show the same relationship without checking.

## What this attack is, and how it differs from indirect injection

PoisonedRAG plants a *wrong fact*, not a wrong *instruction*. Indirect injection (Attack 1) tried to hijack the model's behavior; PoisonedRAG tries to corrupt what it believes to be true, by getting crafted adversarial passages retrieved instead of (or alongside) the real answer-bearing passage. Keep this distinction sharp — PoisonedRAG's "attack success" means the model states the poisoned/incorrect answer, not that it followed an injected command.

## Grounding

Zou et al., *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models* (USENIX Security 2025). Official implementation: `sleeepeer/PoisonedRAG` on GitHub — clone and read before building; this is the reference implementation your attack should faithfully adapt, not reinvent. Note: the official repo uses BEIR-format NQ, which is the same corpus-construction gap already identified for your project's nq_open — this reinforces (doesn't newly discover) that nq_open needs real Wikipedia passages to be used validly with this specific attack; since nq_open is already excluded from your real sweeps, this is just confirmation, not new work.

## Scope: hotpot_qa and ms_marco only

Same exclusion as the injection attack. `ATTACK_ELIGIBLE_CORPORA` (already defined in `evaluation/result_paths.py` from Attack 1) should be reused here, not redefined.

## Tasks

- [ ] Pull the poison-count and dataset-size decisions confirmed in Meeting 3 (referenced in `PHASE2_ROADMAP.md` — check there first rather than re-deciding these from scratch).
- [ ] Implement PoisonedRAG's poison-generation method: craft adversarial passages designed to (a) be retrieved for a target question (via similarity to the question, matching the paper's approach) and (b) contain a specific incorrect target answer.
- [ ] Implement the **retrieval-verification step**: confirm the poisoned passage(s) actually get retrieved into the top-k context before generation — this is a real, measurable intermediate check (distinct from final answer-correctness), and should be logged per-question, not just assumed. A poisoned passage that never gets retrieved can't be blamed for the model's answer either way.
- [ ] Decide and document: number of poisoned passages injected per question (Meeting 3 should have confirmed this — pull the number, don't guess), and whether poisoned passages replace or are added alongside the real top-k retrieval.
- [ ] Build success scoring: does the model's answer match the poisoned target answer? Rule-based, same substring-match caution as the injection ASR scorer — document known limitations up front given what Attack 1 found.
- [ ] Extend `evaluation/result_paths.py`'s naming convention for this attack's output files, following the same pattern as `attack_raw_{model}_{corpus}_{injection_template}_{engine}.jsonl` (adjust the axis name appropriately, e.g. `poison_config` or similar).

## Statistics — same discipline as Attack 1, apply from the start

- **McNemar's test**, paired per (model, corpus) — valid here since it's the same paired structure as the injection attack.
- **Fisher's exact test** for corpus-vs-corpus comparisons per model — same reasoning as the injection attack's fix: hotpot_qa and ms_marco are independent, unpaired question sets, so McNemar's pairing assumption doesn't hold there. Use `evaluation.stats.fisher_exact_asr_comparison`, already built for Attack 1 — don't reimplement.
- **Paired bootstrap 95% CI** on the utility drop vs. Phase 1 baseline — baseline is already confirmed valid, proceed directly.
- **Holm-Bonferroni correction** across the McNemar/Fisher p-value family — apply from the first sweep, not retroactively. If PoisonedRAG has fewer independent "template-like" conditions than the injection attack's five, adjust the family structure accordingly and document why.

## Done means

- Retrieval-verification confirms poisoned passages are actually reaching model context at a real, measured rate (not just assumed to be 100%).
- Attack success rate table across the model/corpus grid.
- Utility-under-attack comparison against the Phase 1 baseline, using a baseline confirmed valid per the pre-flight check above.
- `PHASE2_POISONEDRAG_INSIGHTS.md` written: which models were most/least susceptible to knowledge corruption and why, given the mechanism — not just numbers restated. Compare qualitatively against Attack 1's findings where relevant (e.g., does a model resistant to instruction-hijacking also resist fact-corruption, or are these genuinely independent robustness dimensions?).
- Attack conditions stable enough for Phase 3 to layer a defense on without redesigning anything.

Do not start Crescendo's real sweep until this attack's insight file is written and closed out, per the same sequential-phase discipline as Attack 1.
