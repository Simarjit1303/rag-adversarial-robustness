# Task: Build Indirect Prompt Injection (Phase 2, Attack 1 of 3)

Branch: `phase2-injection-templates` first, then `phase2-injection-sweep` once the templates are stable. Base off `main` (confirm you're not accidentally branching off `runpod-self-termination` or another in-flight feature branch — check `git log --oneline -3` before creating the branch).

Before touching any code, run this quick audit and report back:

- [ ] Confirm `PHASE2_INJECTION_INSIGHTS.md` does not already exist (it shouldn't — this task produces it, not consumes it).
- [ ] Confirm `evaluation/result_paths.py`'s naming pattern still applies: `{model}_{corpus}_{engine}` — this task extends it to `{model}_{corpus}_{injection_template}_{engine}`.
- [ ] Confirm no stray duplicate copies of the repo exist elsewhere on disk (one was found and removed this session; `.gitignore` now excludes `rag-adversarial-robustness/` as a safeguard — just confirm it's still gone).

If any of those turn up something unexpected, stop and report before proceeding.

## What this attack is, and how it differs from PoisonedRAG

PoisonedRAG plants a wrong fact. Indirect prompt injection plants a wrong instruction — content that tries to override what the model does with the question, rather than misinform it about the answer. Keep this distinction sharp in the implementation and in variable/function naming: these are testing genuinely different attack surfaces, not two flavors of "put bad text in the corpus."

## Grounding

Liu, Jia, Geng, Jia, and Gong, *Formalizing and Benchmarking Prompt Injection Attacks and Defenses* (USENIX Security 2024). Public benchmark platform: `https://github.com/liu00222/Open-Prompt-Injection`. Clone and read before writing anything new — most of the injection-template scaffolding needed here already exists there, and adapting a validated taxonomy is stronger footing than inventing one from scratch. Their own results found larger, more capable models more susceptible to prompt injection, not less — worth keeping in mind when the results come in and look surprising in that direction; that's a known pattern in the source literature, not necessarily a bug.

## Tasks

- [ ] Pull the five attack strategies from Liu et al. and decide which subset to implement (all five is the default; the combined attack is their strongest performer and must be included even if others get trimmed).
- [ ] Map each of the five strategies to **goal hijacking** (answer a different question) or **process hijacking** (take a different action/process than asked). The five formal strategies weren't written with this split in mind, so this mapping is a real design decision, not a lookup — document the reasoning per strategy in the code comments or the eventual insight file.
- [ ] Injection point: **organic placement** — append the injected instruction to an organically retrieved document, not forced into top-k. This is deliberate: PoisonedRAG already covers the forced-retrieval case, and using the same mechanism here would weaken the claim that the three attacks test different surfaces.
- [ ] Extend the Phase 1 filename convention in `evaluation/result_paths.py`: `attack_raw_{model}_{corpus}_{injection_template}_{engine}.jsonl`, mirroring `baseline_raw_{model}_{corpus}_{engine}.jsonl`. Same atomic-write pattern as Phase 1 (`_atomic_open`) — no exceptions.
- [ ] Build ASR scoring: did the output contain or follow the injected instruction's target string? Rule-based per template, not a fuzzy match.
- [ ] Pin the injection template set the same way `CORPORA` entries are pinned — template edits must not silently invalidate cached results.
- [ ] Decide and document: single injected sentence vs. a longer injected block. The original paper's strategies (escape-character, context-ignoring) already show the validated range — stay within it rather than inventing new injection styles.
- [ ] Decide and document: full five-strategy sweep vs. a curated subset. All five × four models × three corpora is 60 cells before sample size is even decided — this needs an intentional call, not a default.

## Statistics — decided this session, apply from the first sweep

- **McNemar's test**, paired per (model, corpus), same structure as Phase 1's stats plan.
- **Paired bootstrap 95% CI** on the utility drop (Phase 1 F1 vs. this attack's F1-under-attack), paired by question ID.
- **Holm-Bonferroni correction across the full McNemar p-value family for this attack.** This was decided this session, specifically because the 4×3×3 design generates many pairwise comparisons, and running them uncorrected inflates the false-positive rate on "significant" findings. Collect every McNemar p-value produced for this attack (all pairwise model comparisons per corpus, all pairwise corpus comparisons per model) before reporting any of them as significant, apply Holm-Bonferroni across that family, and report the corrected significance alongside the raw p-value in the insight file. Verified via repo audit this session: no McNemar results exist yet for this attack anywhere in the codebase, so this applies cleanly from the start — nothing to retroactively fix.

## Done means

- ASR table across the model/corpus grid (or documented reduced subset) for the injection strategies implemented.
- Utility-under-attack comparison against the Phase 1 baseline, with McNemar's + Holm-Bonferroni-corrected significance + paired bootstrap CI.
- `PHASE2_INJECTION_INSIGHTS.md` written: which models were most/least vulnerable to which injection strategies, a real explanation of why given the mechanism (not just the number restated), anything surprising named as surprising.
- Attack conditions stable enough that Phase 3 can put a defense in front of this without redesigning how the attack works.

Do not start PoisonedRAG's real sweep until this attack's insight file is written and closed out — code for attack 2 can begin once this attack's build is stable, but running and interpreting its results waits for this gate, per Meeting 4 supervisor feedback.
