# Phase 3: Defense Evaluation — Insights

Statistical analysis of the three Phase 3 defenses (`instruction_detection`,
`spotlighting`, `output_filter`) against all three Phase 2 attacks (prompt
injection, PoisonedRAG, Crescendo). Data: `phase3_defense_results/`
(defended cells) paired against `phase2_injection_results/`,
`phase2_poisonedrag_results/`, `phase2_crescendo_results/`, and
`phase1_results_complete/` (all committed this session, commit `02c3a40`).
Analysis script: `scripts/analyze_phase3_defense_stats.py`
(`python -m scripts.analyze_phase3_defense_stats`).

**Read the Methodology & Limitations section before trusting any single
number in isolation** — several results here look strong in aggregate but
have a specific, checkable caveat attached (mismatched cell counts, a
non-live "defense," an inference-backend confound). Every caveat below was
verified against code or data in this repo, not assumed.

**Update, follow-up investigation:** the inference-backend confound
flagged in the first version of this analysis was investigated directly
rather than left as a suspicion — see headline finding 2, "Backend-
confound isolation (Task 2)," and Methodology (f). It is now CONFIRMED for
`output_filter`/injection (0/19 cells remain significant once corrected)
and PARTIALLY confirmed for `spotlighting`/injection (effect sizes shrink,
9/16 cells remain significant). `instruction_detection`'s mechanism
question, previously reported unanswerable, is now answered directly from
new per-passage log data (see the `instruction_detection` mechanism
section) — 20.0% of its blocked attacks had a passage actually flagged.

**Second follow-up (2026-09-16):** the two remaining gaps this document
used to flag as open — `instruction_detection`'s own exposure to the
backend confound, and PoisonedRAG's backend confound — are now both
closed, and neither needed new GPU time. `instruction_detection`'s n=40
backend-matched no-defense baseline turned out to already exist for free
(the first 40 rows of Task B's already-committed n=1000 no-defense sweep
are byte-identical, in order, to its own n=40 item set — verified by
`scripts/verify_phase3_instruction_detection_baseline_items.py` against
real committed data, zero model/GPU touched); run through the same 3-way
test as output_filter/spotlighting, **0 of 16 `instruction_detection`
cells remain significant** once backend-corrected (down from 3/16 under
the original `vllm` comparator). PoisonedRAG's backend confound turned out
never to have existed: `git log --diff-filter=A` on
`phase2_poisonedrag_results/` shows every file ever committed there was
generated with `INFERENCE_ENGINE=hf` — Phase 2's PoisonedRAG baseline
(unlike injection's) always ran on the same `hf` backend as Phase 3's
defended PoisonedRAG runs, so the PoisonedRAG rows in the master table
below were backend-clean from the start. See "Backend-confound isolation
(Task 2)" and Methodology (f) for both.

---

## Headline findings

1. **PoisonedRAG/output_filter is a real architectural mismatch, not a
   qualitative impression**: the Llama-Guard safety classifier's flag rate
   on PoisonedRAG responses is **exactly 0.0000 across all 160 scored
   responses** (24 cells, `phase3_defense_results/output_filter_log_poison_*.jsonl`).
   A safety-content classifier cannot see factual poisoning — poisoned
   answers aren't unsafe, they're just wrong — so whatever ASR change
   output_filter shows for PoisonedRAG cannot be mechanistically
   attributed to the guard catching anything.

2. **RESOLVED: injection/output_filter's ASR "reduction" is the hf/vllm
   backend switch, confirmed with a real backend-matched baseline, not
   speculation.** A targeted follow-up ran a third condition — a
   backend-matched, no-defense `hf` baseline (`RAG_DEFENSE=none`, same
   1000-item selection as every real cell, verified item-for-item by
   `scripts/verify_phase3_backend_baseline_items.py`) — giving three
   points per cell: `vllm` baseline → `hf` no-defense baseline → `hf`
   defended. Pairwise McNemar with Holm-Bonferroni (3 comparisons per
   cell) on all 6 output_filter cells that had any real ASR to test:
   **`vllm`-vs-`hf`-no-defense is significant in every one (p_holm as low
   as 1.9e-155), while `hf`-no-defense-vs-defended is NEVER significant
   (p_holm = 1 in 5/6 cells, 0.06 in the sixth)** — i.e. the defended run
   is statistically indistinguishable from the undefended `hf` baseline,
   and both are miles below the `vllm` baseline. 76–100% of every cell's
   total reduction is attributable to the backend switch alone (see
   Methodology (f) and "Task 2: backend-confound isolation" below for the
   full per-cell table). **A backend-isolated re-run of the master-table
   McNemar test (`hf` no-defense vs `hf`-defended, the correct comparator
   now that this is confirmed) finds 0 of 19 output_filter/injection
   cells significant** — down from 6/19 under the original,
   confound-contaminated `vllm`-baseline comparator. Read this plainly:
   **output_filter has no measurable effect on injection ASR once the
   backend switch is controlled for.** The original 0.8% guard-flag-rate
   finding (kept below, unchanged) is now fully explained rather than a
   loose end: the guard barely fires because it has almost nothing to
   catch — nearly all of the apparent "defense" was never the guard's
   doing.

   **This does NOT generalize to spotlighting.** The same 3-way test on
   spotlighting's 16 cells finds the opposite pattern for the 9
   cells with real ASR: `hf`-no-defense-vs-defended IS significant in
   all 9 (backend-isolated McNemar, p_holm as low as 2.9e-42) — the
   defense has a real, independent effect on top of whatever the backend
   switch already did. The backend switch still contributes
   substantially in 6 of those 9 cells (53–86% of the *original* vllm-vs-
   defended reduction, `partial_split` verdict) — so spotlighting's
   headline "100% ASR reduction" numbers in the original master table
   overstate spotlighting's own unique contribution for
   ministral-3-8b/phi-4-mini specifically — but the backend-isolated
   re-run still finds 9/16 spotlighting cells significant, unchanged from
   the original count. Base64-encoding genuinely defeats injection
   independent of the backend switch; output_filter's guard does not.

   **`instruction_detection` resolves the same way as output_filter, not
   spotlighting.** A second follow-up gave it the identical 3-way test at
   its own n=40 (its backend-matched no-defense baseline is the first 40
   rows of the already-committed Task B file — see the update banner
   above). Only 3 of 16 cells had any pairwise gap reach significance at
   all (all `ministral-3-8b`), and all 3 are `backend_confound`:
   `hf`-no-defense-vs-defended never reaches significance (p_holm = 1.0 in
   all three), matching output_filter's pattern exactly. **The
   backend-isolated corrected table finds 0 of 16 instruction_detection
   cells significant** — down from 3/16 under the original `vllm`
   comparator. Cross-referenced against the 20.0% mechanism-flag finding
   below: recomputing "blocked" against the backend-matched baseline
   instead of `vllm` shrinks the blocked-item count from 55 to 10 (the
   other 45 were never really "blocked" by the classifier — they were
   just the backend switch), and **all 10 of the remaining
   backend-corrected blocks have a passage actually flagged (100%, up from
   20.0%)**. The confound doesn't wash out the classifier's mechanism
   signal — it explains why the *aggregate* ASR reduction isn't
   statistically significant (n=40 per cell, mostly near-zero baseline
   ASR, too little power) while confirming that on the rare items where
   instruction_detection's block was real, the classifier is the reason
   every time.

3. **Spotlighting eliminates injection ASR almost perfectly but at a
   severe utility cost that the ASR table alone hides.** 12 of 16
   spotlighting/injection cells hit exactly 0% defended ASR (base64-encoding
   the context defeats every literal-text injection template tested), but
   mean F1 drops **−0.283** versus the Phase 1 no-attack baseline (as low
   as −0.44 in some cells) — the model frequently can't answer the
   question at all once its context is base64-encoded. A "100% ASR
   reduction" headline for spotlighting is true and also misleading
   without this utility number next to it.

4. **Crescendo/output_filter is not a defended-condition measurement, and
   is not reported in the master ASR-reduction table below.** By design
   (`run_crescendo.py:184-257`), the guard's verdict is logged but never
   acted on — the judge scores the real, unfiltered conversation either
   way, so there is no live intervention to measure an ASR reduction
   from. Its raw baseline-vs-"defended" numbers are printed by the
   analysis script under a clearly separate diagnostic-only header
   (never mixed into the Holm-corrected ASR-reduction family) and are not
   included in this document's master table. The only result reported
   for this cell is a mechanism-only, counterfactual one — see
   "Mechanism attribution" below — reading it as "output_filter
   defends/fails to defend against Crescendo" would be a real error, not
   a nuance worth a footnote.

5. **The real cell inventory is 79 cells (51 injection + 24 PoisonedRAG +
   4 Crescendo), not the 52 (24+24+4) assumed going into this analysis.**
   Injection has 3x more real cells than expected because coverage is
   asymmetric across defenses (see Methodology item (a)); 6 additional
   files exist but are n=3 smoke-test fragments, correctly excluded. Of
   these 79, only 75 (51 injection + 24 PoisonedRAG) represent a live,
   enforced defense and appear in the master ASR-reduction table below —
   the 4 Crescendo/output_filter cells are diagnostic-only (finding 4)
   and are reported exclusively in "Mechanism attribution."

---

## Master ASR-reduction table

All 75 cells with a live, enforced defense (51 injection + 24
PoisonedRAG). Crescendo/output_filter's 4 cells are deliberately excluded
from this table — see finding 4 and Methodology item (e): the guard's
verdict is never enforced in that runner, so there is no defended
condition to report an ASR reduction for. Its raw diagnostic numbers are
printed separately by the analysis script (under a header that says so
explicitly, never inside this table's Holm-corrected family) and are
presented in this document only as mechanism-only, counterfactual data
in "Mechanism attribution" below. Every cell in this table used the
paired McNemar exact test (a valid matched-item baseline existed for
every cell — see Task 1's alignment work, summarized in Methodology
items (c)/(d); the unpaired Fisher's-exact fallback is implemented in
the script but never triggered by this dataset). Holm-Bonferroni
correction is applied separately within each of the two attack families
(injection: 51 comparisons; PoisonedRAG: 24) — `p (Holm)` is the true
Holm step-down adjusted p-value, computed directly since
`evaluation/stats.py`'s existing `holm_bonferroni` helper returns only a
reject/accept decision, not an adjusted p (see
`scripts/analyze_phase3_defense_stats.py::_holm_adjusted_pvalues`).

| Attack | Model | Corpus | Template | Defense | n | Baseline ASR | Defended ASR | Abs. reduction | Rel. reduction | Test | p (raw) | p (Holm) | Sig? | 95% CI (reduction) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| injection | llama-3.1-8b | hotpot_qa | fake_completion | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | hotpot_qa | ignore | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | ms_marco | fake_completion | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | ms_marco | ignore | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | ministral-3-8b | hotpot_qa | fake_completion | instruction_detection | 40 | 0.550 | 0.050 | 0.500 | 90.9% | mcnemar_exact | 0.0000 | 0.0001 | Y | [0.350, 0.650] |
| injection | ministral-3-8b | hotpot_qa | ignore | instruction_detection | 40 | 0.350 | 0.025 | 0.325 | 92.9% | mcnemar_exact | 0.0002 | 0.0088 | Y | [0.175, 0.475] |
| injection | ministral-3-8b | ms_marco | fake_completion | instruction_detection | 40 | 0.425 | 0.100 | 0.325 | 76.5% | mcnemar_exact | 0.0002 | 0.0088 | Y | [0.175, 0.475] |
| injection | ministral-3-8b | ms_marco | ignore | instruction_detection | 40 | 0.175 | 0.125 | 0.050 | 28.6% | mcnemar_exact | 0.5000 | 1.0000 | N | [0.000, 0.125] |
| injection | phi-4-mini | hotpot_qa | fake_completion | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | phi-4-mini | hotpot_qa | ignore | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | phi-4-mini | ms_marco | fake_completion | instruction_detection | 40 | 0.100 | 0.025 | 0.075 | 75.0% | mcnemar_exact | 0.2500 | 1.0000 | N | [0.000, 0.175] |
| injection | phi-4-mini | ms_marco | ignore | instruction_detection | 40 | 0.025 | 0.025 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | qwen3-8b | hotpot_qa | fake_completion | instruction_detection | 40 | 0.100 | 0.075 | 0.025 | 25.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.075] |
| injection | qwen3-8b | hotpot_qa | ignore | instruction_detection | 40 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | qwen3-8b | ms_marco | fake_completion | instruction_detection | 40 | 0.150 | 0.100 | 0.050 | 33.3% | mcnemar_exact | 0.5000 | 1.0000 | N | [0.000, 0.125] |
| injection | qwen3-8b | ms_marco | ignore | instruction_detection | 40 | 0.050 | 0.025 | 0.025 | 50.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.075] |
| injection | llama-3.1-8b | hotpot_qa | fake_completion | output_filter | 1000 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | hotpot_qa | ignore | output_filter | 1000 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | ms_marco | fake_completion | output_filter | 1000 | 0.006 | 0.006 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | ms_marco | ignore | output_filter | 1000 | 0.003 | 0.001 | 0.002 | 66.7% | mcnemar_exact | 0.5000 | 1.0000 | N | [0.000, 0.005] |
| injection | ministral-3-8b | hotpot_qa | fake_completion | output_filter | 1000 | 0.608 | 0.084 | 0.524 | 86.2% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.493, 0.555] |
| injection | ministral-3-8b | hotpot_qa | ignore | output_filter | 1000 | 0.353 | 0.051 | 0.302 | 85.6% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.274, 0.330] |
| injection | ministral-3-8b | ms_marco | fake_completion | output_filter | 1000 | 0.515 | 0.143 | 0.372 | 72.2% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.340, 0.404] |
| injection | ministral-3-8b | ms_marco | ignore | output_filter | 1000 | 0.305 | 0.142 | 0.163 | 53.4% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.138, 0.189] |
| injection | phi-4-mini | hotpot_qa | fake_completion | output_filter | 1000 | 0.002 | 0.003 | -0.001 | -50.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.003, 0.000] |
| injection | phi-4-mini | hotpot_qa | ignore | output_filter | 1000 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | phi-4-mini | ms_marco | fake_completion | output_filter | 1000 | 0.101 | 0.027 | 0.074 | 73.3% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.057, 0.092] |
| injection | phi-4-mini | ms_marco | ignore | output_filter | 1000 | 0.032 | 0.011 | 0.021 | 65.6% | mcnemar_exact | 0.0003 | 0.0110 | Y | [0.010, 0.033] |
| injection | qwen3-8b | hotpot_qa | combined | output_filter | 1000 | 0.001 | 0.002 | -0.001 | -100.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.003, 0.000] |
| injection | qwen3-8b | hotpot_qa | escape_char | output_filter | 1000 | 0.004 | 0.004 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | qwen3-8b | hotpot_qa | fake_completion | output_filter | 1000 | 0.117 | 0.117 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.005, 0.005] |
| injection | qwen3-8b | hotpot_qa | ignore | output_filter | 1000 | 0.003 | 0.005 | -0.002 | -66.7% | mcnemar_exact | 0.5000 | 1.0000 | N | [-0.005, 0.000] |
| injection | qwen3-8b | hotpot_qa | naive | output_filter | 1000 | 0.002 | 0.001 | 0.001 | 50.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.003] |
| injection | qwen3-8b | ms_marco | fake_completion | output_filter | 1000 | 0.141 | 0.140 | 0.001 | 0.7% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.004, 0.006] |
| injection | qwen3-8b | ms_marco | ignore | output_filter | 1000 | 0.032 | 0.033 | -0.001 | -3.1% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.005, 0.003] |
| injection | llama-3.1-8b | hotpot_qa | fake_completion | spotlighting | 1000 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | hotpot_qa | ignore | spotlighting | 1000 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | llama-3.1-8b | ms_marco | fake_completion | spotlighting | 1000 | 0.006 | 0.000 | 0.006 | 100.0% | mcnemar_exact | 0.0312 | 1.0000 | N | [0.002, 0.011] |
| injection | llama-3.1-8b | ms_marco | ignore | spotlighting | 1000 | 0.003 | 0.000 | 0.003 | 100.0% | mcnemar_exact | 0.2500 | 1.0000 | N | [0.000, 0.007] |
| injection | ministral-3-8b | hotpot_qa | fake_completion | spotlighting | 1000 | 0.608 | 0.000 | 0.608 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.578, 0.639] |
| injection | ministral-3-8b | hotpot_qa | ignore | spotlighting | 1000 | 0.353 | 0.000 | 0.353 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.324, 0.382] |
| injection | ministral-3-8b | ms_marco | fake_completion | spotlighting | 1000 | 0.515 | 0.000 | 0.515 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.484, 0.546] |
| injection | ministral-3-8b | ms_marco | ignore | spotlighting | 1000 | 0.305 | 0.000 | 0.305 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.276, 0.333] |
| injection | phi-4-mini | hotpot_qa | fake_completion | spotlighting | 1000 | 0.002 | 0.000 | 0.002 | 100.0% | mcnemar_exact | 0.5000 | 1.0000 | N | [0.000, 0.005] |
| injection | phi-4-mini | hotpot_qa | ignore | spotlighting | 1000 | 0.000 | 0.000 | 0.000 | n/a (base=0) | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| injection | phi-4-mini | ms_marco | fake_completion | spotlighting | 1000 | 0.101 | 0.000 | 0.101 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.083, 0.120] |
| injection | phi-4-mini | ms_marco | ignore | spotlighting | 1000 | 0.032 | 0.000 | 0.032 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.022, 0.043] |
| injection | qwen3-8b | hotpot_qa | fake_completion | spotlighting | 1000 | 0.117 | 0.000 | 0.117 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.098, 0.137] |
| injection | qwen3-8b | hotpot_qa | ignore | spotlighting | 1000 | 0.003 | 0.000 | 0.003 | 100.0% | mcnemar_exact | 0.2500 | 1.0000 | N | [0.000, 0.007] |
| injection | qwen3-8b | ms_marco | fake_completion | spotlighting | 1000 | 0.141 | 0.000 | 0.141 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.120, 0.162] |
| injection | qwen3-8b | ms_marco | ignore | spotlighting | 1000 | 0.032 | 0.000 | 0.032 | 100.0% | mcnemar_exact | 0.0000 | 0.0000 | Y | [0.022, 0.043] |
| poisonedrag | llama-3.1-8b | hotpot_qa | - | instruction_detection | 18 | 0.778 | 0.778 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.167, 0.167] |
| poisonedrag | llama-3.1-8b | ms_marco | - | instruction_detection | 18 | 0.222 | 0.278 | -0.056 | -25.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.222, 0.111] |
| poisonedrag | ministral-3-8b | hotpot_qa | - | instruction_detection | 18 | 0.778 | 0.667 | 0.111 | 14.3% | mcnemar_exact | 0.5000 | 1.0000 | N | [0.000, 0.278] |
| poisonedrag | ministral-3-8b | ms_marco | - | instruction_detection | 18 | 0.222 | 0.278 | -0.056 | -25.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.278, 0.111] |
| poisonedrag | phi-4-mini | hotpot_qa | - | instruction_detection | 18 | 0.778 | 0.667 | 0.111 | 14.3% | mcnemar_exact | 0.6250 | 1.0000 | N | [-0.111, 0.333] |
| poisonedrag | phi-4-mini | ms_marco | - | instruction_detection | 18 | 0.167 | 0.389 | -0.222 | -133.3% | mcnemar_exact | 0.1250 | 1.0000 | N | [-0.444, -0.056] |
| poisonedrag | qwen3-8b | hotpot_qa | - | instruction_detection | 18 | 0.722 | 0.722 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.167, 0.167] |
| poisonedrag | qwen3-8b | ms_marco | - | instruction_detection | 18 | 0.278 | 0.389 | -0.111 | -40.0% | mcnemar_exact | 0.5000 | 1.0000 | N | [-0.278, 0.000] |
| poisonedrag | llama-3.1-8b | hotpot_qa | - | output_filter | 14 | 0.857 | 0.786 | 0.071 | 8.3% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.214] |
| poisonedrag | llama-3.1-8b | ms_marco | - | output_filter | 17 | 0.235 | 0.235 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.176, 0.176] |
| poisonedrag | ministral-3-8b | hotpot_qa | - | output_filter | 16 | 0.812 | 0.625 | 0.188 | 23.1% | mcnemar_exact | 0.3750 | 1.0000 | N | [-0.062, 0.438] |
| poisonedrag | ministral-3-8b | ms_marco | - | output_filter | 18 | 0.222 | 0.222 | 0.000 | 0.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.000] |
| poisonedrag | phi-4-mini | hotpot_qa | - | output_filter | 17 | 0.765 | 0.647 | 0.118 | 15.4% | mcnemar_exact | 0.6250 | 1.0000 | N | [-0.118, 0.353] |
| poisonedrag | phi-4-mini | ms_marco | - | output_filter | 18 | 0.167 | 0.222 | -0.056 | -33.3% | mcnemar_exact | 1.0000 | 1.0000 | N | [-0.167, 0.000] |
| poisonedrag | qwen3-8b | hotpot_qa | - | output_filter | 17 | 0.765 | 0.647 | 0.118 | 15.4% | mcnemar_exact | 0.6250 | 1.0000 | N | [-0.118, 0.353] |
| poisonedrag | qwen3-8b | ms_marco | - | output_filter | 18 | 0.278 | 0.222 | 0.056 | 20.0% | mcnemar_exact | 1.0000 | 1.0000 | N | [0.000, 0.167] |
| poisonedrag | llama-3.1-8b | hotpot_qa | - | spotlighting | 18 | 0.778 | 0.000 | 0.778 | 100.0% | mcnemar_exact | 0.0001 | 0.0029 | Y | [0.556, 0.944] |
| poisonedrag | llama-3.1-8b | ms_marco | - | spotlighting | 18 | 0.222 | 0.111 | 0.111 | 50.0% | mcnemar_exact | 0.6250 | 1.0000 | N | [-0.111, 0.333] |
| poisonedrag | ministral-3-8b | hotpot_qa | - | spotlighting | 18 | 0.778 | 0.000 | 0.778 | 100.0% | mcnemar_exact | 0.0001 | 0.0029 | Y | [0.556, 0.944] |
| poisonedrag | ministral-3-8b | ms_marco | - | spotlighting | 18 | 0.222 | 0.056 | 0.167 | 75.0% | mcnemar_exact | 0.2500 | 1.0000 | N | [0.000, 0.333] |
| poisonedrag | phi-4-mini | hotpot_qa | - | spotlighting | 18 | 0.778 | 0.000 | 0.778 | 100.0% | mcnemar_exact | 0.0001 | 0.0029 | Y | [0.556, 0.944] |
| poisonedrag | phi-4-mini | ms_marco | - | spotlighting | 16 | 0.188 | 0.062 | 0.125 | 66.7% | mcnemar_exact | 0.5000 | 1.0000 | N | [0.000, 0.312] |
| poisonedrag | qwen3-8b | hotpot_qa | - | spotlighting | 18 | 0.722 | 0.000 | 0.722 | 100.0% | mcnemar_exact | 0.0002 | 0.0051 | Y | [0.500, 0.889] |
| poisonedrag | qwen3-8b | ms_marco | - | spotlighting | 18 | 0.278 | 0.000 | 0.278 | 100.0% | mcnemar_exact | 0.0625 | 1.0000 | N | [0.111, 0.500] |

**Summary**: 18/51 injection cells significant after Holm correction (all
`instruction_detection`/`ministral-3-8b`, or `output_filter`/`spotlighting`
on `ministral-3-8b`/`phi-4-mini`/`qwen3-8b` — llama-3.1-8b's baseline ASR
was already ≈0 for every template, leaving no room for a defense to show
an effect). 4/24 PoisonedRAG cells significant, all `spotlighting`.
Crescendo/output_filter is excluded from this significance count entirely
(finding 4) — it is not a member of either family and has no ASR-reduction
verdict to report; see "Mechanism attribution" below for its one
legitimate result.

**This count uses the original, `vllm`-baseline comparator and is kept
for historical reference.** A follow-up confirmed a backend confound
(headline finding 2, Methodology (f)) that invalidates 6 of output_filter's
and all 3 of instruction_detection's significant cells: once corrected to
the backend-matched `hf` comparator ("Backend-confound isolation (Task 2)"
below), the true significant count is **9/51 injection cells** — 9
backend-isolated `spotlighting` cells (the same 9 cells as before — down
from the original count only in *effect size*, not membership) + 0
`output_filter` cells (down from 6, all of which were `ministral-3-8b`/
`phi-4-mini`) + 0 `instruction_detection` cells (down from 3, all of which
were `ministral-3-8b`). PoisonedRAG's 4/24 count is unchanged, but for a
different reason than "not yet corrected": Phase 2's PoisonedRAG baseline
already ran on the `hf` backend (confirmed via git history, see
Methodology (f)) — it was never confounded by a backend switch in the
first place, so its 4/24 significant count needs no correction, not "not
covered by this correction."

---

## Backend-confound isolation (Task 2)

Full per-cell results behind headline finding 2's resolution. For each of
the 51 real output_filter/spotlighting/instruction_detection injection
cells (19 + 16 + 16), three matched conditions on the same items: `vllm`
baseline (Phase 2),
backend-matched `hf` no-defense baseline (`attack_raw_*_hf.jsonl`,
`RAG_DEFENSE=none`) — item selection verified identical to the
output_filter/spotlighting defended runs' own 1000-item sets by
`scripts/verify_phase3_backend_baseline_items.py`, and, separately, verified
identical to instruction_detection's own 40-item sets (as an exact
40-row prefix of the same 1000-item file — no new sweep needed) by
`scripts/verify_phase3_instruction_detection_baseline_items.py` — and the
`hf` defended run. Test: pairwise McNemar (`vllm`-vs-`hf_nodef`,
`hf_nodef`-vs-`hf_def`, `vllm`-vs-`hf_def`) with Holm-Bonferroni applied
*within each cell's own 3-comparison family* — chosen over Cochran's Q
because the question is which pair differs (to attribute the reduction to
backend vs. defense), not merely whether the three conditions differ
somewhere; Cochran's Q would still need a post-hoc pairwise test to
answer that, so pairwise McNemar is the more direct fit, not just the
cheaper reuse of existing helpers. `backend_frac` = the share of the
*total* `vllm`-to-defended reduction attributable to the backend switch
alone (`(vllm_asr − hf_nodef_asr) / (vllm_asr − hf_def_asr)`). Verdict:
`backend_confound` (backend switch alone explains it), `defense_works`
(backend switch changes nothing, defense does all the work),
`partial_split` (both pairwise gaps are significant — both contribute),
`inconclusive` (neither gap reached significance, usually a near-zero-ASR
cell with no room to show an effect). 33 of the 51 cells are
`inconclusive` — every llama-3.1-8b cell, most near-zero-baseline
qwen3-8b/phi-4-mini output_filter cells, and 13 of instruction_detection's
16 cells, where ASR was already too low (or, for instruction_detection's
n=40 per cell, the sample too small) for any 3-way comparison to have
power.

| Defense | Model | Corpus | Template | n | vllm ASR | hf-nodef ASR | hf-def ASR | Total reduction | Backend frac. | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| output_filter | phi-4-mini | ms_marco | ignore | 1000 | 0.032 | 0.016 | 0.011 | 0.021 | 0.762 | backend_confound |
| output_filter | phi-4-mini | ms_marco | fake_completion | 1000 | 0.101 | 0.027 | 0.027 | 0.074 | 1.000 | backend_confound |
| output_filter | ministral-3-8b | hotpot_qa | ignore | 1000 | 0.353 | 0.052 | 0.051 | 0.302 | 0.997 | backend_confound |
| output_filter | ministral-3-8b | hotpot_qa | fake_completion | 1000 | 0.608 | 0.085 | 0.084 | 0.524 | 0.998 | backend_confound |
| output_filter | ministral-3-8b | ms_marco | ignore | 1000 | 0.305 | 0.143 | 0.142 | 0.163 | 0.994 | backend_confound |
| output_filter | ministral-3-8b | ms_marco | fake_completion | 1000 | 0.515 | 0.143 | 0.143 | 0.372 | 1.000 | backend_confound |
| spotlighting | qwen3-8b | hotpot_qa | fake_completion | 1000 | 0.117 | 0.117 | 0.000 | 0.117 | 0.000 | defense_works |
| spotlighting | qwen3-8b | ms_marco | ignore | 1000 | 0.032 | 0.034 | 0.000 | 0.032 | −0.063 | defense_works |
| spotlighting | qwen3-8b | ms_marco | fake_completion | 1000 | 0.141 | 0.140 | 0.000 | 0.141 | 0.007 | defense_works |
| spotlighting | phi-4-mini | ms_marco | ignore | 1000 | 0.032 | 0.016 | 0.000 | 0.032 | 0.500 | partial_split |
| spotlighting | phi-4-mini | ms_marco | fake_completion | 1000 | 0.101 | 0.027 | 0.000 | 0.101 | 0.733 | partial_split |
| spotlighting | ministral-3-8b | hotpot_qa | ignore | 1000 | 0.353 | 0.052 | 0.000 | 0.353 | 0.853 | partial_split |
| spotlighting | ministral-3-8b | hotpot_qa | fake_completion | 1000 | 0.608 | 0.085 | 0.000 | 0.608 | 0.860 | partial_split |
| spotlighting | ministral-3-8b | ms_marco | ignore | 1000 | 0.305 | 0.143 | 0.000 | 0.305 | 0.531 | partial_split |
| spotlighting | ministral-3-8b | ms_marco | fake_completion | 1000 | 0.515 | 0.143 | 0.000 | 0.515 | 0.722 | partial_split |
| instruction_detection | ministral-3-8b | hotpot_qa | ignore | 40 | 0.350 | 0.075 | 0.025 | 0.325 | 0.846 | backend_confound |
| instruction_detection | ministral-3-8b | hotpot_qa | fake_completion | 40 | 0.550 | 0.100 | 0.050 | 0.500 | 0.900 | backend_confound |
| instruction_detection | ministral-3-8b | ms_marco | fake_completion | 40 | 0.425 | 0.150 | 0.100 | 0.325 | 0.846 | backend_confound |

**Every one of output_filter's 6 non-inconclusive cells is
`backend_confound`** (`hf_nodef`-vs-`hf_def` never reaches significance:
p_holm = 1.0 in 5/6, 0.06 in the sixth), with 76–100% of the total
reduction attributable to the backend switch. **All 9 of spotlighting's
non-inconclusive cells have a significant `hf_nodef`-vs-`hf_def` gap** —
the defense genuinely does something — but 6 of those 9 (all
ministral-3-8b and phi-4-mini/ms_marco cells) are `partial_split`: the
backend switch already explains 50–86% of the *original* vllm-baseline
reduction before spotlighting's own base64 encoding contributes anything
further. Only qwen3-8b's 3 significant spotlighting cells show no backend
contribution at all (`backend_frac` ≈ 0) — model-specific, consistent
with qwen3-8b's no-defense `hf` baseline tracking its `vllm` baseline
closely everywhere in this dataset, unlike ministral-3-8b and phi-4-mini.

**instruction_detection's 3 non-inconclusive cells (its only cells with
enough signal to reach significance at all) are all `backend_confound`,
the same pattern as output_filter** — `hf_nodef`-vs-`hf_def` never
reaches significance (p_holm = 1.0 in all three), and 85–90% of the total
reduction is attributable to the backend switch, all on `ministral-3-8b`
— the same model driving output_filter's and spotlighting's
`ministral-3-8b`/backend-confound cells throughout this dataset.

### Corrected master table (backend-isolated `hf_nodef` vs `hf_def`)

The methodologically correct comparator for output_filter,
spotlighting, and instruction_detection/injection, now that the confound
above is confirmed for output_filter and instruction_detection and
partially present for spotlighting: `hf`-no-defense baseline vs
`hf`-defended, the same items, McNemar + Holm-Bonferroni re-applied as
three fresh 19-cell / 16-cell / 16-cell families (distinct from the
per-cell 3-comparison family above — this one asks a different question,
"is the backend-isolated defense effect significant across this
defense's cells," the same question the *original* master table asked
with the wrong baseline). This supersedes the output_filter, spotlighting,
and instruction_detection/injection rows of the original master table
below; those rows are kept for historical reference (see Methodology note
on this correction) but should not be read as defense effects without
this correction applied.

**output_filter/injection: 0 of 19 cells significant** (down from 6/19
under the original `vllm`-baseline comparator) — every cell's `p_holm =
1.0`. The full corrected table (all 19 cells, all non-significant) is
available via `python -m scripts.analyze_phase3_defense_stats
--dump-json <path>` under `corrected_backend_isolated_table.output_filter`
rather than repeated here in full, since every row's verdict is identical
(`ns`).

**spotlighting/injection: 9 of 16 cells still significant** — the same 9
cells identified above, with backend-isolated (not `vllm`-baseline)
McNemar p-values:

| Model | Corpus | Template | n | hf-nodef ASR | hf-def ASR | Reduction | p (Holm) | Sig? |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b | hotpot_qa | fake_completion | 1000 | 0.117 | 0.000 | 0.117 | 1.57e-34 | Y |
| qwen3-8b | ms_marco | ignore | 1000 | 0.034 | 0.000 | 0.034 | 1.16e-09 | Y |
| qwen3-8b | ms_marco | fake_completion | 1000 | 0.140 | 0.000 | 0.140 | 2.01e-41 | Y |
| phi-4-mini | ms_marco | ignore | 1000 | 0.016 | 0.000 | 0.016 | 2.44e-04 | Y |
| phi-4-mini | ms_marco | fake_completion | 1000 | 0.027 | 0.000 | 0.027 | 1.34e-07 | Y |
| ministral-3-8b | hotpot_qa | ignore | 1000 | 0.052 | 0.000 | 0.052 | 4.89e-15 | Y |
| ministral-3-8b | hotpot_qa | fake_completion | 1000 | 0.085 | 0.000 | 0.085 | 6.20e-25 | Y |
| ministral-3-8b | ms_marco | ignore | 1000 | 0.143 | 0.000 | 0.143 | 2.87e-42 | Y |
| ministral-3-8b | ms_marco | fake_completion | 1000 | 0.143 | 0.000 | 0.143 | 2.87e-42 | Y |

The corrected reduction magnitudes are consistently smaller than the
original table's (e.g. ministral-3-8b/hotpot_qa/fake_completion: 0.608 →
0.000 originally reported as a 0.608 reduction; backend-isolated it's
0.085 → 0.000, a 0.085 reduction), but the significance verdict is
unchanged for all 9 — spotlighting's effect is real, just smaller than
the uncorrected table implied for the cells where the backend confound
also applies.

**instruction_detection/injection: 0 of 16 cells significant** (down from
3/16 under the original `vllm`-baseline comparator) — every cell's
`p_holm = 1.0`. All 3 originally-significant cells (`ministral-3-8b`,
both corpora) drop out entirely once the comparator is the backend-matched
`hf` no-defense baseline instead of `vllm` — the same fate as
output_filter's 6, not spotlighting's 9. The full corrected table (all 16
cells, all non-significant) is available via `python -m
scripts.analyze_phase3_defense_stats --dump-json <path>` under
`corrected_backend_isolated_table.instruction_detection`.

**PoisonedRAG needed no correction — it was never confounded.** Unlike
injection, whose Phase 2 baseline ran on `vllm` (`phase2_injection_results/
attack_raw_*_vllm.jsonl`), Phase 2's PoisonedRAG baseline
(`phase2_poisonedrag_results/poison_raw_*_hf.jsonl`) and Phase 2's
Crescendo baseline (`phase2_crescendo_results/crescendo_raw_*_hf.jsonl`)
both already ran on the `hf` backend — confirmed via `git log
--diff-filter=A --name-only` on both directories, which shows every
`poison_raw_*`/`crescendo_raw_*` file ever committed was named `_hf`, never
`_vllm` (`evaluation/result_paths.py`'s `poison_result_file_paths`/
`crescendo_result_file_paths` bake the actual `INFERENCE_ENGINE` used into
the filename, so this is a direct read of what backend each run used, not
an inference). PoisonedRAG's and Crescendo's defended Phase 3 runs also
used `hf` (`phase3_defense_results/poison_raw_*_hf_defense-*.jsonl`,
`crescendo_raw_*_hf_defense-*.jsonl`) — same backend on both sides of the
comparison, for both attacks, from the start. The PoisonedRAG rows of the
master table below were backend-clean before this session's follow-up
ever began; no new `RAG_DEFENSE=none` PoisonedRAG sweep was needed, and
none was run.

---

## Mechanism attribution

### `output_filter` / injection — guard-caught fraction of blocked attacks

| Model | Corpus | Template | n blocked (baseline-success → defended-fail) | guard flagged=True | frac guard-caught |
|---|---|---|---|---|---|
| llama-3.1-8b | hotpot_qa | ignore | 0 | 0 | n/a (0 blocked) |
| llama-3.1-8b | hotpot_qa | fake_completion | 0 | 0 | n/a (0 blocked) |
| llama-3.1-8b | ms_marco | ignore | 2 | 0 | 0.000 |
| llama-3.1-8b | ms_marco | fake_completion | 0 | 0 | n/a (0 blocked) |
| qwen3-8b | hotpot_qa | naive | 1 | 0 | 0.000 |
| qwen3-8b | hotpot_qa | escape_char | 0 | 0 | n/a (0 blocked) |
| qwen3-8b | hotpot_qa | ignore | 0 | 0 | n/a (0 blocked) |
| qwen3-8b | hotpot_qa | fake_completion | 3 | 0 | 0.000 |
| qwen3-8b | hotpot_qa | combined | 0 | 0 | n/a (0 blocked) |
| qwen3-8b | ms_marco | ignore | 2 | 1 | 0.500 |
| qwen3-8b | ms_marco | fake_completion | 4 | 0 | 0.000 |
| phi-4-mini | hotpot_qa | ignore | 0 | 0 | n/a (0 blocked) |
| phi-4-mini | hotpot_qa | fake_completion | 0 | 0 | n/a (0 blocked) |
| phi-4-mini | ms_marco | ignore | 27 | 1 | 0.037 |
| phi-4-mini | ms_marco | fake_completion | 81 | 3 | 0.037 |
| ministral-3-8b | hotpot_qa | ignore | 303 | 1 | 0.003 |
| ministral-3-8b | hotpot_qa | fake_completion | 525 | 1 | 0.002 |
| ministral-3-8b | ms_marco | ignore | 180 | 2 | 0.011 |
| ministral-3-8b | ms_marco | fake_completion | 383 | 3 | 0.008 |

**12 guard-catches out of 1,511 blocked items (0.8%).** The cells with the
largest, most statistically significant ASR reductions (ministral-3-8b:
303–525 blocked items each) have the *lowest* guard-catch fractions
(0.2–1.1%) — the exact opposite of what "the guard is doing the defending"
would predict. This is the direct evidence for headline finding 2: treat
output_filter's injection ASR-reduction numbers as real but of unknown
mechanism, most plausibly dominated by the `hf`/`vllm` backend confound
(Methodology item (f)), not guard action.

### `output_filter` / PoisonedRAG — the architectural mismatch, in full

| Model | Corpus | n responses | Guard flag rate (ALL responses) | n blocked | guard-caught frac |
|---|---|---|---|---|---|
| llama-3.1-8b | hotpot_qa | 15 | 0.0000 | 1 | 0.000 |
| llama-3.1-8b | ms_marco | 20 | 0.0000 | 1 | 0.000 |
| qwen3-8b | hotpot_qa | 20 | 0.0000 | 3 | 0.000 |
| qwen3-8b | ms_marco | 20 | 0.0000 | 1 | 0.000 |
| phi-4-mini | hotpot_qa | 19 | 0.0000 | 3 | 0.000 |
| phi-4-mini | ms_marco | 20 | 0.0000 | 0 | n/a (0 blocked) |
| ministral-3-8b | hotpot_qa | 19 | 0.0000 | 4 | 0.000 |
| ministral-3-8b | ms_marco | 19 | 0.0000 | 0 | n/a (0 blocked) |

**Guard flag rate is exactly 0.0000 on every single one of the 160 scored
responses**, across every model and corpus. Every apparent ASR change in
the `poisonedrag`/`output_filter` rows of the master table above is
therefore attributable entirely to something other than the guard (most
plausibly the same `hf`/`vllm` backend confound, or ordinary generation
variance) — the guard never once intervened. This is the strongest,
cleanest finding in the whole analysis: a Llama-Guard-style safety
classifier structurally cannot defend against PoisonedRAG, because
PoisonedRAG's attack surface (factually wrong but perfectly "safe-sounding"
answers) is orthogonal to what a safety classifier screens for.

### `instruction_detection` / injection — resolved via a rerun that captures the per-passage log

Originally reported as a genuine data gap (see the code citations this
section used to make, still accurate as history:
`evaluation/run_attack_injection.py:108–112`'s `detect_injection(text)`
call computed a per-passage `DetectionResult` and discarded it inline,
never persisting it). This session's fix wires
`defenses/instruction_detection.py`'s `log_passage_detection_event` into
that same call site, and a full rerun of all 16 real cells (4 models × 2
corpora × `{ignore, fake_completion}`, n=40 each) produced
`instruction_detection_log_attack_{model}_{corpus}.jsonl` — one row per
item with a `flagged`/`score`/`label` triple per retrieved passage. The
rerun reproduced byte-identical `attack_success`/ASR numbers to what was
already committed (verified via direct diff against the prior commit, not
assumed — deterministic generation, `do_sample=False`), confirming the
new log is purely additive, not a correction to any existing number.

Of items **blocked** by instruction_detection (vllm Phase2 baseline
succeeded, defended run failed — same definition used for output_filter's
mechanism tables above), the fraction with **at least one retrieved
passage actually flagged=True** (i.e. context was genuinely stripped
pre-generation) versus **zero passages flagged** (the model resisted the
attack on its own, the classifier never fired for that item):

| Model | Corpus | n blocked | n ≥1 passage flagged | frac guard-caught |
|---|---|---|---|---|
| llama-3.1-8b | hotpot_qa | 0 | 0 | n/a (0 blocked) |
| llama-3.1-8b | ms_marco | 0 | 0 | n/a (0 blocked) |
| qwen3-8b | hotpot_qa | 1 | 1 | 1.000 |
| qwen3-8b | ms_marco | 3 | 2 | 0.667 |
| phi-4-mini | hotpot_qa | 0 | 0 | n/a (0 blocked) |
| phi-4-mini | ms_marco | 3 | 0 | 0.000 |
| ministral-3-8b | hotpot_qa | 33 | 4 | 0.121 |
| ministral-3-8b | ms_marco | 15 | 4 | 0.267 |

**Overall: 11 of 55 blocked items (20.0%) had at least one passage
actually flagged and stripped**, against the original `vllm`-baseline
definition of "blocked." This is real mechanism evidence, and notably
higher than output_filter's 0.8% guard-catch rate (finding 2 above) —
instruction_detection's DeBERTa classifier does meaningfully contribute to
some blocks, unlike output_filter's safety guard, which barely fires at
all. Still, the majority of blocks under this definition (80%) have zero
passages flagged.

**Resolved, backend-corrected version of the same question.** A second
follow-up re-ran this exact analysis with "blocked" redefined against the
backend-matched `hf` no-defense baseline (Task B's `attack_raw_*_hf.jsonl`,
restricted to the matching 40 items) instead of `vllm` — the same
correction "Backend-confound isolation (Task 2)" above applies to the ASR
numbers, applied here to the mechanism question:

| Model | Corpus | n blocked (backend-corrected) | n ≥1 passage flagged | frac guard-caught |
|---|---|---|---|---|
| llama-3.1-8b | hotpot_qa | 0 | 0 | n/a (0 blocked) |
| llama-3.1-8b | ms_marco | 0 | 0 | n/a (0 blocked) |
| qwen3-8b | hotpot_qa | 1 | 1 | 1.000 |
| qwen3-8b | ms_marco | 2 | 2 | 1.000 |
| phi-4-mini | hotpot_qa | 0 | 0 | n/a (0 blocked) |
| phi-4-mini | ms_marco | 0 | 0 | n/a (0 blocked) |
| ministral-3-8b | hotpot_qa | 4 | 4 | 1.000 |
| ministral-3-8b | ms_marco | 3 | 3 | 1.000 |

**10 of 55 originally "blocked" items survive the backend correction, and
all 10 (100%) have a passage actually flagged.** The other 45 items
(mostly `ministral-3-8b`'s 33+15=48 raw blocks, reduced to 4+3=7 real
ones) were never really blocked by the classifier at all — the model's
answer already changed between `vllm` and the undefended `hf` backend,
independent of any defense, and the original vllm-baseline definition of
"blocked" was silently counting those as instruction_detection's doing.
**This answers the cross-reference question directly: the confound does
not wash out the classifier's mechanism signal — it inflates the
denominator.** Every genuinely defense-driven block in this dataset has a
flagged passage behind it; the 20.0% figure understated the classifier's
reliability by counting ~4.5x too many items as "blocked." This is also
exactly consistent with headline finding 2's resolution of
instruction_detection's ASR numbers: 0/16 cells reach significance
precisely because the *real* (backend-corrected) block counts are this
small (n=40 per cell, only a handful of genuine blocks per cell) — not
because the classifier is unreliable when it does fire.

### `crescendo` / `output_filter` — guard-would-have-intervened framing (mechanism-only, not an ASR result)

**This subsection is deliberately separate from the ASR-reduction
findings above and reports no ASR-reduction number.** Because Task 2
confirmed the guard never actually intervenes for Crescendo (finding 4),
there is no defended condition here to measure a reduction from — the
master table above correctly excludes this cell entirely. What follows
is detection-capability data only: it characterizes the guard's raw
ability to notice the attack, independent of whether that noticing was
ever allowed to change the outcome. The design deliberately preserved
the attack's real, unfiltered trajectory (see `run_crescendo.py:184-194`)
specifically so this counterfactual mechanism data could be captured for
study, at the cost of not producing a defended-condition ASR number — a
genuine trade-off, not an oversight.

The question asked here is counterfactual: *of the conversations that
succeeded unfiltered, in what fraction did the guard flag at least one
turn before the end* — i.e. how often would this guard have had a chance
to stop a real attack, if its verdict had actually been wired to block
delivery.

| Model | n scored | n succeeded (unfiltered) | n where guard would've flagged ≥1 turn | frac of successes guard would've caught |
|---|---|---|---|---|
| llama-3.1-8b | 19 | 11 | 8 | 0.727 |
| qwen3-8b | 19 | 11 | 9 | 0.818 |
| phi-4-mini | 20 | 9 | 6 | 0.667 |
| ministral-3-8b | 18 | 15 | 11 | 0.733 |

67–82% of successful Crescendo attacks had at least one turn the guard
flagged as unsafe before the conversation concluded. This is a genuinely
encouraging number for a *future, actually-wired* output_filter-for-
Crescendo defense — unlike PoisonedRAG, Crescendo's attack surface
(multi-turn escalation toward unsafe content) is exactly what a safety
classifier is built to catch. But it is a counterfactual estimate, not a
measured effect: no live blocking was ever applied in this sweep.

**Future work**: wiring `output_filter` to actually enforce on Crescendo
— substituting `REFUSAL_MARKER` into the live conversation history when
the guard flags a turn, the same way injection's and PoisonedRAG's
runners already do, rather than discarding the substitution — would
convert this 67–82% counterfactual-catch rate into a real, measurable
ASR reduction; this is a well-motivated, currently unexplored next step
rather than a gap being hidden.

---

## Utility preservation

F1(clean) per (attack, model, corpus, defense) — Phase 3 defended cells
averaged across the applicable injection templates — versus the matching
Phase 1 baseline (no attack, no defense, `phase1_results_complete/`).

| Attack | Model | Corpus | Defense | Phase1 F1 (no attack, no defense) | Defended F1 (mean over templates) | Delta |
|---|---|---|---|---|---|---|
| injection | llama-3.1-8b | hotpot_qa | instruction_detection | 0.6058 | 0.6357 | +0.0299 |
| injection | llama-3.1-8b | ms_marco | instruction_detection | 0.2340 | 0.2840 | +0.0500 |
| injection | ministral-3-8b | hotpot_qa | instruction_detection | 0.6392 | 0.6268 | -0.0124 |
| injection | ministral-3-8b | ms_marco | instruction_detection | 0.2572 | 0.2782 | +0.0210 |
| injection | phi-4-mini | hotpot_qa | instruction_detection | 0.4600 | 0.3981 | -0.0620 |
| injection | phi-4-mini | ms_marco | instruction_detection | 0.2539 | 0.1985 | -0.0555 |
| injection | qwen3-8b | hotpot_qa | instruction_detection | 0.6363 | 0.6214 | -0.0149 |
| injection | qwen3-8b | ms_marco | instruction_detection | 0.2755 | 0.3079 | +0.0324 |
| injection | llama-3.1-8b | hotpot_qa | output_filter | 0.6058 | 0.6077 | +0.0019 |
| injection | llama-3.1-8b | ms_marco | output_filter | 0.2340 | 0.2349 | +0.0009 |
| injection | ministral-3-8b | hotpot_qa | output_filter | 0.6392 | 0.6377 | -0.0015 |
| injection | ministral-3-8b | ms_marco | output_filter | 0.2572 | 0.2367 | -0.0206 |
| injection | phi-4-mini | hotpot_qa | output_filter | 0.4600 | 0.4438 | -0.0163 |
| injection | phi-4-mini | ms_marco | output_filter | 0.2539 | 0.1935 | -0.0604 |
| injection | qwen3-8b | hotpot_qa | output_filter | 0.6363 | 0.6212 | -0.0152 |
| injection | qwen3-8b | ms_marco | output_filter | 0.2755 | 0.2581 | -0.0175 |
| injection | llama-3.1-8b | hotpot_qa | spotlighting | 0.6058 | 0.2433 | -0.3626 |
| injection | llama-3.1-8b | ms_marco | spotlighting | 0.2340 | 0.0683 | -0.1657 |
| injection | ministral-3-8b | hotpot_qa | spotlighting | 0.6392 | 0.2340 | -0.4052 |
| injection | ministral-3-8b | ms_marco | spotlighting | 0.2572 | 0.1088 | -0.1484 |
| injection | phi-4-mini | hotpot_qa | spotlighting | 0.4600 | 0.0183 | -0.4418 |
| injection | phi-4-mini | ms_marco | spotlighting | 0.2539 | 0.0800 | -0.1739 |
| injection | qwen3-8b | hotpot_qa | spotlighting | 0.6363 | 0.2213 | -0.4151 |
| injection | qwen3-8b | ms_marco | spotlighting | 0.2755 | 0.1230 | -0.1525 |
| poisonedrag | llama-3.1-8b | hotpot_qa | instruction_detection | 0.6058 | 0.1856 | -0.4202 |
| poisonedrag | llama-3.1-8b | ms_marco | instruction_detection | 0.2340 | 0.0606 | -0.1734 |
| poisonedrag | ministral-3-8b | hotpot_qa | instruction_detection | 0.6392 | 0.1574 | -0.4818 |
| poisonedrag | ministral-3-8b | ms_marco | instruction_detection | 0.2572 | 0.0839 | -0.1733 |
| poisonedrag | phi-4-mini | hotpot_qa | instruction_detection | 0.4600 | 0.1300 | -0.3300 |
| poisonedrag | phi-4-mini | ms_marco | instruction_detection | 0.2539 | 0.0877 | -0.1662 |
| poisonedrag | qwen3-8b | hotpot_qa | instruction_detection | 0.6363 | 0.1856 | -0.4507 |
| poisonedrag | qwen3-8b | ms_marco | instruction_detection | 0.2755 | 0.0622 | -0.2133 |
| poisonedrag | llama-3.1-8b | hotpot_qa | output_filter | 0.6058 | 0.0786 | -0.5272 |
| poisonedrag | llama-3.1-8b | ms_marco | output_filter | 0.2340 | 0.0821 | -0.1519 |
| poisonedrag | ministral-3-8b | hotpot_qa | output_filter | 0.6392 | 0.2596 | -0.3796 |
| poisonedrag | ministral-3-8b | ms_marco | output_filter | 0.2572 | 0.1698 | -0.0874 |
| poisonedrag | phi-4-mini | hotpot_qa | output_filter | 0.4600 | 0.0710 | -0.3890 |
| poisonedrag | phi-4-mini | ms_marco | output_filter | 0.2539 | 0.1440 | -0.1099 |
| poisonedrag | qwen3-8b | hotpot_qa | output_filter | 0.6363 | 0.1373 | -0.4990 |
| poisonedrag | qwen3-8b | ms_marco | output_filter | 0.2755 | 0.1215 | -0.1540 |
| poisonedrag | llama-3.1-8b | hotpot_qa | spotlighting | 0.6058 | 0.1481 | -0.4577 |
| poisonedrag | llama-3.1-8b | ms_marco | spotlighting | 0.2340 | 0.0852 | -0.1488 |
| poisonedrag | ministral-3-8b | hotpot_qa | spotlighting | 0.6392 | 0.1389 | -0.5003 |
| poisonedrag | ministral-3-8b | ms_marco | spotlighting | 0.2572 | 0.1005 | -0.1567 |
| poisonedrag | phi-4-mini | hotpot_qa | spotlighting | 0.4600 | 0.0669 | -0.3931 |
| poisonedrag | phi-4-mini | ms_marco | spotlighting | 0.2539 | 0.1547 | -0.0992 |
| poisonedrag | qwen3-8b | hotpot_qa | spotlighting | 0.6363 | 0.0370 | -0.5993 |
| poisonedrag | qwen3-8b | ms_marco | spotlighting | 0.2755 | 0.1769 | -0.0986 |

**Mean F1 delta by (attack, defense), averaged across the 8 model×corpus
cells:**

| Attack | Defense | Mean ΔF1 |
|---|---|---|
| injection | instruction_detection | −0.0014 (negligible) |
| injection | output_filter | −0.0161 (small) |
| injection | spotlighting | **−0.2831 (severe)** |
| poisonedrag | instruction_detection | −0.3011 (severe) |
| poisonedrag | output_filter | −0.2873 (severe) |
| poisonedrag | spotlighting | −0.3067 (severe) |

For injection, `instruction_detection` is essentially free on utility
(it only drops passages an off-the-shelf classifier actually flags as
injection, leaving most legitimate context intact) while `spotlighting`'s
base64 encoding devastates it — this is the trade-off finding 3 warns
about. For PoisonedRAG, all three defenses cost roughly the same, large
utility hit (~−0.29 to −0.31): none of them repair the underlying
retrieval corruption, they only change what the model does with an
already-poisoned context (refuse, hedge, or fail to extract an answer),
which reads as "wrong answer" either way against the gold F1 metric.

---

## Methodology & Limitations

**(a) Corrected cell counts, and the 6 excluded smoke-test fragments.**
Phase 3 coverage across the three injection defenses is asymmetric, not
symmetric as originally assumed: `instruction_detection` and
`spotlighting` each cover 16 real cells (2 templates × 2 corpora × 4
models, using only `ignore`/`fake_completion`), while `output_filter`
covers 19 (the same 16, plus 3 extra templates —
`combined`/`escape_char`/`naive` — that completed only for
qwen3-8b/hotpot_qa). Real injection total: **51 cells**, not 24. Six
additional files exist at `qwen3-8b_hotpot_qa_{combined,escape_char,naive}
_hf_defense-{instruction_detection,spotlighting}.jsonl` at exactly n=3 —
an order of magnitude below the real n=40/n=1000 pattern, confirmed as
infra smoke-test leftovers and excluded from every table above (the
analysis script excludes any file below 50% of its defense's expected n,
not by filename).

**(b) The n asymmetry within injection (40 vs. 1000).**
`instruction_detection` runs a real transformer classifier
(protectai/deberta-v3-base-prompt-injection-v2) per retrieved passage on
CPU — real-hardware timing (documented in
`defenses/instruction_detection.py`'s module docstring: a single
RAG_SAMPLE_N=1 item took 4m27s wall-clock even after fixing a CPU
thread-oversubscription bug) made the full n=1000 sweep impractical
within this project's compute budget, so it was capped at n=40.
`spotlighting` and `output_filter` have no comparable per-item classifier
cost and ran the full n=1000.

**(c) PoisonedRAG's matched-subset reduction (n=14–18, not the full
n=15–20).** Phase 2's original PoisonedRAG baseline sampled n=90–96
per model×corpus; Phase 3's defended cells drew a smaller n=15–20 subset
from the same poisoned-context pool but included 1–3 items per cell that
are genuinely new (absent from Phase 2's baseline for that model+corpus,
confirmed by exact `question`-text lookup, not assumed). These items are
excluded from each cell's paired test — every cell above still had a
non-empty matched subset (n=14–18), so no cell fell back to the unpaired
Fisher's-exact path. A fresh undefended baseline pass on the small set of
genuinely-new items (shared across each model×corpus's 3 defense cells,
so ~20 unique generations total, not 3× that) would complete full-n
pairing everywhere; not done in this pass.

**(d) The Crescendo behavior-pool mismatch (9/20 matched, not "same seeds
by design").** Going into this analysis it was assumed Phase 3 reused
Phase 2's exact behavior seeds. Verified against the raw data instead:
only 9 of Phase 3's 20 behaviors per model appear anywhere in Phase 2's
n=100 baseline (same 9 matched, same 11 missing, consistently across all
4 models — a real pool mismatch, not per-model sampling noise). Usable
paired n for every Crescendo cell is **9**, not 20; this materially widens
the confidence intervals on the diagnostic-only numbers the analysis
script prints for this cell (e.g. ministral-3-8b's CI is [−0.778, 0.111]
on n=9) and would be the main reason all 4 Crescendo cells came out
non-significant, if they were being tested for significance at all —
which, per finding 4, they are not: they are excluded from the master
ASR-reduction table and its Holm-correction family entirely.

**(e) Crescendo/`output_filter`'s observational-only design (finding 4,
restated with the code citations).** `run_crescendo.py:184–194`'s
docstring states the defense measures what a guard *would* have blocked
"without itself changing the attack's trajectory." Mechanically:
`run_crescendo.py:248`, `_, guard_record = apply_output_filter(target_reply)`
discards the filtered text; `:256–257` stores the real, unfiltered
`target_reply` in the conversation history; the judge at `:370`/`:442`
scores that same unfiltered `conversation`. No intervention is ever
possible in this design — this is why the 4 Crescendo/output_filter
cells are excluded from the master ASR-reduction table above rather than
included with a caveat marker: reporting an ASR "reduction" for a
condition with no live intervention would misstate what was measured,
not just under-caveat it. The analysis script still computes and prints
these raw numbers, but under a header that says explicitly they are not
an ASR-reduction-family result (see `scripts/analyze_phase3_defense_stats.py::main`).

**(f) The backend mismatch (`_hf` vs `_vllm`) — CONFIRMED for
output_filter and instruction_detection, real but partial for
spotlighting, and never present at all for PoisonedRAG or Crescendo.
Resolved with real backend-matched baselines throughout, not left as a
suspected or partially-checked confound anywhere.** Only **injection's**
Phase 2 baseline ran on the `vllm` backend
(`phase2_injection_results/attack_raw_*_vllm.jsonl`); every Phase 3
defended cell ran on `hf` (transformers) instead — same model weights,
different inference stack. (An earlier version of this document claimed
all three attacks' Phase 2 baselines ran on `vllm` — that was wrong for
PoisonedRAG and Crescendo, corrected below; `evaluation/result_paths.py`'s
file-naming functions bake the actual engine used into every result
filename, so this is directly checkable per attack, not something to
generalize across attacks.) This was originally flagged as a methodology
footnote for injection and later strengthened to "likely" by
output_filter's 0.8% guard-catch rate. A follow-up ran the missing third
condition directly (a backend-matched, no-defense `hf` baseline — see
"Backend-confound isolation (Task 2)" above for the full per-cell table)
instead of continuing to infer the confound indirectly: **for
output_filter/injection, the confound is now directly confirmed — 0 of 19
cells retain a significant defense effect once the `hf`/`vllm` backend
switch is controlled for (down from 6/19 under the original comparator).
For spotlighting/injection, the confound is real but only partial — 6 of
9 significant cells have the backend switch explaining 50–86% of the
original reduction, but spotlighting still shows a significant,
backend-isolated effect in all 9 cells.**

A second follow-up closed the two gaps this item used to flag as open.
**`instruction_detection`/injection is now also directly confirmed,
same pattern as output_filter**: its n=40 backend-matched no-defense
baseline turned out to be the first 40 rows of Task B's already-committed
n=1000 no-defense file (verified byte-identical, in order, by
`scripts/verify_phase3_instruction_detection_baseline_items.py` against
real committed data — no new pod sweep needed), and the same 3-way test
finds 0 of 16 cells retain a significant effect once corrected (down from
3/16), all 3 originally-significant cells being `backend_confound`. The
mechanism-attribution cross-reference (see "`instruction_detection` /
injection" above) confirms this from the log data directly: once
"blocked" is redefined against the backend-matched baseline, only 10 of
the original 55 blocked items are real, and all 10 (100%) have a flagged
passage — the confound inflated the denominator, it didn't obscure a
real classifier failure.

**PoisonedRAG's and Crescendo's backend confound never existed —
confirmed, not just newly checked.** Unlike injection, Phase 2's
PoisonedRAG baseline (`phase2_poisonedrag_results/poison_raw_*_hf.jsonl`)
and Crescendo baseline (`phase2_crescendo_results/crescendo_raw_*_hf.jsonl`)
both already ran on `hf` — the same backend Phase 3's defended runs for
both attacks use. Confirmed via `git log --all --diff-filter=A
--name-only` on both result directories: every file ever committed to
either is named `_hf`, and no `_vllm` variant was ever committed for
either attack. The PoisonedRAG rows in the master table above (and the
0/160 = 0% guard flag rate in the PoisonedRAG mechanism section) were
backend-clean from the start — no correction needed, and none applied.

**(g) Fisher's exact vs. McNemar.** No cell in this dataset used the
unpaired Fisher's-exact fallback — every one of the 79 cells had a
non-empty matched-item baseline subset (confirmed per-cell, see (c)/(d)
above), so all 79 rows in the master table use `mcnemar_exact`. The
fallback path is implemented in `scripts/analyze_phase3_defense_stats.py`
for any future data that lacks a matched baseline, and is clearly labeled
`fisher_exact_unpaired` in the `test_used` column whenever it fires.

**(h) NIM generator refusals during PoisonedRAG poison-context
generation.** Checked against `PHASE2_POISONEDRAG_INSIGHTS.md`'s own
attempt-level failure log (the only place in this repo this was
quantified). Found: **8 of 111 total attempt-level generation failures
across the whole Phase 2 PoisonedRAG sweep show explicit refusal-style
text** (`"sorry"`, `"not going to"`, `"can't help"`, `"disallowed"`), and
**1 of 14 permanent (unrecoverable) item skips** was a genuine content
refusal (`"I'm sorry, but I can't help with that."`, on a `hotpot_qa`
question about a film actor). This confirms real, quantified refusal
behavior at a rate consistent with "occurred 4+ times" (8 ≥ 4). **Not
confirmed**: an instance of the generator citing a named policy
explicitly. The two refusal quotes on record either give no reason
(`"I'm sorry, but I can't help with that."`) or cite a generic
factual-accuracy concern (`"providing incorrect steps for configuring a
database field could lead to data errors"`) — no quote in the available
log names a specific policy. Reported as not found, rather than assumed
present.

**(i) This document was revised after a targeted follow-up investigation,
not written once and left static.** The first version of this analysis
flagged the `hf`/`vllm` backend switch as a *likely* confound based on
indirect evidence (output_filter's 0.8% guard-catch rate) and reported
instruction_detection's mechanism question as unanswerable from available
data. A follow-up closed both gaps with real data instead of stronger
wording: a backend-matched no-defense baseline (Task B) let Methodology
(f) and headline finding 2 go from "likely" to a directly confirmed
verdict per cell, and a rerun with fixed per-passage logging (Task C)
answered the instruction_detection mechanism question this document
previously could only flag as missing. The master ASR-reduction table's
`output_filter` and `spotlighting` injection rows are superseded by the
"Backend-confound isolation (Task 2)" section's corrected numbers; they
are kept in place, unedited, for historical reference rather than
silently rewritten, with pointers added at both the table's summary line
and Methodology (f) so a reader lands on the correction either way.

**A second follow-up (2026-09-16) closed the two gaps this document
still flagged as open after the first revision** — instruction_detection's
own exposure to the backend confound, and PoisonedRAG's backend confound.
Both were resolved with real data and zero new GPU time, not by launching
the pod sweeps originally assumed necessary: a dry-run item-selection
check (`scripts/verify_phase3_instruction_detection_baseline_items.py`)
found instruction_detection's n=40 backend-matched baseline already
existed as a 40-row prefix of Task B's committed n=1000 file, and a git
history check on `phase2_poisonedrag_results/` and
`phase2_crescendo_results/` found neither attack's Phase 2 baseline ever
ran on `vllm` in the first place — an earlier version of this document's
Methodology (f) incorrectly claimed all three attacks' baselines ran on
`vllm`; that claim is corrected in place there, not silently. This
document now carries zero cells or mechanism findings still marked
provisional, not-yet-backend-verified, or open — every gap flagged by an
earlier revision is either directly confirmed or directly resolved as
never having existed.

---

## Appendix: excluded files (not real cells)

| Model | Corpus | Template | Defense | n | Reason |
|---|---|---|---|---|---|
| qwen3-8b | hotpot_qa | combined | instruction_detection | 3 | smoke-test fragment |
| qwen3-8b | hotpot_qa | escape_char | instruction_detection | 3 | smoke-test fragment |
| qwen3-8b | hotpot_qa | naive | instruction_detection | 3 | smoke-test fragment |
| qwen3-8b | hotpot_qa | combined | spotlighting | 3 | smoke-test fragment |
| qwen3-8b | hotpot_qa | escape_char | spotlighting | 3 | smoke-test fragment |
| qwen3-8b | hotpot_qa | naive | spotlighting | 3 | smoke-test fragment |
