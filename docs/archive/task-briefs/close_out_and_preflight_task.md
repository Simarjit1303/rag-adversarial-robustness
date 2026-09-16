# Task: Close out Phase 1 baseline fix + pre-flight checks before PoisonedRAG

Run these in order. Each section depends on the one before it being genuinely correct, not just started — confirm each before moving to the next.

## 1. Verify and overwrite the corrected baseline

- [ ] The Phase 1 baseline rerun is complete and downloaded to `phase1_baseline_rerun/` in the repo root: 16 files (8 raw JSONL + 8 summary CSV, hotpot_qa/ms_marco × all 4 models, generated on the `rerun-phase1-baseline-context-fix` branch's image, commit `6c31521`).
- [ ] Verify each new file's schema/naming matches what's already in `phase1_results_complete/` before touching anything.
- [ ] Overwrite the matching 8 raw + 8 summary files in `phase1_results_complete/` with these corrected versions. Leave the 4 nq_open files completely untouched — nq_open remains separately unfixable and excluded from real sweeps regardless.
- [ ] Delete the now-redundant `phase1_baseline_rerun/` folder once the overwrite is confirmed correct.
- [ ] Run the full test suite. Confirm it still passes before committing.
- [ ] Commit this on its own, clearly-scoped commit — do not mix with anything below.

## 2. Recompute the injection insights file against the corrected baseline

- [ ] Recompute `PHASE2_INJECTION_INSIGHTS.md`'s F1-collapse comparison using the corrected `phase1_results_complete/` baseline.
- [ ] Rewrite the nq_open-vs-ms_marco section per the reasoning already worked out: nq_open is not a valid comparison point for anything (leaks unfixably by construction), so it should be excluded from difficulty comparisons entirely, not just flagged provisional. State plainly that nq_open's F1 is a separate, documented-limitation number.
- [ ] Check whether the leaked-vs-clean F1 difference for hotpot_qa/ms_marco is actually small across all 8 cells, the way a few manually-spot-checked cells suggested (llama-3.1-8b hotpot_qa: 0.6115→0.6035; llama-3.1-8b ms_marco: 0.2191→0.2212 — both under 0.01 difference). If this pattern holds broadly, say so explicitly in the insights file: the leak was real and needed fixing for correctness and for a valid attack-vs-baseline comparison, but its practical F1 impact on hotpot_qa/ms_marco specifically was apparently minor — unlike nq_open's catastrophic near-ceiling inflation. Do not let the write-up imply a dramatic swing across the board if the actual numbers don't show one; report what the data actually says, cell by cell if the pattern isn't uniform.
- [ ] Confirm the provisional ministral-3-8b flag and the substring-match-ASR-scoring methodology section both survive this rewrite untouched — those are independent findings from a different investigation and shouldn't be disturbed by the baseline correction.
- [ ] Re-run the test suite. Commit on its own commit, separate from part 1's file-overwrite commit.

## 3. CUDA / recurring warning review

- [ ] Investigate the `deep_gemm` / `_find_cuda_home()` `AssertionError` warning — appears identically on every model, every run, all night (`Module vllm.third_party.deep_gemm was found but failed to import`). Determine: is `deep_gemm` actually used by anything in this pipeline, or does it silently no-op when unavailable (the immediately-following "Skipping CuTeDSL warmup" line suggests the latter)? If genuinely unused, check whether `CUDA_HOME` is simply unset in the Docker image and whether a one-line `ENV CUDA_HOME=...` fix is worth an image rebuild, or whether it's purely cosmetic and not worth touching.
- [ ] Review Phi-4-mini's `rope_parameters['original_max_position_embeddings']` warning — confirm it's cosmetic (already suspected) or flag if it affects correctness.
- [ ] Review the `Triton kernel JIT compilation during inference: kernel_unified_attention` latency-spike warnings — confirm whether extending warmup coverage is worth doing given actual sweep timing impact, or not worth the complexity.
- [ ] Note any other recurring warning seen across tonight's logs that hasn't already been investigated.
- [ ] Produce a short table: **warning | cause | verdict (ignore / fix now / fix later)**. Any "fix now" item goes on its own small commit, separate from everything else in this task.

## 4. Disk / storage cleanup pass

- [ ] Confirm `.github/workflows/ghcr-cleanup.yml` (built earlier, keeps last 5 GHCR images) is ready to activate — I will create the `GHCR_CLEANUP_TOKEN` PAT secret (`read:packages` + `delete:packages` scopes) myself; just confirm the workflow doesn't need anything else from me first.
- [ ] Note what to do with `baseline_rerun.log` (the failed first attempt) vs. `baseline_rerun2.log` (the successful relaunch) sitting on the pod — the pod may already be stopped; if unreachable, just note that this is a leftover-cleanup item for the next time a pod on that same volume is up, not urgent.
- [ ] Decide and state clearly: does `phase2_injection_results/` (the folder used to transfer the injection sweep's downloaded data) get merged into the proper results location, or does it stay as a deliberate snapshot? Pick one and say why.
- [ ] Once `6c31521...` is confirmed as the current standard image (it is, per part 1 above), the older tag `634010d...` is a candidate for the GHCR cleanup workflow to prune on its next run — no manual action needed beyond confirming the workflow will catch it.
- [ ] Do NOT touch `HF_HOME` model weight caches — reused across all attacks, not worth clearing.

## Done means

All four sections complete, each on its own clearly-scoped commit, test suite green throughout, and a short summary back to me covering: confirmation of the baseline overwrite, the corrected insights file's key numbers, the warnings table, and the cleanup decisions made. Once this is done, PoisonedRAG's task brief is next.
