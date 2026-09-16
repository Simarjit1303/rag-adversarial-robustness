# Phase 2 Prep Log

Durable record of the pre-Phase-2 cleanup orchestrated per `Drafts/CLAUDE_CODE_RUNBOOK.md` against `Drafts/PRE_PHASE2_CHECKLIST.md`. Append-only, timestamped, newest at bottom.

---

## 2026-08-31 — Session start, pre-dispatch reconnaissance

Before dispatching Phase A subagents, did a manual recon pass to correct stale context in the runbook (each subagent runs isolated and needed accurate starting facts):

- **Repo state:** on `runpod-self-termination`, clean vs HEAD, up to date with origin. No stash.
- **Branch list has 4 more entries than the runbook's stack description:** `dissertation-run-readiness` (PR #5, merged), `fix-deploy-registry-credential` (PR #2, merged), `observable-persistent-run` (PR #3+#4, merged), `python314-modernization` (PR #1, merged). All pre-date and are already merged into `main` — not part of the open stack, not a concern for checklist items 1–7, but the runbook's context block didn't mention them.
- **PR #11's body text is stale.** It documents "six fixes" and says the `INFERENCE_ENGINE=vllm` path was never exercised on RunPod. But `origin/runpod-self-termination` (the same branch backing PR #11) has 7 more commits pushed since that description was written, undocumented in the PR body:
  - `7937519` — `harness/vllm_engine.py` (batched vLLM load+generate path)
  - `8a1f861` — attention backend as constructor kwarg; renames `VLLM_MAX_MODEL_LEN` → `RAG_VLLM_MAX_MODEL_LEN`, defaults to `TRITON_ATTN`
  - `e2f5c95` — `RAG_VLLM_MAX_MODEL_LEN = 13056` (measured via `compute_max_model_len.py`)
  - `8840922` — ffmpeg apt-layer fix (PR body says this was still uncommitted as of its writing — it since landed)
  - `13a9b7d` — result filename collision fix across model/corpus/engine
  - `df86d12` — 3-cell partial-success test for `verify_success()`
  - `eedce80` — extract leaked dict/list-repr answers in `clean_generation()`
  - (plus `e545bb8`, `34fc818` — MCP tooling wiring, unrelated to vLLM)
- **Untracked working-tree files found:** `baseline_raw_qwen3.jsonl`, `phase1_results_partial/`, `phase1_results_complete/` — filenames and content (spot-checked one record) match a real 4-model × 3-corpus vLLM sweep. `phase1_results_complete/` timestamps run later (Jul 28, 20:xx–23:xx) than `phase1_results_partial/` (Jul 28, 15:xx–21:xx) — looks like an interrupted run followed by a clean one. **Not yet confirmed gitignored; sitting uncommitted = data-loss risk.**
- **Flashinfer JIT fixes NOT found.** Checklist explicitly calls these out as missing. A grep for "flashinfer" across every local and remote branch only turns up Dockerfile/requirements.txt comments about Python 3.13-vs-3.14 dependency compatibility — no JIT-compile-related code anywhere in git.

**Conclusion carried into Phase A:** checklist item 1 ("find the missing work") looks substantially resolved for the vLLM engine/backend/max-len code — it's already on PR #11, just undocumented — but NOT resolved for the flashinfer JIT fixes specifically, and the actual result data is sitting untracked and at risk. Dispatching A1–A4 with this corrected context rather than the runbook's original framing.

---

## 2026-08-31 — Phase A dispatched (A1–A4, parallel, all read-only)

Dispatched 4 Explore subagents concurrently, each briefed with the corrected context above (not the runbook's stale framing).

### A4 complete — PR #8 Azure-specific surface map

Diffed `stage2-vllm-path..stage3-job-conversion`: 2 files changed, 32 insertions / 3 deletions, nothing else touched.

- **`.github/workflows/thesis-AutoDeployTrigger-*.yml`**: step name "Update Container Apps Job image" (Azure-specific), `azure/CLI@v2` action wrapper (Azure-specific), registry-set comment (Azure-specific), the `az containerapp job update --name rag-sweep-job --resource-group Master-Thesis` command block (Azure-specific — no RunPod equivalent). One 8-line explanatory comment block (lines 43–50) mixes the generic "no auto-start on push" guardrail principle with Azure-specific wording — needs a human split.
- **`README.md`**: section heading "push no longer executes anything" (generic), two prose paragraphs (99–104, 113–114) each mix the generic safety claim ("push can no longer start a paid GPU sweep", "nothing in CI calls start, deliberately") with Azure-specific nouns in the same sentence (needs a human split), a literal `az containerapp job start` code block (Azure-specific), and a line about unprovisioned `rag-sweep-job` / "Part B of Stage 3" (Azure-specific, superseded roadmap item).
- **Bottom line**: the underlying idea — "don't let a push auto-start a paid GPU sweep" — is 100% platform-neutral and already true for RunPod today (RunPod has no CI trigger at all, confirmed separately by A3). Everything else in PR #8 is Azure CLI plumbing with no RunPod equivalent. No recommendation made per instructions — this is Gate 1 material.

### A3 complete — what push-to-main currently triggers

`.github/workflows/thesis-AutoDeployTrigger-29cfe416-0f29-4508-94fb-e7a4ca191e9d.yml` is the only workflow, same path on every branch. Trigger: `on: push: branches: [main], paths: ['**']` plus `workflow_dispatch` — any push to main, any path, fires it.

**Critical finding: PR #8's fix has NOT reached `main`.** `main`'s current HEAD (`1335892`) still runs the pre-PR-8 step: `az containerapp update --name thesis --resource-group Master-Thesis --image ghcr.io/...` against the regular, always-on Container App `thesis` — which auto-activates a new revision immediately on image update. The Job-based, no-auto-start version (`az containerapp job update --name rag-sweep-job`) only exists on `runpod-self-termination`/`stage3-job-conversion`, unmerged. `git diff main runpod-self-termination -- .github/workflows/` confirms this is the only diff in the file.

**Bottom line:** pushing to `main` right now would immediately redeploy the live, paid Azure Container App — the old auto-start behavior PR #8 was written to kill. The "never merge to main" rule is not stale caution, it's live and necessary today.

RunPod confirmed fully manual: no RunPod reference anywhere in the workflow file on any branch, no CI/CD path touches it at all — pod console "Start command" override only.

### A2 complete — RAG_CORPORA fix timing

`a1b6347` committed **2026-07-27 23:33:56 +0200**. Diff confirmed to match its description exactly (wires `RAG_CORPORA` into both `data/build_index.py` and `evaluation/run_baseline.py`'s corpus loops); no later commit reverts it.

`phase1_results_complete/` (the real vLLM sweep — all files `_vllm`-suffixed, correctly distinguished from PR #11's separate HF-transformers 2026-07-28 02:44 CEST confirmation run, per the PR's own "still unverified: vLLM never exercised" text): earliest file **Jul 28 22:17**, latest **Jul 28 23:57** — fix predates sweep start by **~22.7 hours**. `phase1_results_partial/` (earlier attempt): earliest **Jul 28 20:08** — fix predates by ~20.5 hours either way.

**High confidence** the reported numbers were not produced under the buggy behavior. One honestly-flagged residual assumption: git history proves the fix was on the branch well before the sweep, but can't directly prove which exact commit SHA the RunPod pod had checked out at run time (no pod deployment log available to this repo scan) — inferred from output shape (12 distinct per-model×corpus files implies the pod also had the later `13a9b7d` filename-collision fix, i.e. ran near HEAD, not stale).

### A1 complete — flashinfer fixes + result data

**Attention-backend fix** (`TRITON_ATTN`): already committed+pushed, commit `8a1f861`. Already accounted for.

**Sampler fix**: the "second" flashinfer JIT fix is `VLLM_USE_FLASHINFER_SAMPLER=0`. Exhaustive search (git stash/reflog/fsck --unreachable across all refs, filesystem search beyond this repo, Docker build logs, shell history) found it NOWHERE in git or in code — it only ever existed as an ephemeral env var set directly in the RunPod pod session. It IS documented in prose, though: `Drafts/PHASE1_SUMMARY.md` (same folder as the checklist/runbook) states verbatim: *"`RAG_VLLM_ATTENTION_BACKEND=TRITON_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0` are workarounds, not defaults. They exist because flashinfer's JIT path needs a CUDA compiler this image doesn't have."* So this is recoverable as a precise one-line env var to re-add (e.g. Dockerfile `ENV` or alongside `RAG_VLLM_ATTENTION_BACKEND` in `harness/vllm_engine.py`) — not recoverable as "find existing code," because none exists to find.

**Result data**: `.gitignore` only covers `results/*.jsonl`/`results/*.csv` — does NOT cover `baseline_raw_qwen3.jsonl`, `phase1_results_complete/`, `phase1_results_partial/` (confirmed via `git check-ignore -v`, no matches). Genuinely untracked, no backup — real data-loss risk. `phase1_results_complete/` has all 12/12 model×corpus combos, 1000 lines each (12,000 total), spot-checked clean; one CSV (phi-4-mini/ms_marco, F1=0.3701/EM=0.17) cross-confirmed exact match against `Drafts/PHASE1_SUMMARY.md`'s reported table — this is confirmed the real, final Phase 1 sweep. `phase1_results_partial/` is missing `ministral-3-8b_ms_marco_vllm` entirely — consistent with being the superseded interrupted attempt.

**Revised inventory for checklist item 1** — of the four things it names as missing, only one is genuinely still missing as *code*: the flashinfer sampler env var (now precisely known, not just "somewhere"). Everything else is already on PR #11 or sitting untracked in the working tree.

---

## Phase A complete — synthesis handed to user for Gate 1

Presented Phase A findings plus two decisions to the user: (1) runbook's built-in Gate 1 question on PR #8's Azure code, (2) a new decision surfaced by the findings — amend PR #11 vs. open a redundant new stacked PR, since the "missing work" turned out to already be on #11.

**User answered both:** (1) strip PR #8's Azure code now. (2) amend PR #11 rather than open a new PR.

---

## 2026-08-31 — Phase C + amended Phase B executed

**PR #8 (strip Azure code):** checked out `stage3-job-conversion`, removed the Azure Login step and `az containerapp job update` step from `.github/workflows/thesis-AutoDeployTrigger-*.yml` (push to main now only builds+pushes to GHCR, nothing else), rewrote README's Stage 3 section to describe RunPod's manual "Start command" override instead of the Azure CLI command. Committed `ddf7479`, pushed to `origin/stage3-job-conversion` (auto-updates PR #8). Added a PR #8 comment documenting the change.

**PR #11 (amend instead of new PR):** checked out `runpod-self-termination`, added `ENV VLLM_USE_FLASHINFER_SAMPLER=0` to the Dockerfile (the second, previously-nowhere-to-be-found flashinfer JIT fix — bakes it into the image since it's vLLM's own recognized env var, not a custom RAG_-prefixed default). Committed `2d33fa5`, pushed to `origin/runpod-self-termination`. Rewrote PR #11's full description via `gh pr edit` to document the 7 previously-undocumented commits (vllm_engine.py, attention backend, max model len, ffmpeg, filename collision, partial-success test, leaked-answer cleanup) plus this new one, numbered as Fixes 7–12 alongside the original six, updated "What's still unverified" (vLLM path is now verified — nothing outstanding there), updated the Commits list.

**Checklist updated** (`Drafts/PRE_PHASE2_CHECKLIST.md`): items 1–6 checked off with notes on what actually happened vs. the original plan.

**Correction:** caught and fixed a transcription typo in this log and in memory — commit hash is `13a9b7d`, not `13c9b7d` (the earlier "Fix result filename collision" commit).

**Not yet done, deliberately:**
- Checklist item 7 (merge the stack) — Gate 2 not yet asked/answered. Per the runbook, silence or this session's earlier go-ahead does not count as consent to merge to main.
- Checklist item 8 (watch pod termination live) — explicitly out of scope, needs the user at the RunPod console.
- Checklist item 9 (reproducibility re-run from main) — depends on item 7.
- The untracked result-data risk (`phase1_results_complete/`, `phase1_results_partial/`, `baseline_raw_qwen3.jsonl`) flagged to the user alongside Gate 1 — not yet resolved, no backup action taken, awaiting direction since this wasn't part of the original checklist and shouldn't be guessed at.
- Checklist's second section (competitor list, novelty claim, Canvas date, meetings) — all explicitly `[Your call]`/`[You]`, out of scope for this session per the runbook.

---

## 2026-08-31 — Data backup done; Gate 2 answered; Phase D executed locally (NOT pushed)

**Data backup:** user chose to back up now, decide git-tracking later. Zipped `phase1_results_complete/` + `baseline_raw_qwen3.jsonl` to `Drafts/TEMP_BACKUP_phase1_results_2026-08-31.zip` (861 KB), named per user's explicit instruction to read as temporary, not permanent storage. `.gitignore`/git-tracking deliberately untouched, as instructed.

**Gate 2:** user answered yes — merge now, locally, no pushes between merges, plus a specific requirement: before the final push, directly grep the merged main's actual workflow file (not trust PR history) to confirm PR #8's guard is genuinely there. Report the confirmation, then stop and wait — user will open the Actions tab and give explicit go-ahead before the push happens.

**Phase D executed, locally only:** checked out `main` (confirmed synced with `origin/main` first), merged all six branches in order with `--no-ff`, no push between any step: `phase1-metric-fixes`(#6) → `stage2-vllm-path`(#7) → `stage3-job-conversion`(#8) → `cache-integrity-fix`(#9) → `corpus-revision-pinning`(#10) → `runpod-self-termination`(#11). All merges clean, no conflicts. Local `main` now at `b7f1f06`, six commits ahead of `origin/main` — **not pushed yet**.

**Required confirmation, done directly against the file, not inferred:** read the full merged `.github/workflows/thesis-AutoDeployTrigger-*.yml` on local `main`. Confirmed: zero occurrences of `containerapp`, `azure/CLI`, or `azure/login` anywhere in the file. The only deploy-adjacent step left is "Build and push image to GHCR." Push to main, once pushed, will do exactly that and nothing else — the Azure auto-deploy risk flagged before Gate 2 is fully closed by the merge as it stands locally right now.

**Incidental finding, not blocking:** merge history includes commit `6a7f4a6` ("TEMPORARY: diagnostic print of RUNPOD_API_KEY length + prefix" — printed `len(api_key)` and `api_key[:6]`, never the full key, to a pod's RunPod-visible logs, one run, 2026-07-28). Checked: removed by the very next commit (`f8395d5`), confirmed absent from current code via grep (`scripts/run_and_terminate.py` only has a comment referencing why `RUNPOD_API_KEY` isn't used, no print). Dead history, already self-resolved before this session started — mentioned for completeness, no action needed.

**Status: holding here.** Local `main` has all six merges ready. Waiting for the user's explicit go-ahead (they said they'll check the GitHub Actions tab first) before `git push origin main`.
