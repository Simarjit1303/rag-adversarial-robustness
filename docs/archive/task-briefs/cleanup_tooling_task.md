Build cleanup tooling for three places that have accumulated debris tonight, each with different risk levels — treat them very differently, since two of these touch real, irreplaceable data if done carelessly.

## 1. GHCR old images — safe to automate, low risk

Every `workflow_dispatch`/push to this repo's deploy workflow builds and pushes a new commit-SHA-tagged image (~10GB each, confirmed from tonight's builds). These accumulate in the container registry with no cleanup. Add a step to `thesis-AutoDeployTrigger-*.yml` (or a separate scheduled workflow) that keeps only the N most recent images (suggest N=5) and deletes older ones via the GitHub API (`gh api` or the registry's own delete endpoint). This is safe to automate fully — old images are always reproducible by rebuilding from the corresponding commit, nothing is lost.

## 2. RunPod network volume — dry-run first, never auto-delete real results

The `rag-scratch` volume has accumulated real debris tonight: 5 stray `.tmp_*` files from a July 27 crashed run (already manually identified and removed once from a local copy, but confirm whether they still exist on the actual volume itself — they may not have been cleaned there). Build a script (`scripts/clean_scratch_volume.py` or similar) that:
- Lists any `.tmp_*` files in `results/` older than some threshold (e.g., 24 hours) as candidates — these are always safe by design, since the atomic-write pattern (`_atomic_open`) means a `.tmp_*` file is by definition an incomplete/abandoned write, never a live result.
- Reports candidates and their sizes/ages, does NOT delete anything, on a normal run.
- Only deletes when explicitly passed `--confirm-delete`, and even then, prints exactly what it deleted afterward for the log.
- Never touches anything matching `attack_raw_*`, `attack_summary_*`, `baseline_raw_*`, `baseline_summary_*`, `phase1_results_complete/`, `phase1_results_partial/`, `PHASE2_PREP_LOG.md`, or any `.md`/`.log` file — these are real, sometimes-irreplaceable data or documentation. Exclude by pattern explicitly, don't just exclude what the script currently knows about, since new result-file naming conventions will appear as more attacks get built.

## 3. GitHub Actions runner disk — already exists, just confirm scope

The "Free disk space" step already added tonight (fixing the earlier build failure) covers this for CI. No new work needed here — just confirm in passing that it isn't overly aggressive in a way that could someday interfere with a build step that legitimately needs one of the toolchains it removes (unlikely given this repo's actual dependencies, but a one-line sanity check is cheap).

## What NOT to build

Do not build anything that runs automatically/unattended against the RunPod volume — no cron job, no "run this on every pod start" hook. Every volume cleanup action stays a manually-triggered, dry-run-first, explicitly-confirmed action for the rest of this project, given how much real data (and how much of tonight's actual debugging) has depended on that volume holding exactly what was put there and nothing silently removed.

Report back: what you built for #1, the dry-run output of #2 run against the actual current volume state (not a hypothetical), and confirm #3 needs no changes.
