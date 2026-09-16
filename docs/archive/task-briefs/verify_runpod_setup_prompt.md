Before I touch the RunPod "Edit Pod" screen, verify each of the following by actually reading the relevant files — do not guess or infer from how similar projects usually work. If something can't be determined by reading this repo, say so explicitly rather than assuming.

## 1. How does code get into the running container — baked into the image, or pulled at runtime?

- [ ] Read the full `Dockerfile`. Does it `COPY` the repo's Python code into the image at build time, or does it only install dependencies and rely on something else (a mounted network volume, a git clone/pull step in the entrypoint or start command) to bring in the actual code at container start?
- [ ] Check the network volume mount path shown in the pod config (`/workspace`). Is `/workspace` where the actual git repo lives on the persistent network volume, separate from whatever's baked into the image? If so, the image itself could be old while the *code* the container actually runs is current, because it's reading from the volume, not from what was baked in at build time.
- [ ] Check `scripts/run_and_terminate.py` and any entrypoint script referenced by the Dockerfile (`ENTRYPOINT`, `CMD`) for any `git pull`, `git checkout`, or file-copy-from-volume logic that would sync code at startup.
- [ ] State plainly: as of right now, does the image tagged `runpod-eedce80` produce a running container with tonight's injection attack code available (`attacks/injection_templates.py`, `evaluation/run_attack_injection.py`, etc.) or not? Yes/no, with the evidence.

## 2. What actually builds and pushes a new image, and on what trigger?

- [ ] Read `.github/workflows/thesis-AutoDeployTrigger-*.yml` in full (the same file already audited earlier tonight for the Azure deploy issue). What does it build, on what trigger (push to which branch(es)), and does it tag the resulting image with the commit SHA the way `runpod-eedce80` suggests?
- [ ] Does this workflow trigger on push to feature branches like `phase2-injection-templates`, or only on `main`? If only `main`, and we are deliberately not pushing to `main` yet — is there currently *any* automated path to get a fresh image without pushing to `main`?
- [ ] If there's no automated path, what is the correct manual process to build and push an image tagged from `phase2-injection-templates`'s current commit? Give the actual `docker build` / `docker push` commands, correctly targeting `ghcr.io/simarjit1303/rag-adversarial-robustness/thesis` with an appropriate tag (e.g. the current commit SHA on this branch), assuming Docker is available wherever I'd run this (confirm whether that needs to happen locally, in a GitHub Actions manual dispatch, or elsewhere).
- [ ] Check whether `workflow_dispatch` is already configured on that workflow file (earlier audit notes mentioned `on: push: branches: [main], paths: ['**']` plus `workflow_dispatch`) — if so, can it be manually triggered against `phase2-injection-templates` via `gh workflow run` or the Actions tab, producing a correctly-tagged image without touching `main` at all? This may be the actual answer here — check before assuming a manual local Docker build is needed.

## 3. Does `run_and_terminate.py` support the injection attack, and does the self-terminating pattern fit the compute-length-check-then-smoke-test-then-scale workflow?

- [ ] Read `scripts/run_and_terminate.py` in full. Does it currently support running `evaluation.run_attack_injection`, or is it hardcoded/defaulted to the baseline sweep only?
- [ ] If it needs extending to support the injection sweep, what would that change look like (new env var to select which script runs, e.g. `RAG_RUN_TARGET=attack_injection`)? Is this a small change or does it touch the self-termination/timeout logic in a way that needs care?
- [ ] Separately: `compute_max_model_len.py --mode attack` and the smoke test both need their output read and a decision made before proceeding — they are not "fire and forget, let it terminate" tasks. Does `run_and_terminate.py`'s pattern (designed to run to completion and then kill the pod) fit this at all, or do these two specific steps need to run through an interactive terminal/SSH session on the pod instead, with the pod's termination handled manually afterward? State which approach is actually correct for each of the three steps (compute-length, smoke test, full sweep) — they may not all want the same execution pattern.

## Report back

For each of the three sections above, give a direct answer, not a summary of possibilities. I need to know exactly what to put in the RunPod "Edit Pod" screen — the correct image tag, the correct start command (or confirmation that I should be using an interactive session instead of the container image + terminate pattern for steps 1 and 2), and the correct environment variables — before I touch that screen again.
