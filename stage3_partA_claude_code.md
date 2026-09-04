# Stage 3, Part A — repo/workflow changes (Claude Code)

This is the code-only half of Stage 3. It does not touch your live Azure
subscription — no new resources, no cost, safe to run regardless of credit
balance. Part B (actual provisioning) is a separate document, run directly
by the student, not handed to Claude Code.

Work on a new branch, `stage3-job-conversion`, stacked on `stage2-vllm-path`.
Same standing rule as PR #6 and #7: open the PR, don't merge to main.

---

## Context

The deploy workflow currently builds an image and pushes it to GHCR on every
push to main, and the existing Container App's revision-update behavior
means that also starts execution — this is the mechanism behind the
overnight restart-loop billing scare earlier in the project. The fix isn't
a manual-trigger flag bolted onto the same resource type; it's switching to
a **Container Apps Job**, a different resource type that by design does not
auto-execute when its image updates. That structural difference is the
actual fix. Job creation itself happens in Part B — this part only prepares
the repo for a Job to exist.

---

## Task A1 — GitHub Actions workflow

Change the post-build step from whatever currently touches the Container
App to updating the Job's image reference only:

```yaml
- name: Update Container Apps Job image
  run: |
    az containerapp job update \
      --name rag-sweep-job \
      --resource-group <resource-group> \
      --image ghcr.io/simarjit1303/rag-adversarial-robustness:${{ github.sha }}
```

Do not add a step that calls `az containerapp job start` anywhere in this
workflow. Starting an execution is a manual, human decision from here
forward — that's the entire point of this change. If the Job named
`rag-sweep-job` doesn't exist yet (Part B hasn't run), this step will fail
— that's expected and fine until Part B is done; don't work around it by
adding fallback logic that points back at the old Container App.

## Task A2 — confirm `RAG_SCRATCH_DIR` works unmodified against a mounted path

`config.py`'s existing `RAG_SCRATCH_DIR` env var override should work as-is
once an Azure Files share gets mounted at some path (e.g.
`/mnt/rag-scratch`) — from Python's perspective that's just a filesystem
path, no code change should be needed. Confirm this rather than assume it:
check `data/loader.py`, `data/build_index.py`, and `evaluation/run_baseline.py`
for anything that assumes local-disk behavior an Azure Files SMB/NFS mount
might not provide (certain file-locking semantics, or performance
assumptions baked into batch sizes for FAISS index writes in particular).
If nothing turns up, say so explicitly in the PR rather than silently skip
this check.

## Task A3 — README update

Document the new operational model: pushing to main updates the Job's image
but does not run anything; running the sweep is now a separate, explicit
action (`az containerapp job start`, with the actual command shown). This is
a real change from how the project worked before — worth spelling out
clearly rather than leaving it to be rediscovered later.

---

## Verification

1. Confirm the workflow YAML has no path that calls `job start`, under any
   condition.
2. Confirm A2's check was actually done and documented, not skipped.
3. Open the PR. It will reference a Job (`rag-sweep-job`) that doesn't exist
   yet until Part B runs — that's expected, not a bug to fix here.
