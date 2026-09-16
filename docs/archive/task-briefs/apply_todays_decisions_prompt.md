Before continuing the Phase 2 injection build, apply the following decisions across every relevant file. Report back what you changed and where — don't just say "done."

## 0. File locations — check both before editing anything

The files this task touches may live in either of two sibling folders under the same parent (`Simarjit Singh\`):
- The repo: `Simarjit Singh\Codes\rag-adversarial-robustness` (config.py, data/normalize.py, phase2_indirect_injection_task.md, nq_open_leakage_finding.md, and possibly PHASE2_ROADMAP.md if it was ever copied in)
- Drafts: `Simarjit Singh\Drafts` (PHASE2_ROADMAP.md likely lives here, possibly only here)

For each file referenced below, check both locations before concluding a file doesn't exist. If a file exists in both places, treat that as worth flagging back to me rather than silently picking one — it may mean the copies have already diverged, or it may just be an intentional duplicate (reference copy in Drafts, working copy in the repo). Don't edit both silently without saying so. If `PHASE2_ROADMAP.md` only exists in Drafts and not in the repo, edit it in place in Drafts — don't move or copy it into the repo as part of this task.

## 1. nq_open decision (from the leakage finding)

This should already be partly reflected from `fix-rag-context-answer-leak`, but make sure it's consistent everywhere:
- [ ] `PHASE2_ROADMAP.md` (repo root or `docs/`, wherever it lives): add a clearly marked section stating nq_open is excluded from all real Phase 2/3 sweeps going forward, with the leakage finding as the reason. State explicitly that the attack/defense matrix is effectively 4 models × 2 corpora (hotpot_qa, ms_marco) × 3 attacks for real results, with nq_open's Phase 1 result retained only as a documented limitation/finding, not a valid baseline.
- [ ] `phase2_indirect_injection_task.md`: update the "Done means" section and any corpus-loop references to explicitly exclude nq_open from the sweep scope.
- [ ] Anywhere in `config.py` or `data/normalize.py` that lists nq_open alongside the other two corpora as if all three are equally valid for real sweeps — add a comment flagging the exclusion, without deleting or breaking the corpus definition itself (the flagged Phase 1 data and the regression test proving the leak stay as-is, per the earlier decision).
- [ ] `nq_open_leakage_finding.md`: no changes needed, just confirm it's committed and not lost.

## 2. Judge model: Llama-Guard-3-8B → Llama-Guard-4-12B

- [ ] Everywhere `Llama-Guard-3-8B` or `llama-guard-3-8b` appears as a judge reference (config, roadmap, task file, any judge-harness scaffolding if it exists yet) — update to `Llama-Guard-4-12B` / `meta-llama/Llama-Guard-4-12B`.
- [ ] Note in the relevant doc: this is a dense architecture (not MoE, despite being pruned from Llama-4-Scout), fully text-capable, chosen because the judge harness wasn't built yet so there's no baseline to invalidate by switching.
- [ ] Confirm `Qwen3Guard-Gen-8B` (not `Qwen3Guard-8B`) is used consistently — this was already corrected earlier, just verify it stuck everywhere, including any new files created since.
- [ ] Add a citation note wherever the judge ensemble is described: Qwen3Guard-Gen-8B was independently benchmarked as top-performing among open guard models in an ICLR 2026 workshop paper ("Benchmarking Open-Source Safety Guard Models," 83.97% recall) — worth having this citation ready for the methodology chapter.

## 3. Spotlighting defense: datamarking → encoding mode

- [ ] Wherever Spotlighting is described as the defense (roadmap's Phase 3 section, any defense-scaffolding code if it exists yet — likely not built yet, so this is probably just a documentation change): switch the stated mode from datamarking to encoding (base64/ROT13).
- [ ] Add the reasoning: Hines et al.'s original paper shows encoding is the strongest of the three spotlighting variants (delimiting, datamarking, encoding), pushing ASR closest to zero — using the paper's own best-performing configuration, not a different technique.

## 4. HARC — flag, do not resolve

- [ ] Do NOT invent or assume what HARC stands for. Add a visible `TODO` or note wherever HARC is referenced (roadmap, config) stating: "Source of this acronym not yet verified against current literature — needs tracing back to its original document (course material or specific paper) before citing in the methodology chapter."

## 5. After all edits

- [ ] Run the existing test suite to confirm nothing broke.
- [ ] Commit these as a clearly separated commit (or a couple of small ones) on the current branch — don't mix this into the injection-attack code itself, since these are cross-cutting doc/config updates, not attack logic.
- [ ] Report: exact files touched, and confirm none of this affected the fix-rag-context-answer-leak branch (should be untouched, still unmerged, per earlier instruction).

Once this is done, resume exactly where you left off: the injection template/ASR/stats build, scoped to hotpot_qa and ms_marco only, per the nq_open decision above.
