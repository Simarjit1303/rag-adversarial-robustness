# Phase 3: instruction-detection defense (retrieval stage)

User already specified the full design in-session (detector choice
criteria, drop-not-refuse behavior, per-passage logging, integration
convention, smoke-test scope) — this note records it rather than
re-deriving it, per brainstorming's purpose being already satisfied.

## Decisions
- Detector: `protectai/deberta-v3-base-prompt-injection-v2` (HF Hub) —
  DeBERTa-v3-base binary classifier purpose-built for prompt injection,
  labels `SAFE`/`INJECTION` (confirmed via config.json), CPU-viable
  (~184M params), well-established (Apache-2.0, high download count).
- Behavior: flag → drop that passage from context, never refuse the
  whole query (measure utility impact of partial context loss).
- Integration point: mirrors `harness.pipeline.build_rag_user_prompt`'s
  `retrieved -> context` render loop exactly (same `extract_passage_text`
  call, same `[{i+1}] text` numbering), so callers swap their
  `context = "\n\n".join(...)` line for
  `context, log = filter_retrieved_passages(retrieved, corpus_name)`.
  One-line change in both `harness/pipeline.py` and
  `attacks/indirect_injection.py`'s render loops (not applied yet —
  this task ships the module + smoke test only, per the ask).
- `classifier` param on both public functions is the CPU-only-test /
  swap-detector-later injection point — real pipeline lazy-loaded only
  when not overridden, so the smoke test never needs network/GPU beyond
  the one real download it does to validate against the real model.

## Task
1. `defenses/instruction_detection.py`: `detect_injection` (per-passage)
   + `filter_retrieved_passages` (per-context-list, matches sweep calling
   convention).
2. `tests/test_instruction_detection.py`: smoke test, real CPU model,
   clean / ignore / fake_completion template examples reused from
   `attacks/injection_templates.py`.
3. Full pytest run, confirm no regressions against 247 passed / 1 skipped
   baseline.
4. One isolated commit.
