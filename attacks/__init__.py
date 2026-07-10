"""
Phase 2 lands here: corpus/knowledge poisoning (PoisonedRAG method),
indirect prompt injection (Greshake / Formalizing-PI harness), and the
Crescendo-adapted multi-turn jailbreak orchestrated by DeepSeek V4 Pro
(with a reproducible open-model fallback orchestrator).

Each attack should wrap harness.pipeline.run_query rather than duplicating
the retrieve-then-generate logic, so Phase 2 results stay measured against
the exact same clean baseline established in Phase 1.
"""
