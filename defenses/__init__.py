"""
Phase 3 lands here: instruction detection (retrieval stage), Spotlighting
datamarking (prompt stage, Hines et al. CAMLIS 2024), and Llama-Guard
output filtering (output stage).

Judges: Llama-Guard-3-8B (primary), Qwen3Guard-8B (second, for inter-judge
agreement), DeepSeek V4 Pro (cross-validation + multi-turn orchestrator
only — never a target model, since its MoE architecture would confound
the dense-only comparison in SQ1).
"""
