"""
Phase 3 lands here: instruction detection (retrieval stage), Spotlighting
encoding mode (prompt stage, Hines et al. CAMLIS 2024 -- encoding, not
datamarking: the paper shows encoding (base64/ROT13) is the strongest of
its three variants (delimiting, datamarking, encoding), pushing ASR
closest to zero; this is the paper's own best-performing configuration,
not a different technique), and Llama-Guard output filtering (output
stage).

Judges: Llama-Guard-4-12B (primary -- updated from Llama-Guard-3-8B; a
dense architecture, not MoE, despite being pruned from Llama-4-Scout, and
fully text-capable; chosen now because the judge harness wasn't built yet,
so no baseline is invalidated by switching), Qwen3Guard-Gen-8B (second,
for inter-judge agreement -- independently benchmarked as top-performing
among open guard models in an ICLR 2026 workshop paper, "Benchmarking
Open-Source Safety Guard Models" (83.97% recall)), DeepSeek V4 Pro
(cross-validation + multi-turn orchestrator only — never a target model,
since its MoE architecture would confound the dense-only comparison in
SQ1).
"""
