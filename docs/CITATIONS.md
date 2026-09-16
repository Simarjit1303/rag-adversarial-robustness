# Citations and References — Phases 0–3

**Inclusion policy:** every entry below is a peer-reviewed, published source — conference proceedings, journal, or workshop proceedings with a formal publisher (ACM, IEEE, PMLR, USENIX, ACL Anthology, CEUR-WS). No bare arXiv preprints are cited as primary sources. Where a paper started as an arXiv preprint and was later accepted at a venue, the venue-published version is cited, not the arXiv listing. Eight entries are knowingly *not* peer-reviewed papers — every model release actually run in this project's pipeline (target models, attacker/judge/generator models, and defense-pipeline models) that has no peer-reviewed paper behind it — documented as model artifacts, not studies, with that distinction stated explicitly (see "Model artifacts" below). Ordered newest-first within each section.

---

## Phase 0/1 — Baseline (datasets, retrieval, serving)

**Datasets**

- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2369–2380. Association for Computational Linguistics. https://aclanthology.org/D18-1259/

- Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., & Petrov, S. (2019). *Natural Questions: A Benchmark for Question Answering Research*. Transactions of the Association for Computational Linguistics, 7, 453–466. https://doi.org/10.1162/tacl_a_00276
  — *Source of nq_open; cited in support of the corpus-construction finding that led to nq_open's exclusion from real sweeps.*

- Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., & Deng, L. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. In Proceedings of the Workshop on Cognitive Computation: Integrating Neural and Symbolic Approaches 2016, co-located with NIPS 2016. CEUR Workshop Proceedings, Vol. 1773. https://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf

**Retrieval and serving infrastructure**

- Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-Scale Similarity Search with GPUs*. IEEE Transactions on Big Data, 7(3), 535–547. https://doi.org/10.1109/TBDATA.2019.2921572
  — *FAISS, used for dense retrieval indexing.*

- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. In Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23), pp. 611–626. Association for Computing Machinery. https://doi.org/10.1145/3600006.3613165
  — *vLLM, the inference engine used throughout the sweep harness.*

**Embedding model (dense retrieval)**

- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992. Association for Computational Linguistics. https://aclanthology.org/D19-1410/
  — *Foundational architecture (Siamese/triplet fine-tuning of a transformer encoder for cosine-comparable sentence embeddings) underlying `sentence-transformers/all-mpnet-base-v2`, the embedder used to build the FAISS retrieval index for every corpus.*

- Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). *MPNet: Masked and Permuted Pre-training for Language Understanding*. Advances in Neural Information Processing Systems 33 (NeurIPS 2020). https://proceedings.neurips.cc/paper/2020/hash/c3a690be93aa602ee2dc0ccab5b7b67e-Abstract.html
  — *Pre-training objective underlying the base encoder of `all-mpnet-base-v2`, the specific checkpoint used.*

---

## Phase 2, Attack 1 — Indirect Prompt Injection

- Liu, Y., Jia, Y., Geng, R., Jia, J., & Gong, N. Z. (2024). *Formalizing and Benchmarking Prompt Injection Attacks and Defenses*. In Proceedings of the 33rd USENIX Security Symposium (USENIX Security 24), pp. 1831–1847. USENIX Association. https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei
  — *Directly relevant formalization of injection ASR methodology; useful for methodology-chapter framing alongside your own template taxonomy.*

- Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec '23), pp. 79–90. Association for Computing Machinery. https://doi.org/10.1145/3605764.3623985
  — *Foundational paper establishing indirect prompt injection as an attack class; the canonical citation for this attack family.*

**Related benchmarking work (context/discussion, not directly used)**

- Zhan, Q., Fang, R., Bindu, R., Gupta, A., Hashimoto, T., & Kang, D. (2024). *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents*. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 10471–10506. Association for Computational Linguistics. https://aclanthology.org/2024.findings-acl.624/

---

## Phase 2, Attack 2 — PoisonedRAG (Knowledge Corruption)

- Zou, W., Geng, R., Wang, B., & Jia, J. (2025). *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models*. In Proceedings of the 34th USENIX Security Symposium (USENIX Security 25), pp. 3827–3844. USENIX Association. https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag
  — *The attack implemented directly in this thesis.*

**2026 follow-on defense work (for discussion chapter — corroborates PoisonedRAG's continued relevance as the field's reference threat model)**

- Moradi, R., Alizadeh Noughabi, H., Zarrinkalam, F., & Dehghantanha, A. (2026). *Defending RAG Against Knowledge Poisoning Using Cross-Encoder Activation Signals*. In Proceedings of the 39th Canadian Conference on Artificial Intelligence. Proceedings of Machine Learning Research, Vol. 318, pp. 366–376. PMLR. https://proceedings.mlr.press/v318/moradi26a.html
  — *2026 defense benchmarked directly against PoisonedRAG-style corruption; confirms the attack is still the field's active reference point.*

---

## Phase 2, Attack 3 — Crescendo (Multi-Turn Jailbreak) and Behavior Pool

- Russinovich, M., Salem, A., & Eldan, R. (2025). *Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack*. In Proceedings of the 34th USENIX Security Symposium (USENIX Security 25). USENIX Association. https://www.usenix.org/conference/usenixsecurity25/presentation/russinovich
  — *The attack implemented directly in this thesis.*

- Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramèr, F., Hassani, H., & Wong, E. (2024). *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models*. Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/63092d79154adebd7305dfd498cbff70-Abstract.html
  — *Source of JBB-Behaviors, 20% of the stratified behavior pool.*

- Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., & Hendrycks, D. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal*. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024). Proceedings of Machine Learning Research, Vol. 235, pp. 35181–35224. PMLR. https://proceedings.mlr.press/v235/mazeika24a.html
  — *Source of HarmBench behaviors, 80% of the stratified behavior pool.*

**2026 comparison/discussion work**

- Nakka, K., & Saxena, N. (2026). *BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage*. In Findings of the Association for Computational Linguistics: EACL 2026. Association for Computational Linguistics.
  — *Confirmed accepted, EACL 2026 Findings; useful discussion-chapter citation showing jailbreak research remains an active, currently-publishing field in 2026, directly contemporaneous with this thesis.*

---

## Phase 3 — Defenses

**Instruction detection**

- He, P., Gao, J., & Chen, W. (2023). *DeBERTaV3: Improving DeBERTa Using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. In Proceedings of the Eleventh International Conference on Learning Representations (ICLR 2023). https://openreview.net/forum?id=sE7-XhLxHA
  — *Peer-reviewed architecture underlying the fine-tuned classifier used for the instruction-detection defense.*

- Liu, Y., Jia, Y., Jia, J., Song, D., & Gong, N. Z. (2025). *DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks*. In 2025 IEEE Symposium on Security and Privacy (S&P), pp. 2190–2208. IEEE. https://doi.org/10.1109/SP61157.2025.00119
  — *Not the classifier used in this thesis, but the closest peer-reviewed comparator for instruction-detection-style defenses; worth citing in the defense-design discussion even though a different (industry) classifier was used for practical reasons.*

**Spotlighting (encoding mode)**

- Hines, K., Lopez, G., Hall, M., Zarfati, F., Zunger, Y., & Kıcıman, E. (2024). *Defending Against Indirect Prompt Injection Attacks With Spotlighting*. In Proceedings of the Conference on Applied Machine Learning for Information Security (CAMLIS 2024). CEUR Workshop Proceedings, Vol. 3920, Paper 03. https://ceur-ws.org/Vol-3920/paper03.pdf
  — *The defense implemented directly in this thesis; encoding-mode variant specifically evaluated.*

**Output filtering (guard model)**

- Meta AI. (2025). *Llama Guard 4 Model Card*. Meta Platforms, Inc. https://huggingface.co/meta-llama/Llama-Guard-4-12B
  — *Model artifact, not a peer-reviewed paper; cited as a model release per standard practice for undocumented-in-literature safety classifiers (see note below).*

**Utility / over-refusal benchmarking**

- Cui, J., Chiang, W.-L., Stoica, I., & Hsieh, C.-J. (2025). *OR-Bench: An Over-Refusal Benchmark for Large Language Models*. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025). Proceedings of Machine Learning Research, Vol. 267, pp. 11515–11542. PMLR. https://proceedings.mlr.press/v267/cui25a.html

- Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2024). Association for Computational Linguistics. https://aclanthology.org/2024.naacl-long.301/
  — *Dataset access note: the original `paul-rottger/xstest` Hub handle has been retired; the dataset is current and live under `Paul/XSTest` (the author's current handle), same 450-prompt suite. Config updated accordingly.*

---

## Statistical methodology

- McNemar, Q. (1947). *Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages*. Psychometrika, 12(2), 153–157. https://doi.org/10.1007/BF02295996

- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65–70.

- Fisher, R. A. (1922). *On the Interpretation of χ² from Contingency Tables, and the Calculation of P*. Journal of the Royal Statistical Society, 85(1), 87–94. https://doi.org/10.2307/2340521

- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552

---

## Model artifacts (not peer-reviewed papers — cited as artifacts, per standard practice)

Every model release this project actually runs that has no peer-reviewed paper behind it is listed here explicitly, grouped by pipeline role, rather than presented as having equivalent academic standing to the papers above. Where a company-published technical report exists (arXiv or a company-hosted PDF), it is cited as a technical-report artifact, not as a substitute for peer review; where none exists at all, only the model card/announcement is cited.

**Target models under test (Phases 1–3)** — the four models whose robustness this project measures; these are the actual research subjects, not pipeline tooling.

- **Llama-3.1-8B-Instruct** (Meta AI, 2024). Grattafiori, A., et al. *The Llama 3 Herd of Models*. arXiv:2407.21783. https://arxiv.org/abs/2407.21783 — technical report, not peer-reviewed as of this writing. Model: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

- **Qwen3-8B** (Qwen Team, Alibaba, 2025). Yang, A., et al. *Qwen3 Technical Report*. arXiv:2505.09388. https://arxiv.org/abs/2505.09388 — technical report, not peer-reviewed as of this writing. Model: https://huggingface.co/Qwen/Qwen3-8B

- **Phi-4-mini-instruct** (Microsoft, 2025). Abouelenin, A., et al. *Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs*. arXiv:2503.01743. https://arxiv.org/abs/2503.01743 — technical report, not peer-reviewed as of this writing. Model: https://huggingface.co/microsoft/Phi-4-mini-instruct

- **Ministral-3-8B-Instruct-2512** (Mistral AI, 2025) — no technical report or paper exists for this release; documented via the announcement blog post ("Introducing Mistral 3," https://mistral.ai/news/mistral-3/, 2 December 2025) and the Hugging Face model card only. Model: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512

**Attacker / generator / judge models** — used only to *produce or score* attack data (PoisonedRAG's poison passages; Crescendo's escalation turns and verdicts), never evaluated as this project's research subjects.

- **`nvidia/nemotron-3-ultra-550b-a55b`** (NVIDIA, 2025) — PoisonedRAG's poison-passage generator (Section 5.2 of `THESIS_MASTER_RECORD.md`). Documented via NVIDIA's own technical report ("NVIDIA Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning," NVIDIA, 2025/2026, https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) — a company-published PDF report, not a peer-reviewed venue. Model card: https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard

- **`deepseek-ai/deepseek-v4-pro-0813`** (DeepSeek-AI, 2026) — Crescendo's attacker and judge model, never a target (Section 5.3 of `THESIS_MASTER_RECORD.md`). DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. arXiv:2606.19348. https://arxiv.org/abs/2606.19348 — technical report, not peer-reviewed as of this writing.

**Defense-pipeline models**

- **`protectai/deberta-v3-base-prompt-injection-v2`** (Protect AI, 2024) — industry-released classifier fine-tuned from the peer-reviewed DeBERTaV3 architecture (He et al., ICLR 2023, cited above). No accompanying paper; documented via Hugging Face model card only. https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2

- **Llama Guard 4** (Meta AI, 2025) — model release, no accompanying peer-reviewed paper as of this writing. Documented via model card only. https://huggingface.co/meta-llama/Llama-Guard-4-12B

Cite all of the above in the methodology chapter as "a released model" / "an industry technical report," not as a peer-reviewed study — this is the accurate and defensible framing for a viva.

---

## Reference implementations consulted (code provenance, not bibliography)

This project's own attack implementations are original code, not copies: a repo-wide search of `attacks/poisonedrag.py`, `attacks/crescendo.py`, and the rest of the source tree, plus the full git history, found no verbatim external code, no vendored files, and no third-party license headers. Two named reference implementations were consulted for methodology fidelity while building the attacks described above, and both are named inline in `THESIS_MASTER_RECORD.md` Sections 5.2–5.3; their repository URLs are recorded here for completeness:

- **PoisonedRAG** (Section 5.2) — informed by the paper's own official reference implementation, `sleeepeer/PoisonedRAG`: https://github.com/sleeepeer/PoisonedRAG

- **Crescendo** (Section 5.3) — informed by Microsoft's PyRIT `CrescendoOrchestrator`, the reference implementation of the paper's escalation/backtrack strategy. Originally consulted at `Azure/PyRIT`; that repository is now archived and redirects to its current canonical location, `microsoft/PyRIT`: https://github.com/microsoft/PyRIT

Neither is cited as a source of copied code — both are disclosed here as implementations read for methodological correctness when reproducing each paper's described algorithm, consistent with standard practice for reimplementing a published attack.

---

*Compiled September 2026; extended the same month with the full model-artifact and code-provenance audit above. All venue and page-number details verified against official proceedings pages (PMLR, ACL Anthology, USENIX, IEEE Xplore/CEUR-WS) at time of writing — re-verify any DOI links before final submission in case of link rot.*
