from pathlib import Path
import os
ROOT_DIR = Path(__file__).resolve().parent
LOCAL_SCRATCH = Path(os.environ.get('RAG_SCRATCH_DIR', 'C:\\rag-data-local'))
DATA_DIR = LOCAL_SCRATCH / 'cache'
INDEX_DIR = LOCAL_SCRATCH / 'indices'
RESULTS_DIR = LOCAL_SCRATCH / 'results'
for d in (DATA_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
SEED = 42
MODELS = {'llama-3.1-8b': {'hf_id': 'meta-llama/Llama-3.1-8B-Instruct', 'revision': '0e9e39f249a16976918f6564b8830bc894c89659', 'loader': 'causal_lm', 'gated': True, 'license': 'Llama 3.1 Community License', 'note': 'Retained: no dense 7-8B Llama 4 model exists (Llama 4 went straight to MoE).'}, 'qwen3-8b': {'hf_id': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218', 'loader': 'causal_lm', 'gated': False, 'license': 'Apache-2.0', 'note': 'Replaces Qwen2.5-7B-Instruct. Supports a thinking-mode toggle, see QWEN3_ENABLE_THINKING below.'}, 'phi-4-mini': {'hf_id': 'microsoft/Phi-4-mini-instruct', 'revision': 'cfbefacb99257ffa30c83adab238a50856ac3083', 'loader': 'causal_lm', 'gated': False, 'license': 'MIT', 'note': 'Replaces Phi-3.5-mini-instruct.'}, 'ministral-3-8b': {'hf_id': 'mistralai/Ministral-3-8B-Instruct-2512', 'revision': 'aae06a2125402f2a89efbacf0881623c15a711d0', 'loader': 'mistral3', 'gated': False, 'license': 'Apache-2.0', 'note': 'Replaces Mistral-7B-Instruct-v0.3. Ships as an 8.4B language backbone plus a 0.4B vision encoder. This harness only ever sends text, so the vision encoder is loaded but never invoked — confirmed by scripts/verify_ministral_text_only.py.'}}
QWEN3_ENABLE_THINKING = False
CORPORA = {'nq_open': {'hf_id': 'google-research-datasets/nq_open', 'revision': '5dd9790a83002ad084ddeb7c420dc716852c6f28', 'dev_n': 1000, 'eval_n': 10000}, 'hotpot_qa': {'hf_id': 'hotpotqa/hotpot_qa', 'hf_config': 'distractor', 'revision': '1908d6afbbead072334abe2965f91bd2709910ab', 'dev_n': 1000, 'eval_n': 10000}, 'ms_marco': {'hf_id': 'microsoft/ms_marco', 'hf_config': 'v2.1', 'revision': 'a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a', 'dev_n': 1000, 'eval_n': 10000}}
BEHAVIOR_DATASETS = {'jbb_behaviors': {'hf_id': 'JailbreakBench/JBB-Behaviors', 'hf_config': 'behaviors', 'split': 'harmful', 'revision': '886acc352a31533ffbcf4ef22c744658688086fc', 'behavior_column': 'Goal', 'expected_n': 100}, 'harmbench': {'hf_id': 'walledai/HarmBench', 'hf_config': ['standard', 'contextual', 'copyright'], 'split': 'train', 'revision': 'fb6c2afd5a2a943d701d6db3efab87d077e81be5', 'behavior_column': 'prompt', 'expected_n': 400}}
XSTEST_HF_ID = 'Paul/XSTest'
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
TOP_K = 5
RAG_VLLM_MAX_MODEL_LEN = 13056
