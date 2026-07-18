# Adversarial Robustness of Open-Source RAG Pipelines — Phase 1 Scaffold

Cross-model benchmark of prompt injection, knowledge poisoning, and multi-turn
attacks under EU AI Act Article 15. This is the **Phase 1 baseline harness**:
retrieval + generation + clean-query evaluation, no attacks or defenses yet.
Phase 2 (attacks) and Phase 3 (defenses) land in `attacks/` and `defenses/`
once this baseline is validated.

Every design decision here (which four models, why dense-only, why Ministral
needs a text-only verification step) traces back to the dissertation plan.
Section references in comments point back to that document.

## 1. Setup

Requires **Python 3.13** (the Dockerfile pins `python:3.13-slim`; 3.14 is
blocked by vLLM's dependency tree — see vllm-project/vllm#34096 and the
Dockerfile comment).

```bash
git init
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
cp .env.example .env   # then fill in your HuggingFace token
huggingface-cli login  # or export HUGGINGFACE_HUB_TOKEN from .env
```

**Before you run anything:** Llama-3.1-8B-Instruct is gated on HuggingFace.
Go to the model page, accept Meta's license, and wait for approval (usually
fast, occasionally up to a day or two). Do this first, it's the most likely
thing to block you mid-week if you leave it for later. Qwen3-8B,
Phi-4-mini-instruct, and Ministral-3-8B-Instruct-2512 are all ungated.

**Ministral-3-8B needs a recent `transformers` version** to support
`Mistral3ForConditionalGeneration`. If the import in `harness/model_loader.py`
fails, run `pip install -U transformers` and check the model card on
HuggingFace for the minimum version it needs.

## 2. Pin your model revisions

Open `config.py`. Every model's `revision` field is `None` right now, which
means "load whatever `main` currently points to" — fine for getting the
harness working, not fine for anything you intend to report. Before your
real baseline run, go to each model's "Files and versions" tab on
HuggingFace, copy the commit hash, and fill it in. That hash is what goes in
your reproducibility appendix.

## 3. Build the retrieval side (model-independent, do this first)

```bash
python -m data.loader        # caches a 1,000-doc dev sample of each corpus
python -m data.build_index   # embeds + builds a FAISS index per corpus
```

Sanity-check retrieval before touching any LLM:

```python
from data.build_index import build_index, retrieve
index, records = build_index("nq_open", split="dev")
retrieve(index, records, "What is the capital of France?", k=3)
```

If the results look sensible, retrieval is solid and any generation
problems you see later aren't coming from this layer.

## 4. Verify the Ministral text-only claim

```bash
python -m scripts.verify_ministral_text_only
```

This checks that Ministral-3-8B's vision encoder never fires when the
pipeline sends text-only input. If it fails, the "vision encoder present
but unused" caveat in the dissertation's Approach chapter needs revisiting
before you trust any Ministral baseline numbers.

## 5. Run the Phase 1 baseline sweep

```bash
python -m evaluation.run_baseline
```

This loops all four models across all three corpora on clean queries, and
writes:

- `results/baseline_raw.jsonl` — one line per (model, corpus, question):
  the generated answer, EM, F1, and retrieved doc IDs. Keep this. You'll
  need the raw per-question outputs later for the McNemar significance
  tests, and re-running everything in September to regenerate them would
  be expensive.
- `results/baseline_summary.csv` — aggregated EM/F1 per model × corpus,
  the numbers that go straight into your first results table.

Loading four ~4-9B models sequentially on modest hardware will take a
while. If you're VRAM-constrained, run one model at a time by passing
`model_keys=["llama-3.1-8b"]` etc. into `run_baseline_sweep()`.

## 6. Cloud runs: push no longer executes anything (Stage 3)

This changed from how the project worked before, so read it once:

- **Pushing to `main`** builds the image, pushes it to GHCR, and updates the
  image reference on the Container Apps **Job** `rag-sweep-job`. That is all.
  A Job — unlike the old Container App — does not execute when its image
  changes, so a push can no longer start (or restart-loop) a paid GPU sweep.
- **Running the sweep** is a separate, explicit, human action:

  ```bash
  az containerapp job start \
    --name rag-sweep-job \
    --resource-group Master-Thesis
  ```

  Nothing in CI calls `job start`, deliberately. If you didn't run that
  command, nothing is billing.
- Until Part B of Stage 3 provisions `rag-sweep-job`, the workflow's
  image-update step fails — expected, not a bug.

## Repo structure

```
config.py                  Central registry: models, corpora, paths, seed
data/
  loader.py                 Corpus loading + fixed-seed sampling
  normalize.py               Per-corpus field mapping (question/answer/context)
  build_index.py            FAISS index build + retrieval
harness/
  model_loader.py            Loads each model with the correct class/settings
  pipeline.py                 Retrieve -> prompt -> generate (Phase 1 core path)
evaluation/
  metrics.py                 EM, F1, Recall@5
  run_baseline.py           Phase 1 sweep entry point
attacks/                    Phase 2 lands here (poisoning, injection, multi-turn)
defenses/                   Phase 3 lands here (instruction detection, spotlighting, guard filtering)
scripts/
  verify_ministral_text_only.py
results/                    Raw + summary outputs (gitignored except structure)
```

## Known things to double-check before trusting output

1. **Dataset field names.** `data/normalize.py` maps each corpus's question/
   answer/context fields to a common schema based on documented schemas —
   confirm against the live dataset the first time you load it
   (`print(records[0].keys())`), since HF dataset schemas do shift between
   versions.
2. **Qwen3 thinking mode.** `config.QWEN3_ENABLE_THINKING` is set once for
   the whole benchmark. Don't let it vary silently between runs or
   conditions — state whichever you pick in the methodology chapter.
3. **Greedy decoding.** Generation uses `do_sample=False` for reproducibility.
   If you deliberately want sampling later (e.g. for some attack variants),
   change it explicitly and log the seed used.
