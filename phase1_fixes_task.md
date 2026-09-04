# Phase 1 fixes — task brief for Claude Code

Three changes, in order. Do NOT push to `main` when done — see the safety note at the
bottom before committing anything. Work on a branch, e.g. `phase1-metric-fixes`.

---

## Task 1 — Check and fix the nq_open gold-answer bug

**File:** `data/loader.py` (or wherever `nq_open` examples get turned into `(question, gold)` pairs)

**What to look for:** the HuggingFace `nq_open` dataset stores `answer` as a **list** of
acceptable strings, not a single string. Search the loader for anything that does:

```python
gold = example["answer"][0]      # BUG: takes only the first gold, ignores the rest
```

or equivalent (`.iloc[0]`, unpacking the first element, etc.) anywhere the nq_open
example is converted into the eval-ready form.

**If found**, this is suppressing EM for every model, not just Qwen/Ministral, because
NQ-Open often has 2-4 acceptable phrasings per question ("Lincoln" / "Abraham Lincoln" /
"President Abraham Lincoln").

**Fix:** keep the full list of golds through to evaluation, and change the EM/F1
computation for nq_open specifically to **max over all golds**:

```python
def exact_match_multi(pred, golds):
    return max(exact_match(pred, g) for g in golds)

def f1_multi(pred, golds):
    return max(f1(pred, g) for g in golds)
```

This must NOT be applied to hotpot_qa or ms_marco unless those datasets also have
multi-answer fields — check before generalizing. If nq_open's loader already keeps the
full list and evaluates against it correctly, this task is a no-op — just confirm it
and report back, don't change working code.

---

## Task 2 — Add the deterministic cleanup step

**File:** `harness/pipeline.py`

**Where:** immediately after the existing `generated = generated.strip()` line (around
line 57 per the current handoff notes), before the string reaches `evaluation/metrics.py`.

**Add this exact function** (tested against 40 real Qwen/Ministral generations — see
`verify_cleanup.py` below for the test that validated it):

```python
import re

PREAMBLE_PATTERNS = [
    r'^the answer is[:\s]*',
    r'^the correct answer is[:\s]*',
    r'^according to (the )?(context|passage|document)[,:]?\s*',
    r'^the context (provided |states|indicates|says)[^,\.]*(states|indicates|says|that)[,:]?\s*',
]

def clean_generation(text):
    # 1. strip markdown emphasis
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # 2. strip empty/leftover think tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    # 3. strip leading boilerplate preambles (single pass, first match only)
    t = text.strip()
    for pat in PREAMBLE_PATTERNS:
        new_t = re.sub(pat, '', t, flags=re.IGNORECASE)
        if new_t != t:
            t = new_t.strip()
            break
    # 4. strip a single trailing period
    t = t.rstrip('.').strip()
    return t
```

Apply it identically to all four models — no per-model branching, no per-model
patterns. Store BOTH `generated` (raw) and `clean_generation(generated)` (cleaned) in
the output JSONL/results row, don't overwrite the raw string. We want the raw string
preserved for the debug/audit trail.

**Known limitation, document it in a code comment where this function lives:** this
only catches template-style preambles ("The answer is X."). It does NOT extract answers
embedded mid-sentence ("Kirk Cousins played for the Washington Redskins in 2017.") —
that would require actual span extraction, which is out of scope. Confirmed on real
data: this roughly doubles Qwen's EM, has zero effect on Ministral's EM (Ministral's
failure mode is mid-sentence embedding, not preambles).

---

## Task 3 — Wire the metric hierarchy through reporting

**File:** `evaluation/metrics.py` and wherever the summary CSV / console output is built
(likely `evaluation/run_baseline.py`)

Change the reported hierarchy to:

- **F1 = primary utility metric.** This is the headline number in every summary table,
  chart, and the trade-off matrix.
- **EM = secondary metric**, computed on the *cleaned* string (from Task 2), reported
  alongside F1, not instead of it.
- **`contains_answer` = diagnostic only.** Keep it in the raw JSONL / debug output for
  audit purposes, but do NOT include it in the robustness-utility trade-off matrix or
  any headline table. Label its column clearly as "diagnostic" wherever it appears so
  it's obvious in the CSV/plots that it's not a scored metric.

No computation changes needed here beyond what Task 1 and Task 2 already produce —
this task is about labeling and which columns feed the headline tables vs. the debug
output.

---

## Safety note — do not trigger a paid run

The GitHub Actions workflow currently builds to GHCR and deploys to the Azure
Container App **on every push to main** (per the handoff doc, this was never fixed —
build/run separation is still an open item). Merging this branch to `main` may
therefore trigger a real GPU deployment.

**Do not merge to main.** Leave this on its branch, open a PR, and stop. The student
will merge manually once vLLM batching is validated on free tier — the remaining Azure
credit ($36.78 as of today) is not enough to safely re-run the current unbatched
pipeline, so no GPU run should happen until batching is in place.

---

## Verification

Run `verify_cleanup.py` (same directory as this brief) after Task 2 is patched in. It
replays the exact 40 real generations from the last debug run and confirms the cleanup
function's known behavior hasn't regressed: Qwen EM should move from 2/20 to 4/20,
Ministral EM should stay at 7/20, both should hit 19/20 on `contains_answer` (the one
miss on both is "drowned" vs "drowning" — expected, a morphology gap, not a bug).
