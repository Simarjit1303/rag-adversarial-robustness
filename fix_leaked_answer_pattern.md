# Fix: clean_generation() doesn't handle the "[N] {dict}" leak artifact

## What was found

Comparing real `baseline_raw_hf.jsonl` vs `baseline_raw_vllm.jsonl`
(phi-4-mini x nq_open, n=1000, identical retrieval confirmed) surfaced a
real, describable generation artifact present on both engines: the model
occasionally outputs a Python dict/list-repr structure instead of a plain
answer, echoing something that looks like a retrieved-document or
few-shot example format rather than answering directly.

Rate: ~2.1% of vLLM outputs, ~1.0% of HF outputs (base rate across all
1000, not just mismatches). Not an engine-specific bug — both hit it.

**The critical detail: the correct answer is almost always sitting inside
this structure already.** The model isn't wrong on content — the current
`clean_generation()` just doesn't know how to extract an answer from this
shape, so F1/EM collapse even though the right answer is present in the
output.

## Real examples to fix against (exact strings from the actual data)

```python
CASES = [
    # (raw_generation, expected_extracted_answer)
    (
        "[1] {'question': 'what season is pepper in american horror story', 'answer': ['the second and fourth seasons']}",
        "the second and fourth seasons",
    ),
    (
        "[1] {'question': 'when did jackie robinson became rookie of the year', 'answer': ['in 1947']}",
        "in 1947",
    ),
    (
        "[1] {'question': 'who has the best nba record in history', 'answer': ['.890']}",
        ".890",
    ),
    (
        "[1] {'question': 'when were the extra books of the catholic bible added', 'answer': ['393']}",
        "393",
    ),
    (
        "[1] {'question': 'thicknet is the colloquial name for which ethernet standard', 'answer': ['10BASE5']}",
        "10BASE5",
    ),
    (
        "[1] {'question': 'when does the lion witch and the wardrobe take place', 'answer': ['1940']}",
        "1940",
    ),
    ("['.890']", ".890"),  # simpler bare-list variant, no dict wrapper
    ("[1]", "[1]"),  # degenerate case, no 'answer' key present -- must NOT crash,
                       # falls through unchanged since there's nothing to extract
]
```

## Suggested approach

Before or alongside the existing preamble-stripping logic in
`clean_generation()`, detect this shape and extract from it:

```python
import ast
import re

def _try_extract_leaked_answer(text: str) -> str | None:
    """
    Detects a leaked dict/list-repr structure containing an 'answer' key
    and extracts its first value. Returns None if the text doesn't match
    this shape, so callers can fall through to normal cleaning untouched.
    """
    match = re.search(r"\{.*'answer':\s*\[.*?\].*\}", text)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(0))
        answer = parsed.get("answer")
        if isinstance(answer, list) and answer:
            return str(answer[0])
    except (ValueError, SyntaxError):
        pass
    return None
```

Use `ast.literal_eval`, not `eval` — the leaked text uses Python literal
syntax (single-quoted strings), and `literal_eval` only parses safe
literals, never executes arbitrary code. This matters given the input is
model-generated text, not trusted input.

**Call this before the existing cleanup logic**, and if it returns a
non-None result, use that as the starting point for the rest of
`clean_generation()`'s normal processing (trailing punctuation stripping,
etc.) rather than bypassing cleanup entirely.

**The bare `"['.890']"` case needs its own path** — no dict, no 'answer'
key, just a raw list-repr of the answer itself. Handle this as a fallback
if the dict-extraction regex doesn't match but the text still looks like
a bare Python list literal.

## Testing

Add all 8 cases above as explicit test cases, asserting the exact
expected output for each — including the `"[1]"` degenerate case, which
must NOT raise, NOT extract anything, and fall through to whatever
`clean_generation()` currently does with unparseable input.

**Also add negative tests** — confirm this new logic does NOT fire on
normal answers that happen to contain brackets or braces for legitimate
reasons (e.g., an answer that's naturally `"the [REDACTED] files"` or
similar) — the regex should be specific enough to the `'answer': [...]`
shape that it doesn't over-trigger on unrelated bracket usage.

**Re-run `verify_cleanup.py`** (the existing regression script from PR #6)
to confirm this addition doesn't change behavior on any of the previously
verified 40 samples — this should be purely additive, not a regression.

## What this doesn't need to include

This is a `clean_generation()` change only. No Docker rebuild is strictly
required to test the logic (pure Python, no GPU), though a rebuild is
still needed before this fix actually runs on the next real RunPod sweep.
