import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
print('=== original verify_cleanup.py checks (run on import) ===')
import verify_cleanup as vc

def main():
    samples = vc.qwen_samples + vc.ministral_samples
    golds = [gold for gold, _ in samples]
    raws = [raw for _, raw in samples]
    loop_cleaned, loop_metrics = ([], [])
    for gold, raw in samples:
        cleaned = vc.clean_generation(raw)
        loop_cleaned.append(cleaned)
        loop_metrics.append((vc.exact_match(cleaned, gold), vc.f1(cleaned, gold), vc.contains_answer(raw, gold)))
    batch_cleaned = [vc.clean_generation(r) for r in raws]
    batch_metrics = [(vc.exact_match(c, g), vc.f1(c, g), vc.contains_answer(r, g)) for c, g, r in zip(batch_cleaned, golds, raws)]
    mismatches = []
    for i, (lc, bc, lm, bm) in enumerate(zip(loop_cleaned, batch_cleaned, loop_metrics, batch_metrics)):
        if lc != bc or lm != bm:
            mismatches.append((i, lc, bc, lm, bm))
    print(f'\n=== parity: loop shape vs literal string list ({len(samples)} samples) ===')
    if mismatches:
        for i, lc, bc, lm, bm in mismatches:
            print(f'  sample {i}: cleaned {lc!r} vs {bc!r}, metrics {lm} vs {bm}')
        print('FAIL: batch-shape consumption diverges from per-sample consumption.')
        return 1
    print('  all 40 cleaned strings identical; all EM/F1/contains values identical')
    qwen_em = sum((m[0] for m in batch_metrics[:20]))
    mini_em = sum((m[0] for m in batch_metrics[20:]))
    qwen_contains = sum((m[2] for m in batch_metrics[:20]))
    mini_contains = sum((m[2] for m in batch_metrics[20:]))
    expected = (4, 7, 19, 19)
    actual = (qwen_em, mini_em, qwen_contains, mini_contains)
    print(f'  aggregates (EM_clean qwen/ministral, contains qwen/ministral): {actual} expected {expected}')
    if actual != expected:
        print('FAIL: aggregates diverged from the validated PR #6 numbers.')
        return 1
    try:
        from harness.pipeline import clean_generation as pipeline_clean
    except ImportError as e:
        print(f'  (harness.pipeline not importable here — {e.name or e} — cross-check against the real clean_generation runs where faiss/sentence-transformers are installed, e.g. the container)')
    else:
        diverged = [r for r in raws if pipeline_clean(r) != vc.clean_generation(r)]
        if diverged:
            print(f"FAIL: harness.pipeline.clean_generation diverges from verify_cleanup's frozen copy on {len(diverged)} sample(s).")
            return 1
        print('  harness.pipeline.clean_generation matches the frozen copy on all 40')
    print('\nPASS: metric layer is shape-agnostic — identical results on a plain string list.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
