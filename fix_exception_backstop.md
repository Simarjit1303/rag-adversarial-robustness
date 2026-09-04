# Final backstop: catch anything the 5 known failure sites don't

## Why this is separate from Bug 4, even though it's tiny

Bug 4's audit found every failure path *currently written into this
file* — 5 sites, verified via grep, real and thorough for what it covers.
But it can only cover code that exists and was checked. It can't cover an
exception type nobody's hit yet: a `KeyError` from an env var that isn't
one of the ones explicitly checked, an `OSError` if the Network Volume
runs low on space, some exception raised deep inside `torch` or
`sentence-transformers` that nobody's triggered before. Any of those would
crash exactly the same way — process exits, RunPod restarts, full paid
re-run — without ever touching one of the 5 known sites, because it isn't
one of the 5 known sites.

This has now been the actual pattern three times this session: fix a
specific failure, the next real run finds a different one adjacent to it.
This task closes the *category*, not another instance.

## The fix

```python
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _fail_and_idle(f"unhandled exception: {e!r}")
```

This does not replace Bug 4's 5 specific fixes — those still matter,
because they produce clear, specific error messages at each known failure
point rather than a generic "something broke" message. This is purely the
backstop underneath all of them, for whatever isn't a known site yet.

Use `Exception`, not a bare `except:` — deliberately still let
`KeyboardInterrupt`/`SystemExit` propagate normally if this is ever run
interactively, rather than swallowing genuine intentional interrupts too.

## Testing (no GPU needed)

Simulate an exception type that isn't one of the 5 known sites — e.g.
patch something arbitrary in the pipeline call chain to raise a
`RuntimeError` with an unrelated message — and confirm it's caught by
this outer handler, `_fail_and_idle()` is reached, and the process does
not exit. This is specifically testing the *unknown* case, not
re-testing the 5 already-covered ones.

## Verification

Same PR, same branch. State plainly in the PR description that Bug 4's 5
specific fixes and this backstop are complementary, not redundant — the
specific fixes give useful error messages for known cases, this catches
whatever the specific fixes don't yet know about.
