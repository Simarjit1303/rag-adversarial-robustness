# Thesis skeleton — plan, not prose

This folder is scaffolding: a LaTeX build set up from GISMA's CDS dissertation
template, a real bibliography, and a per-chapter content map (as LaTeX comments
inside each `chapters/*.tex` file — they compile to nothing, they're notes to
yourself). **No chapter has actual dissertation prose in it.** That's
deliberate: per the Module Handbook §6, AI-generated text isn't permitted in
the final submission unless the assessment guidelines explicitly say
otherwise, and that hasn't been confirmed in writing yet.

## What's here

```
thesis/
├── main.tex                    ← adapted from the GISMA CDS template;
│                                  title/student/degree/supervisor filled in,
│                                  \graphicspath wired to attachments/ and
│                                  ../results/figures/, inputenc+fontenc added
│                                  (the original template lacked them, which
│                                  would break the bibliography's accented
│                                  author names — Jégou, Röttger, Kıcıman, Tramèr)
├── attachments/
│   ├── logo.png                ← copied from the template repo, as asked
│   ├── campus.jpg               ← copied too — main.tex's title page needs it
│   │                              to compile (used as a faint watermark)
│   └── bibliography.bib         ← 33 real entries, transcribed from
│                                  docs/CITATIONS.md — not the template's
│                                  95 generic placeholder entries
└── chapters/
    ├── abstract.tex, declaration.tex, acknowledgments.tex, dedication.tex
    │                            ← structural stubs; these are personal/
    │                              administrative, nothing to source-map
    └── introduction.tex, foundations.tex, related_work.tex, approach.tex,
        evaluation.tex, conclusion.tex
                                 ← structural stubs + a detailed content-map
                                   comment block in each, pointing at exactly
                                   which docs/ sections, figures, and
                                   bibliography keys belong in that chapter
```

## Word budget (16,000 words total, M599/120 ECTS, solo)

| Chapter | Target words | Confidence |
|---|---|---|
| Introduction | ~1,300 | source material exists (README, Master Record §1-2) |
| Foundations | ~1,800 | **biggest gap** — closest to written-from-scratch |
| Related Work | ~3,200 | sources are real (33 peer-reviewed entries); needs synthesis |
| Approach | ~2,400 | best-documented — mostly assembly |
| Evaluation and Results | ~3,200 | best-covered — 297 verified assertions behind every number |
| Conclusion | ~1,500 | needs honest limitations, not just a summary |
| Front matter (abstract, declaration, acknowledgments, dedication) | ~300-500 | personal; write last |

## Suggested build order

1. Approach and Evaluation and Results first (least new synthesis, highest confidence)
2. Related Work (organize the 33 existing citations by attack/defense family)
3. Foundations (the one chapter closest to a blank page)
4. Introduction and Conclusion last (both need to reference what actually ended up elsewhere)
5. Front matter last of all

## Still open

- **AI-use policy for this module isn't confirmed in writing.** Get that from
  your supervisor or the module leader before treating anything beyond this
  skeleton as settled.
- `declaration.tex` needs GISMA's actual required wording — don't improvise it.
- `attachments/bibliography.bib`'s DOI/URL links are as of September 2026,
  per `docs/CITATIONS.md`'s own note — re-verify before final submission.
