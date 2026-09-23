# Thesis skeleton — plan, not prose

This folder is scaffolding: a LaTeX build set up from GISMA's CDS dissertation
template, a real bibliography, and a per-chapter content map (as LaTeX comments
inside each `chapters/*.tex` file — they compile to nothing, they're notes to
yourself). **No chapter has actual dissertation prose in it.** That's
deliberate: per the Module Handbook Sec.6, AI-generated text isn't permitted in
the final submission unless the assessment guidelines explicitly say
otherwise, and that hasn't been confirmed in writing.

## Ground rules for every chapter (applies everywhere, not repeated per file)

1. **No claim without a source.** Every number, every "X is more robust than
   Y," every "this defense works" needs a `\cite{}` key, a
   `THESIS_MASTER_RECORD.md` section number, or a figure/table `\ref{}` behind
   it. If you can't point at where a claim comes from, cut it or flag it as
   your own interpretation explicitly ("this project's data suggests..." not
   "it is known that...").
2. **`nq_open` is excluded, with a reason unique to that corpus — CONFIRMED
   AND APPROVED by your supervisor.** Write it as a settled scope decision,
   not an open discussion. But the approval is not the reasoning — the
   reasoning is, and stays, the specific corpus-construction fact: `nq_open`
   ships with no real supporting passage, so its passage text *is* the gold
   answer by construction (`NQ_OPEN_SCOPE_DECISION.md` Sec.2-3). Evidence:
   mean F1 0.962 (spread 0.005) vs. `hotpot_qa` 0.585 (spread 0.179) and
   `ms_marco` 0.255 (spread 0.042) — near-ceiling AND near-invariant across
   4 models of different capability is the answer-copying signature, not a
   result. Write the F1 evidence as the argument; mention the approval only
   as the procedural closure of that argument, not as the justification
   itself.
3. **The backend-confound discovery gets full documentation, not a footnote.**
   `THESIS_MASTER_RECORD.md` Sec.8 (8.1 how it was found, 8.2 what it
   affected, 8.3 methodological implication) plus `THESIS_ARCHITECTURE.md`
   Sec.5 (5.1-5.4, per-defense breakdown) are the two sources; use both.
4. **Defense findings are precise, per defense, per attack — never "defenses
   helped."** The three defenses do three different things:
   - `output_filter`: 0/19 injection cells, 0/24 PoisonedRAG cells
     significant after backend correction; 0.0000 flag rate against
     PoisonedRAG specifically (structural mismatch — a safety classifier
     cannot see factual poisoning, not a tuning failure).
   - `spotlighting`: 9/16 injection cells, 4/24 PoisonedRAG cells significant
     — the only defense with a confirmed, backend-isolated real effect — but
     mean ΔF1 −0.283 against injection (as low as −0.44 in some cells).
   - `instruction_detection`: 0/16 injection cells significant at n=40, but
     10/10 backend-corrected genuine blocks have a classifier flag behind
     them — underpowered, not disproven, and its n=40 cap (vs. 1000 for the
     other two) is a real compute constraint, not a design choice.
5. **EU AI Act Article 15 is a technical mapping, not a legal argument.**
   Never write "this system complies with Article 15" or "this is legally
   required." Write "this project's evaluation methodology maps onto
   Article 15(5)'s vulnerability classes as follows..." The scoping caveat
   from `PHASE4_EU_AI_ACT_MAPPING.md` Sec.3.3 applies throughout: this
   project does not establish that any deployed RAG chatbot is legally
   Annex-III high-risk — that depends on deployment context this project's
   benchmark scope doesn't specify.
6. **Classify every result as clearly addressed / partially addressed / not
   addressed / out of scope**, per attack-defense-Article15 combination —
   don't leave it as one blended verdict. The exact classifications already
   worked out in `PHASE4_EU_AI_ACT_MAPPING.md` Sec.3.1-3.4:
   - Article 15(1)/(3) accuracy: **partially demonstrated** (measured and
     declared per condition; "throughout lifecycle" implies deployment
     monitoring this fixed-point benchmark cannot test)
   - Article 15(4) fault-resilience: **not tested — named gap**
   - Article 15(5) data poisoning (PoisonedRAG): **directly tested**, with
     the output_filter structural-mismatch finding as the concrete result
   - Article 15(5) adversarial examples (injection): **directly tested**,
     nuanced — no defense achieves both real effect AND utility preservation
   - Article 15(5) model poisoning / confidentiality attacks: **not
     applicable / out of scope** — this project's attacks are all
     inference-time, none touch training data or weights
   - Article 15(5) general clause (Crescendo): **fits, with a drafting
     observation** — Article 15(5)'s illustrative list doesn't name
     multi-turn conversational escalation as a category, even though it
     plainly falls under the general clause
7. **Every claim links to evidence** — a citation, a `THESIS_MASTER_RECORD.md`
   section+number, a figure/table label, or (per Approach chapter) a specific
   code path/snippet. No floating assertions.
8. **Citations follow the handbook's Harvard-style requirement** —
   `attachments/bibliography.bib`'s `apalike` style is author-year, which is
   the closest built-in match; confirm with your supervisor this satisfies
   "Harvard referencing" as the handbook states it, or adjust the
   `\bibliographystyle{}` in `main.tex` if a stricter Harvard package is
   required.
9. **Related Work chapter title in the handbook = the Literature Review CDS
   requires.** The CDS structure calls it "Related Work," not "Literature
   Review" — same chapter, same requirement, just the CDS-specific name.
10. **Every figure and table gets a real caption AND an in-text
    cross-reference** — never a bare `\includegraphics{}`. Use:
    ```latex
    \begin{figure}[htbp]
      \centering
      \includegraphics[width=0.9\textwidth]{fig03_injection_asr_model_template}
      \caption{Indirect injection ASR by model and template, split by
        goal-hijack vs. process-hijack (Section 5.1 of \texttt{THESIS\_MASTER\_RECORD.md}).}
      \label{fig:injection-asr}
    \end{figure}
    ```
    and reference it in prose: "as shown in Figure~\ref{fig:injection-asr}, ...".
    Tables follow the same pattern with `\begin{table}...\caption{}...\label{}`.
11. **Headings link to Contents automatically** via `\tableofcontents` — no
    action needed for chapter/section headings. Figures and tables are
    listed separately, under **Appendices** (`chapters/appendices.tex`, new
    file, included after References), via `\listoffigures`/`\listoftables`
    with explicit `\addcontentsline{toc}{chapter}{...}` so they show up as
    their own linked entries in the main Table of Contents.
12. **Architecture diagrams from `docs/THESIS_ARCHITECTURE.md`** belong in
    Approach, not Foundations — see that chapter's map for which specific
    sections to reproduce.
13. **Figure and table captions all follow one plain-descriptive style — no
    punchy or title-style captions.** State what the figure/table shows,
    under what condition or method, and its scope, in one flat sentence
    (e.g., "Structure of the dissertation, chapters in reading order." for
    `fig:dissertation-structure`, the only one written so far). Never mix
    a colon-led headline style ("Chapter roadmap: ...") in for some
    captions and plain description for others — consistency across the
    List of Figures / List of Tables matters more than any single
    caption's phrasing.

## What's here

```
thesis/
├── main.tex                    ← GISMA CDS template, adapted; title/student/
│                                  degree/supervisor filled in, \graphicspath
│                                  wired to attachments/ and ../results/figures/
├── attachments/
│   ├── logo.png, campus.jpg     ← from the template repo
│   └── bibliography.bib         ← 33 real entries from docs/CITATIONS.md
└── chapters/
    ├── declaration.tex, dedication.tex, abstract.tex
    │                            ← front matter, in that order (per your
    │                              instruction: Declaration -> Dedication ->
    │                              Abstract). Acknowledgments removed — not
    │                              part of the CDS structure list.
    ├── introduction.tex, foundations.tex, related_work.tex, approach.tex,
    │   evaluation.tex, conclusion.tex
    │                            ← body chapters, structural stubs + detailed
    │                              content-map comments, no prose
    └── appendices.tex            ← NEW: List of Figures / List of Tables,
                                    both linked in the main Table of Contents
```

## Word budget (16,000 words total, M599/120 ECTS, solo)

| Chapter | Target words | Confidence |
|---|---|---|
| Introduction | ~1,300 | source material exists (README, Master Record Sec.1-2) |
| Foundations | ~1,800 | **biggest gap** — closest to written-from-scratch |
| Related Work | ~3,200 | sources are real (33 peer-reviewed entries); needs synthesis |
| Approach | ~2,400 | best-documented — mostly assembly |
| Evaluation and Results | ~3,200 | best-covered — 297 verified assertions behind every number |
| Conclusion | ~1,500 | needs honest limitations, not just a summary |
| Front matter (declaration, dedication, abstract) | ~300-500 | personal/administrative; write last |

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
- ~~`nq_open` exclusion pending supervisor sign-off~~ — **confirmed and
  approved.** Write it as settled in Evaluation and Conclusion; the technical
  reasoning (corpus-construction fact, F1 evidence) still belongs in the text
  in full — approval isn't a substitute for stating why.
- `declaration.tex` needs GISMA's actual required wording — don't improvise it.
- `attachments/bibliography.bib`'s DOI/URL links are as of September 2026,
  per `docs/CITATIONS.md`'s own note — re-verify before final submission.
- Confirm with your supervisor that `apalike` (author-year) satisfies the
  handbook's "Harvard referencing" requirement, or switch styles.
