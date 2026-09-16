# Supervision Meetings Log

Consolidated record of every supervisor meeting (Prof. Loui Al Sardy /
Simarjit Singh), sourced primarily from the real meeting-record files in
the parent project folder — **outside this git repository** — with the
git repo's own task-brief references used only as a secondary,
corroborating source, exactly as requested.

**Search performed:** direct filesystem listing (not `git`) of
`.../Simarjit Singh/` and its subfolders — `Codes/` (this repo),
`Drafts/`, `Final_Submission/`, `Progress_Overview/` — followed by a
targeted `find`/`grep` for anything meeting-related by filename or
content. Full results in [Sources](#sources) below.

---

## Meeting log

| No. | Date | Status | Topics | Decisions / Outcomes |
|---|---|---|---|---|
| 1 | **01/06/2026** | Held, logged | Introductions and supervision expectations; proposed dissertation topic and research motivation; initial research questions, objectives, expected contributions; dissertation research type and methodology; technical contribution requirements, implementation expectations, evaluation strategy; timeline/milestones; ethics requirements and approval process | "Introduction, Discussion of proposal, Research approach, next meeting agenda." Next: prepare a 6-slide presentation for 15 June (intro, motivation, RQ+objectives, related work, architecture diagram, evaluation, references). |
| 2 | **15/06/2026** | Held, logged | Research and review of more papers/topics similar to thesis scope; implementation of proposed plans. (Prep deck covered: introduction/motivation, main RQ + 3 sub-questions, related-work landscape table, evaluation setup — datasets/metrics/statistics plan.) | "Discussion of proposed plan, Research papers, next meeting agenda." Outcome recorded in the dissertation plan: "read more 2026 papers, then start Phase 1." Next: presentation on latest research papers + Phase 1 implementation, targeted for 10 July. |
| 3 | **05/08/2026** | Held, logged | Review of Phase 1 results and feedback; Phase 2 progress update; technical implementation and challenges encountered; review of next steps and dissertation timeline; Q&A | Docx (terse): "Discussion of Phase 1 results, technical implementation and challenges, Phase 2 status, QnA, and Next steps." Corroborated in more detail by `PHASE2_ROADMAP.md`: **5 poisoned passages per query** confirmed as PoisonedRAG's poison count; **goal-hijacking vs. process-hijacking** confirmed as the injection ASR framing; **"Crescendo-via-retrieval"** named as this dissertation's own novel contribution (no prior work adapts Crescendo to a RAG pipeline). Next meeting: "to be scheduled after finishing Phase 4" (per the docx itself). |
| 4 | **Not recorded — confirm exact date with supervisor** | ⚠️ Occurred per corroborating documentation, but no primary meeting-minutes file found anywhere in the searched folder tree | Planned window per the dissertation timeline: 23 Aug – 5 Sep 2026 ("Phase 4 evaluation, ASR judging, significance testing, matrix build"). Content substantively corroborated by `PHASE2_ROADMAP.md` (updated 2026-09-04), which repeatedly cites "Meeting 4 feedback, direct from the supervisor": work one attack fully — build → sweep → written insight extraction — before the next attack's real sweep is taken seriously (the same discipline applies to the dissertation's final synthesis chapter); the competitor list is resolved to **SafeRAG, RSB, and Hidden-in-Plain-Text** (Formalizing-PI dropped in Hidden-in-Plain-Text's favor; TrustRAG excluded as a defense method, not a rival benchmark); Phase 4's EU AI Act Article 15(1)/15(5) mapping "needs the same real interpretation, not a results table with a regulation number attached to it." | **No `.docx`/minutes file exists for this meeting** in `Progress_Overview/` (only meetings 1–3 do) or anywhere else searched. Its content is well-corroborated by a document that post-dates it, but its exact date is not independently confirmable from a primary record — flagged per your instruction rather than guessed. |
| 5 | **Not yet recorded — likely the upcoming/current meeting** | Not yet held per any found record | Planned window per the dissertation timeline: 6–19 Sep 2026 ("Draft all eight chapters, figures, results write-up"). | Today's date falls inside this planned window — this is plausibly the meeting this repo's recent work (`THESIS_MASTER_RECORD.md`, `THESIS_ARCHITECTURE.md`, `CITATIONS.md`, `NQ_OPEN_SCOPE_DECISION.md`) was prepared for. **Confirm date and record outcome with supervisor after it happens.** |
| 6 | **Not yet recorded** | Not yet held | Planned window per the dissertation timeline: 20–26 Sep 2026 — "Full-draft review, references, repo cleanup, viva prep," milestone **"submit."** | Final pre-submission meeting. Not yet occurred. |

**Two meetings remain before submission — Meeting 5 and Meeting 6 — matching your own count.**

---

## Sources

**Primary (real meeting-record files, outside this git repo):**

- `Progress_Overview/01_01_06_2026_Meeting.docx` — Meeting 1, read in full (docx `word/document.xml` extracted directly; structured template: Meeting Details / Agenda Items / Key Decisions / Issues Identified / Next Meeting & Notes).
- `Progress_Overview/02_15_06_2026_Meeting.docx` — Meeting 2, read in full, same template.
- `Progress_Overview/03_05_08_2026_Meeting.docx` — Meeting 3, read in full, same template.
- `Drafts/supervisor_meeting_2_presentation.md` — the presentation prepared for Meeting 2 (not minutes of what was decided there, but the content brought to it).
- `Drafts/Final_Dissertation_Plan_v3.md` — Section 12 (dissertation timeline table, meeting-by-meeting) and Section 14 ("Supervision log status," which explicitly logs Meetings 1–2 and points to Section 12 for Meetings 3–6's planned schedule). This document is dated as of 10 July 2026 and records the *plan*, not post-hoc outcomes for meetings 3 onward.
- `Drafts/PHASE2_ROADMAP.md` — updated 2026-09-04; cited as the corroborating source for Meeting 3 and Meeting 4's substantive content (quoted above), since neither meeting's own docx goes into that level of technical detail.

**Files found but not usable as meeting-content sources:**

- `Drafts/files/supervisor_meeting2_slides.html`, `Drafts/files/supervisor_meeting3_slides.html` — exist (found by filename search) but not opened for this log, since the corresponding presentation-prep `.md`/docx sources already cover the same ground; flagged here for completeness in case they contain additional detail worth a follow-up pass.

**Secondary source, git repo itself (as requested):**

- `git log --all --oneline | grep -i meeting` — **zero results**, no commit message references a meeting.
- `grep -rn "Meeting [0-9]" --include="*.md" .` inside the repo — two files, both task briefs: `phase2_indirect_injection_task.md:45` ("per Meeting 4 supervisor feedback") and `phase2_poisonedrag_task.md:25,28` ("confirmed in Meeting 3... referenced in `PHASE2_ROADMAP.md`"). Both point back to the same external `PHASE2_ROADMAP.md` already used as a primary corroborating source above — the repo's own task briefs never contain meeting content directly, only references to it.
- `THESIS_MASTER_RECORD.md` Section 11 previously stated only that "Task briefs reference 'Meeting 3' and 'Meeting 4' in passing... no consolidated meeting log... was found in this repository" — this file is the resolution of that gap, built from real records outside the repo, not a repo-only guess.

**Not found anywhere in the searched tree:** any `.docx`/`.pdf`/minutes file for Meeting 4, 5, or 6.

---

*Update this file after Meeting 5 and Meeting 6 actually happen — replace their "not yet recorded" rows with real content the same way Meetings 1–3 are documented here, sourced from whatever meeting-record file gets created for each (matching the `NN_DD_MM_YYYY_Meeting.docx` naming convention already used in `Progress_Overview/`).*
