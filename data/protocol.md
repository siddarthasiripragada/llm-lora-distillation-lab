# SidSearch Query Planning Protocol

Version: 1.0.0

SidSearch converts a natural-language search request into one deterministic JSON execution plan. It is an original private protocol for this lab and contains no confidential business information. Rule priority is explicit: lower priority numbers run first, and later rules may refine fields only when the earlier rule leaves ambiguity.

## Rules
SS-001 | exact entity preservation | priority=10 | Preserve quoted phrases and proper-name entities exactly as written.
SS-002 | relative date resolution | priority=20 | Resolve today, yesterday, last week, and last month relative to 2026-07-10.
SS-003 | explicit date handling | priority=21 | Convert explicit years and ISO date ranges into start_date and end_date filters.
SS-004 | source routing | priority=30 | Route document, docs, pdf, slide, spreadsheet, and report requests to documents.
SS-005 | ambiguous source handling | priority=31 | If no source can be inferred, set source unknown and require clarification.
SS-006 | missing entity handling | priority=32 | If no searchable entity exists, require clarification.
SS-007 | query expansion | priority=40 | Expand pr to pull request, docs to documents, and repo to repository in rewritten_query only.
SS-008 | quoted phrase preservation | priority=41 | Quoted phrases must remain a single entity.
SS-009 | freshness prioritization | priority=42 | Latest or recent implies freshness priority but must not invent dates.
SS-010 | negation handling | priority=50 | Exclude, except, without, and not indicate exclusion intent.
SS-011 | exclusion filters | priority=51 | Exclusion words may affect status filters only when a status is explicit.
SS-012 | repository intent detection | priority=60 | GitHub, repo, repository, issue, pull request, pr, commit, and branch route to github.
SS-013 | email intent detection | priority=61 | Email, inbox, mail, thread, sender, and unread route to email.
SS-014 | multi-source search | priority=62 | Requests naming more than one source, all sources, or across everything use combined_search and source all.
SS-015 | confidence assignment | priority=70 | High confidence requires source and entity; medium allows multiple sources; low is used for required clarification.
SS-016 | clarification policy | priority=71 | Maybe and not sure lower confidence and may trigger clarification if source or entity is missing.
SS-017 | file-type inference | priority=80 | pdf, docx, xlsx, csv, pptx, md, py, ipynb, and txt set file_type.
SS-018 | status-filter inference | priority=81 | open, closed, merged, draft, unread, and read set status.
SS-019 | output schema enforcement | priority=90 | Every output must include all SidSearch schema fields exactly once.
SS-020 | unsupported assumption prevention | priority=91 | Do not invent sources, owners, dates, statuses, or rule IDs not justified by the request.

## Output Schema

The required JSON object contains intent, entities, source, filters, rewritten_query, confidence, clarification_required, clarification_question, and applied_rules.

