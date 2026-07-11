# Presentation-Friendly SidSearch Examples

These examples are fictional and generic. They do not use your GitHub profile, personal projects, private repositories, or real business data.

## Where The Training Files Are

The generated dataset files live here:

```text
data/seed_scenarios.jsonl
data/train.jsonl
data/validation.jsonl
benchmarks/heldout.jsonl
```

The code that generates example names is here:

```text
src/sidsearch_lora_lab/data/generator.py
```

## Better Demo Story

Imagine a company has an internal search tool called SidSearch. Employees ask questions across documents, email, and GitHub. The model must convert those plain-English requests into strict JSON execution plans.

The fictional sources are:

- documents
- email
- GitHub
- all sources

The fictional project/entity names are:

- NorthstarCRM
- CloudLedger
- PolicyVault
- SupportPulse
- ReleaseRadar
- ContractHub
- MetricsBridge

The fictional people/owners are:

- Alex
- Morgan
- Taylor
- Jordan
- Casey

## Example 1: Document Search

Input:

```text
Find pdf documents about NorthstarCRM
```

Expected model behavior:

```json
{
  "intent": "document_search",
  "entities": ["NorthstarCRM"],
  "source": "documents",
  "filters": {
    "start_date": null,
    "end_date": null,
    "owner": null,
    "file_type": "pdf",
    "status": null
  },
  "confidence": "high",
  "clarification_required": false
}
```

Interview explanation:

> This tests whether the model can identify a document-search request, preserve the business entity, infer the file type, and return a structured JSON plan.

## Example 2: Email Search

Input:

```text
Search unread email from Morgan about CloudLedger
```

Expected model behavior:

```json
{
  "intent": "email_search",
  "entities": ["CloudLedger"],
  "source": "email",
  "filters": {
    "start_date": null,
    "end_date": null,
    "owner": "Morgan",
    "file_type": null,
    "status": "unread"
  },
  "confidence": "high",
  "clarification_required": false
}
```

Interview explanation:

> This tests source routing, owner extraction, status filtering, and entity extraction.

## Example 3: Repository Search

Input:

```text
Find open GitHub issues about ReleaseRadar
```

Expected model behavior:

```json
{
  "intent": "repository_search",
  "entities": ["ReleaseRadar"],
  "source": "github",
  "filters": {
    "start_date": null,
    "end_date": null,
    "owner": null,
    "file_type": null,
    "status": "open"
  },
  "confidence": "high",
  "clarification_required": false
}
```

Interview explanation:

> This shows the private protocol can map repository language like GitHub/issues/open into a machine-readable execution plan.

## Example 4: Relative Date Handling

Input:

```text
Find recent documents about PolicyVault from last week
```

Expected model behavior:

```json
{
  "intent": "document_search",
  "entities": ["PolicyVault"],
  "source": "documents",
  "filters": {
    "start_date": "2026-07-03",
    "end_date": "2026-07-10",
    "owner": null,
    "file_type": null,
    "status": null
  },
  "confidence": "high",
  "clarification_required": false
}
```

Interview explanation:

> This tests whether the model learned the protocol rule for resolving relative dates into explicit ISO dates.

## Example 5: Combined Search

Input:

```text
Search email and GitHub for SupportPulse
```

Expected model behavior:

```json
{
  "intent": "combined_search",
  "entities": ["SupportPulse"],
  "source": "all",
  "filters": {
    "start_date": null,
    "end_date": null,
    "owner": null,
    "file_type": null,
    "status": null
  },
  "confidence": "medium",
  "clarification_required": false
}
```

Interview explanation:

> This tests whether the model can detect that the user wants multiple sources, not just one.

## Example 6: Clarification Required

Input:

```text
Find ContractHub
```

Expected model behavior:

```json
{
  "intent": "clarification_required",
  "entities": ["ContractHub"],
  "source": "unknown",
  "filters": {
    "start_date": null,
    "end_date": null,
    "owner": null,
    "file_type": null,
    "status": null
  },
  "confidence": "low",
  "clarification_required": true,
  "clarification_question": "Which source should SidSearch use?"
}
```

Interview explanation:

> This is important because a good system should not guess when the source is ambiguous. It should ask a clarification question.

## Best Demo Example For Interview

Use this one:

```text
Find open GitHub issues about ReleaseRadar from last month
```

Why it is good:

- clearly fictional
- easy to understand
- demonstrates entity extraction
- demonstrates GitHub/repository routing
- demonstrates relative date resolution
- demonstrates status filtering
- produces objective JSON

## Important Note

These examples are for presentation. The actual benchmark result must still come from:

```text
benchmarks/heldout.jsonl
```

Do not claim improvement from hand-picked examples. Use them only to explain the task.
