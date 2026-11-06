<!-- id: TK-DECISIONS -->
# Decisions and unresolved questions

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

These design choices are proposals until reviewed. User-directed branch isolation is a task constraint,
not evidence that every architectural interpretation has been accepted.

| ID | Proposal | Rationale | Status |
|---|---|---|---|
| D-001 | Knowledge-only team-knowledge branch | Explicit user request; no application tree merge | Proposed, branch scope authorized |
| D-002 | Shared stdlib Python client on separate implementation branch | Existing Python toolkit; avoid host-specific duplicated logic | Proposed |
| D-003 | Local SQLite lifecycle store | Durable attempts and idempotency without an extra service | Proposed |
| D-004 | Human review for all publication classes initially | No approved automatic-observation policy | Proposed |
| D-005 | Metadata plus literal full-text retrieval first | Establish correctness before advanced retrieval | Proposed |

Unresolved: who reviews and publishes in a multi-user team; whether SSH signatures or an external
review system should replace local attestations; where a future authenticated service runs; which
coding host can expose actual model-request hooks; acceptable freshness duration for long-running
operations; whether survey labels reflect the true dataset semantics.

Do not assume eventual human approval. Rejection and narrowing are valid terminal review outcomes.
