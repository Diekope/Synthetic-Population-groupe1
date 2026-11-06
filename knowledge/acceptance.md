<!-- id: TK-ACCEPTANCE -->
# Acceptance scenarios

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

| Scenario | Required observable result |
|---|---|
| Branch isolation | Only knowledge/, AGENTS.md, README.md, .gitignore in team-knowledge |
| Consistent read | All selected documents from the offered immutable commit |
| Missing/corrupt mandatory document | Validation/offer fails closed |
| Metadata/full-text search | Scope and lifecycle filters apply; mandatory context always offered |
| Wrong session/version/repository acknowledgement | Rejected; previous receipt cannot open a gate |
| Duplicate request or lost response | Stable receipt/attempt ID; status lookup available |
| Restart/compaction | New epoch requires new recorded load |
| Rename/delete/supersede | New snapshot excludes old searchable content; mandatory links remain valid |
| Evidence/policy/approval invalidation | Dependent publication and completion remain pending |
| Concurrent publisher/rejected push | No force push; stale proposal requires reconciliation/review |
| Crash after push | Query remote ancestry/content; reuse durable attempt rather than duplicate publication |
| Missing authorization/review | No publication; exact reviewable diff remains available |
| Wrong principal/state store repository | Access rejected before state use |
| Required knowledge not delivered | Overall completion rejected |
| Offline context | Only explicit cached-safe operations; no publication or overall completion |

These are acceptance requirements, not assertions that every scenario already passed. The implementation
test report identifies actual verified cases. New semantics, additional hosts and authenticated notifications
must add corresponding tests before their status becomes verified.
