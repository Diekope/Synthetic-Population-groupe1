<!-- id: TK-PROTOCOL -->
# Context synchronization protocol

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

Contract version: 1. Deterministic JSON: UTF-8 without BOM, sorted object keys, no insignificant
spaces, ensure_ascii=false; document blobs use UTF-8 without BOM and LF. Hash SHA-256 over exact
bytes. Length is byte length, never platform character count. Manifest order is ascending document ID.
Sections are `whole-document` in version 1. Unsupported schemas and corrupt/missing mandatory blobs fail closed.

Announce supplies repository identity, task ID, fresh session epoch, code base SHA, tracked diff digest,
changed paths, untracked digests, requested scope/query and previous consumed knowledge revision.
Offer pins an immutable knowledge commit and returns mandatory guidance plus relevant full-text results,
their IDs/paths/sections/hashes/lengths and manifest digest. Mandatory content is not semantically filtered out.
The previous revision is reported for comparison; automated semantic change summaries are not implemented.
Acknowledge comes from the host adapter's recorded model request and binds repository, task, epoch,
offer and manifest to a request digest. Receipts are durable and idempotent for identical requests.
An offer delivered to disk is not acknowledgement. Additional retries/status queries are allowed.

Track separately: remote tip observation; offered/delivered knowledge revision; supplied request manifest;
acknowledgement receipt. A new epoch supersedes the previous task session and its receipts. Wrong epoch,
offer or manifest is rejected. Status lookup recovers lost responses without assuming failure.

Freshness is a dated boundary observation. Each batch uses one commit. Any detected knowledge update is
conservatively treated as relevant by this first implementation; reload before the next gate. More selective
unrelated-change classification is proposed only. Code changes invalidate previous boundary observations.
Offline cached context allows only policy-listed read/test/propose operations; publish and overall completion
are blocked. Never claim remote freshness while offline. Explicit reconcile queries remote state.

Lifecycle gates apply only to shared-client operations. They cannot block arbitrary shell commands or an
uninstrumented host. SQLite stores state; Git stores documents; neither is a universal synchronization lock.
Remote notifications are proposed: authenticated events should trigger reconcile, deduplicate by event ID,
tolerate reordering and use periodic/on-demand reconciliation for missed events. They must not mutate
a running model's context. Only on-demand reconciliation exists today.
