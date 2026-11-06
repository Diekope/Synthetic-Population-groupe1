<!-- id: TK-WORKFLOW -->
# Contribution and publication workflow

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

1. At task start, inspect repository instructions, the active code branch and dirty changes. Announce
   task/session/code snapshot; select pinned context and record the host load. Check freshness at
   start, phase changes, before consequential actions and before completion.
2. Complete or roll back code/configuration/build/workflow work. Collect the actual diff and tests.
   Assess shared knowledge impact. A no-update result needs a concrete reason; otherwise prepare
   a structured [proposal](templates/proposal.md).
3. Stage the full candidate knowledge tree in an isolated directory, not an application index.
   Validate it. Record base knowledge SHA, implementation SHA and dirty scope, affected documents,
   claim/evidence IDs and digests. Let reviewers inspect `git diff <base> <candidate>`.
4. Obtain explicit semantic review and publication authorization bound to the proposal fingerprint.
   Missing review or access leaves delivery pending. Local commits and handoffs do not complete delivery.
5. Refresh the actual remote baseline, reject stale proposals and reconcile concurrent updates.
   Keep knowledge-only commits separate from implementation commits. Never merge this tree into an
   application branch; never force-push normal contention.
6. Persist the attempt/idempotency key before push. On uncertain outcomes, inspect actual remote state
   before retrying. Query the remote reference, fetch its immutable commit, check ancestry and intended
   file content including deletions. Record verification time and SHA.
7. Reload changed mandatory context, record a new request/receipt and check completion freshness.
   Overall delivery requires code complete/rolled back and knowledge either concretely not required,
   or published and remotely verified. Publishing a draft does not accept its semantics.

Private progress belongs in ignored `.local/PROJECT_PROGRESS.zh-CN.md`. It is separate from shared
knowledge delivery. The local tooling initializes a per-OS-user policy without auto-publication;
that file is configuration, not a replacement authority for the reviewed team policy above.
