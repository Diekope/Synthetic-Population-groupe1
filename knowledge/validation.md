<!-- id: TK-VALIDATION -->
# Validation evidence and formal scope

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

The baseline source observation is [baseline.json](evidence/baseline.json), with claims in
[claims.json](claims.json). It records source structure at a fixed revision, not a new live model benchmark.
The implementation branch carries executable tests. Its final validation evidence must name the exact
implementation commit, dirty-source digest if applicable, command, date, exit code, environment and limits.
Initial setup evidence and publication handoff are separate from semantic acceptance.

The bounded protocol model on the implementation branch explores receipt freshness, approval,
publication authorization, invalidation, remote verification and completion. Invariants: gated actions
require current receipts; reviewed publication requires matching approval and authorization; required
knowledge completion requires remote verification; invalidation removes validity. Negative controls must
produce counterexample traces. This is finite explicit-state invariant checking, not a general CTL/LTL engine.

Failure transitions must include loss of receipt, restart/compaction, evidence invalidation, changed policy,
push rejection and uncertain push response. Liveness requires eventual connectivity, fair retries and stable
inputs; review may stay pending or be rejected forever. No eventual-approval assumption is allowed.

Model results prove only modeled properties over the stated finite state space. No automatic correspondence
proof to Python, arbitrary natural-language truth or broad host compatibility is claimed. Remote real-project
publication and real LLM transport need separate dated evidence; local bare-repository tests are not deployment.

Dated implementation result: [EV-KNOWLEDGE-001](evidence/implementation.json). The 25 new tests
and 55 existing tests passed. The finite model explored 161 states and 2371 transitions; all
three intentionally broken controls produced counterexamples. Human acceptance remains pending.
