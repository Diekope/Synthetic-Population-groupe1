# Team knowledge client

The shared client lives on `codex/team-knowledge-service`. Reviewed shared knowledge belongs only to
the knowledge-only `team-knowledge` branch. Do not merge that tree into application branches.
The initial knowledge contracts are drafts awaiting review. This document describes executable
tool usage and limits; detailed policies remain at `team-knowledge:knowledge/`.

## Structure and dependencies

Python 3.12+ and Git are required; no new Python package is required.
`knowledgekit/source.py` reads immutable Git objects and applies metadata/literal full-text filters.
`service.py` stores tasks, proposals, requests, receipts and attempts in SQLite. `hosts.py` instruments
the existing project chat host. `__main__.py` exposes one shared CLI.
`tools/check_knowledge_model.py` checks an abstract finite protocol model.
`tests/test_knowledge.py` uses synthetic data and isolated local bare repositories.

Local `.knowledge-local/` contains access policy, state and recorded model requests. Protect it with
OS permissions; do not share it or put secrets in test data. `.local/` holds optional private notes.
Neither is shared knowledge. There is no deployed network service, webhook, scheduler, persistent
search index or semantic retrieval. Full-text search scans pinned blobs, so deleted content naturally
disappears at the next snapshot. The tool does not intercept arbitrary shell operations.

## Establish and validate the source

Run from an implementation checkout. The knowledge branch can be fetched independently:

```bash
git fetch origin team-knowledge
git branch team-knowledge FETCH_HEAD
python -m knowledgekit validate
python -m knowledgekit init
python -m knowledgekit search "survey" --scope project
```

If `team-knowledge` already exists locally, inspect it rather than overwriting it. Bootstrap a new
knowledge-only local root with `python -m knowledgekit bootstrap <candidate-directory>`; it validates
the tree before creating the reference and never copies application files. This command does not push.
The candidate directory must contain the full intended knowledge-only tree.

## Instrument one project host

```bash
python -m knowledgekit start TASK-001 SESSION-001 --scope project --query "survey"
python -m knowledgekit freshness TASK-001 --boundary start
python -m knowledgekit host-request TASK-001 "Explain the supplied context" --agent 1 --provider mock
python -m knowledgekit freshness TASK-001 --boundary consequential
python -m knowledgekit gate TASK-001 --operation read
```

`host-request` also supports the existing `openai` and `ollama` providers; explicitly choose a real
model and configure their existing environment variables. Real requests use the normal provider
transport. Instrumentation records exactly the JSON payload supplied to that transport, before network
processing; it does not establish remote acceptance, understanding or retention. Mock mode proves wiring.
Use one request per process. This is a project chat-host integration, not native Codex task interception.
In guidance-only coding hosts, run explicit tools and never claim automatic context consumption.

After restart, compaction, truncation or context loss, use a new session epoch and reload. Existing
result/proposal state is retained, but prior receipts and boundary observations are cleared.
All detected source updates are conservatively relevant in version 1. `reconcile` queries the actual
remote branch and fetches immutable objects, without rewriting the active code checkout.
`freshness --offline` allows only policy-listed read/test/propose operations; publication and completion
are blocked. On reconnect, observe the boundary again. No event-driven notification adapter exists yet.

## Record results and prepare review

```bash
python -m knowledgekit result TASK-001 complete
python -m knowledgekit freshness TASK-001 --boundary consequential
python -m knowledgekit propose TASK-001 <candidate-directory> <evidence-json>
python -m knowledgekit status proposal <proposal-id>
git diff <base-knowledge-commit> <candidate-commit>
```

The CLI returns base/candidate commits, content/evidence/policy digests and exact proposal fingerprint.
Candidate files become Git objects, not application-index changes. Tracked edits and untracked digests
record the implementation snapshot. Review the diff and evidence before the following separate steps:

```bash
python -m knowledgekit review <proposal-id> <fingerprint> approved <human-review-reference>
python -m knowledgekit authorize <proposal-id> <fingerprint> <publication-authorization-reference>
python -m knowledgekit publish <proposal-id>
```

Only execute these commands with actual matching review and publication authorization. Review commands
are OS-principal attestations with external references, not proof of human identity. No automated-observation
publication policy is enabled. A new policy version, changed implementation or stale source requires new
review. `invalidate <task> <reason>` invalidates receipts and linked review; collect replacement evidence
under a new task/proposal. Old audit entries remain available in the local SQLite store.

Publication uses a durable proposal-keyed attempt and ordinary non-force Git push. Recovery queries
remote ancestry and intended file contents before retrying, including removed paths. An unchanged
remote tip is not assumed from tracking refs. Remote verification is timestamped, not continuous.
The source branch reference is updated after successful publication and recovered publication if
it still equals the reviewed base. If a concurrent local change moved it, inspect and reconcile
that change, then reload the session; the tool does not overwrite it.

## Completion

After a changed knowledge revision, start a new epoch and record the new host request before completion:

```bash
python -m knowledgekit start TASK-001 SESSION-002
python -m knowledgekit host-request TASK-001 "Load current project context" --agent 1
python -m knowledgekit freshness TASK-001 --boundary completion
python -m knowledgekit complete TASK-001
```

Required knowledge must have a remotely verified proposal. For genuinely unchanged shared knowledge,
use `result TASK-001 complete --no-update-reason "<specific reason>"` instead. Code and knowledge delivery
remain separate. A local handoff or open PR does not satisfy required publication.

## Validation and known limits

```bash
python -m unittest discover -s tests -p test_knowledge.py -v
python tools/check_knowledge_model.py
```

Tests cover isolated Git publication, contention, rejected pushes, uncertain outcomes, context and review
gates, repository/principal isolation, offline behavior and the existing chat transport hook. They do not
prove secure multi-user review, real LLM quality or operational deployment. The model checks only its
finite abstraction, with explicit negative-control traces; it does not establish correspondence to Python.
Lifecycle operations use a cross-process OS file lock around read-modify-write work, with short separate
SQLite transactions. Publication contention is rejected by ordinary Git rules. Lock timeout requires a
status query and retry; it never implies success. A multi-user network server needs authenticated identities
and signed external review before production use. Model-request state retention and evidence revalidation also
need a team policy before broader deployment.
