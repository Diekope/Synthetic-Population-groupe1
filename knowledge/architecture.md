<!-- id: TK-ARCH -->
# Architecture and responsibility boundaries

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

Proposed authority: reviewed Markdown and structured claims on `team-knowledge`. This branch starts
from an empty tree and contains no application source, datasets, generated agents or service code.
The application and tooling branches reference it through one shared client; they do not maintain copies.
Existing application READMEs remain primary usage references for their respective implementation revisions.

| Area | Location and owner | Authority |
|---|---|---|
| Shared development knowledge | team-knowledge:knowledge/ | Reviewed Git documents |
| Private notes and handoffs | ignored .local/ | Personal, never shared authority |
| Population and agent runtime data | Application branches: agents/, surveys/ | Application output, not team knowledge |
| Retrieval caches | Disposable local process cache; no persistent semantic index yet | Rebuild from pinned Git objects |
| Tasks, offers, receipts, proposals, attempts | ignored .knowledge-local/state.sqlite | Local service state, not document authority |
| Service and adapters | codex/team-knowledge-service:knowledgekit/ | Executable implementation with its own tests |

The service reads Git blobs at immutable commits and keeps SQLite transactions short. Model calls,
Git fetch/push and human review occur outside write transactions. The current service is scoped to
one OS principal; OS filesystem protection is required. There is no network multi-user service.
Metadata filtering is retrieval, never authentication. Markdown editors are optional views.
