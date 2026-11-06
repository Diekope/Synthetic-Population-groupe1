<!-- id: TK-STATUS -->
# Branch-aware project status

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

Observed application baseline: repository `https://github.com/Diekope/Synthetic-Population-groupe1`, branch `feature/agents-toolkit`,
immutable implementation revision `6a858cb5e5a3fef119fd3153af8f1be917abbf2b`. The application checkout is separate from this tree.
`main` baseline is `34bf110b41a1325befd39c2d9f0ce7b858cbe5ef`.
These are dated observations, not a claim about continuously changing remote heads.

| Area | Baseline state | Limits |
|---|---|---|
| Java reconstruction | Implemented on both application branches | No new solver run in this knowledge setup |
| Python agents toolkit | Implemented on feature/agents-toolkit | 500 identities represent 302 base profiles |
| Labels A0..A11 | Author-supplied plausible interpretation | Original real-world semantics are unknown |
| Team knowledge contracts | Proposed on team-knowledge | Initial semantic review pending |
| Knowledge automation | Implemented locally at `b6c040a7c22ee887a891ec93769cdeb2c9b9f755` on codex/team-knowledge-service | 80 tests passed; no deployed service or live LLM verification |
| Notification/semantic retrieval | Proposed only | No webhook, scheduler or embedding service deployed |

Knowledge revision is the commit selected by the reader, never the implementation SHA above.
Source claims live in [claims.json](claims.json); evidence lives in [baseline.json](evidence/baseline.json).
For uncommitted implementation changes, record base commit plus tracked diff digest, changed paths
and untracked file digests. Never say the base commit contains those edits.

Local implementation validation: [implementation evidence](evidence/implementation.json).
Code commit: `b6c040a7c22ee887a891ec93769cdeb2c9b9f755`. Knowledge publication and semantic review remain pending.
