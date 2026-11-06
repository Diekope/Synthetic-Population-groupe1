<!-- id: TK-SOURCES -->
# Source navigation

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

All application paths below refer to `https://github.com/Diekope/Synthetic-Population-groupe1` at `6a858cb5e5a3fef119fd3153af8f1be917abbf2b`, branch `feature/agents-toolkit`.
Open them with `git show <implementation-revision>:<path>` in an application checkout; they are not
copied into this knowledge-only tree.

| Responsibility | Source paths |
|---|---|
| CSV constraint model | src/Instance.java |
| Current construction and optimization | src/GreedyMain.java, src/ColumnGreedy.java, src/Declone.java |
| Independent Java evaluation | src/Evaluation.java, src/CheckPopulation.java |
| Legacy individual construction and annealing | src/Main.java, src/GreedyBuilder.java, src/LocalSearch.java, src/Population.java |
| Identity and prompt generation | tools/build_agents.py, tools/agent_labels.fr.json |
| Provider transport and conversation | tools/chat_agent.py |
| Local GUI and survey jobs | tools/gui.py |
| Survey execution and aggregation | tools/survey.py |
| Profile fidelity heuristics | tools/eval_fidelity.py, tools/fidelity_probes.fr.json |
| Existing tests and CI | tests/, .github/workflows/tests.yml |
| Environment and application usage | pyproject.toml, README.md, README_AGENTS.md |

Implementation navigation for the new client belongs to its own branch and revision. Its package
contains `source.py` (Git snapshots/search), `service.py` (durable lifecycle), `hosts.py`
(existing project chat host instrumentation), and `__main__.py` (shared CLI).

New tooling implementation revision: `b6c040a7c22ee887a891ec93769cdeb2c9b9f755`, branch `codex/team-knowledge-service`.
