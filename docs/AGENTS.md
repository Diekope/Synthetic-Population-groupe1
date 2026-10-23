# Agents identifiables et interrogeables

Branche `feature/agents`, worktree externe `Synthetic-Population-agents`.
Elle ne modifie **ni** le générateur Java, **ni** les tables de marges.

## Pourquoi ajouter des attributs

`population.csv` contient 500 individus mais seulement **302 personas distinctes** :
les 45 tables imposent au moins 169 copies en trop. Une identité unique ne peut donc
pas être construite avec les seuls `A0..A11`.

On ajoute deux attributs **hors tables** — `prenom`, `nom` — tirés sans remise :

- aucune marge ne change (les tables ne les couvrent pas) ;
- les 500 vecteurs d'attributs deviennent distincts ;
- `identity_key` est un SHA-256 tronqué de **tous** les attributs, donc l'identité
  est dérivée des attributs, pas d'un compteur arbitraire.

## Utilisation

```bash
# 1. construire les agents (déterministe, seed 0)
uv run python tools/build_agents.py constraint_cells.csv population.csv

# 2. vérifier
uv run python -m unittest discover -s tests -v

# 3. lister / choisir
uv run python tools/chat_agent.py --list
uv run python tools/chat_agent.py --agent 42 --message "Tu fais quoi dans la vie ?"
uv run python tools/chat_agent.py --agent 42          # conversation interactive
```

La conversation garde l'historique, `/reset` l'efface, `/agent <id|clé|nom>` change
d'agent, `/attrs` réaffiche le profil.

## Fournisseurs LLM

| `--provider` | Prérequis | Variables |
|---|---|---|
| `mock` (défaut) | aucun, hors ligne | — |
| `openai` | clé API | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| `ollama` | `ollama serve` | `OLLAMA_HOST`, `OLLAMA_MODEL` |

```bash
OPENAI_API_KEY=sk-... uv run python tools/chat_agent.py --agent 42 --provider openai
uv run python tools/chat_agent.py --agent 42 --provider ollama --model llama3.1
```

`mock` ne simule pas une conversation : il sert à tester le câblage du pipeline.

## Fichiers produits (`agents/`)

| Fichier | Contenu |
|---|---|
| `agents.csv` | `id, identity_key, prenom, nom, A0..A11` |
| `agents.jsonl` | un agent par ligne : attributs + `persona` (prompt système) |
| `manifest.json` | contrôles : unicité, erreur de marges, seed |

## Libellés lisibles

`A0..A11` n'ont pas de sémantique connue. Pour un prompt parlant, fournis un JSON :

```json
{ "A0": "sexe", "A0=v1": "femme", "A0=v2": "homme" }
```

```bash
uv run python tools/build_agents.py constraint_cells.csv population.csv \
    --labels tools/agent_labels.example.json --seed 7
```

Les clés absentes restent affichées telles quelles (`A1=v0`).

## Garanties

- Erreur L1 sur les 45 tables = **0** après enrichissement (testé).
- 500 `identity_key` distinctes et 500 vecteurs d'attributs distincts (testé).
- Le CSV enrichi reste lisible par `CheckPopulation` : les colonnes en trop sont ignorées.

## Limites

- `prenom`/`nom` sont des métadonnées d'identité, pas des variables contraintes :
  ne les utilise pas comme signal démographique.
- Le prompt système est volontairement générique tant que les `A*` ne sont pas nommés.
- Les personas restent des échantillons synthétiques : ce sont des agents de test,
  pas des personnes réelles.
