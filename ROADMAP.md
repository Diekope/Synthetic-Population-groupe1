# Pistes d'amélioration

Liste des ajouts et améliorations possibles pour la couche agents / simulation
sociale, classés par thème. Rien ici n'est engagé : c'est un vivier de travail.
Chaque piste indique où intervenir et ce qui permettrait de la considérer comme
terminée.

## Priorités rapides

| Piste | Effort | Gain |
|---|---|---|
| Capture `docs/gui.png` à jour | S | doc fidèle |
| Export CSV des conversations | S | analyse externe |
| Retry/backoff sur les appels LLM | S | robustesse |
| Lint + typage (`ruff`, `mypy`) | M | dette technique |
| CI GitHub Actions | M | non-régression |
| Graphique d'évolution des opinions | M | lisibilité |
| Réseau d'affinités interactif | L | compréhension |

---

## 1. Moteur social (`tools/social_sim.py`)

### 1.1 Propagation des opinions par catégorie
Aujourd'hui une opinion existe par **sujet**. On pourrait ajouter une opinion par
**catégorie** (19 catégories dans `topics.fr.json`) qui sert de prior : un
échange sur un sujet déplace aussi légèrement l'opinion de sa catégorie.
- Où : `_opinion_spread`, `_update_opinions`, `state.json`.
- Fini quand : la catégorie est reprise dans `--resume` et visible dans la GUI.

### 1.2 Polarisation et mesure d'émergence
Mesurer ce que la simulation produit réellement : homophilie (les agents qui
discutent se ressemblent-ils ?), polarisation des opinions, taille des
communautés, vitesse de convergence des affinités.
- Où : nouveau module `tools/social_metrics.py` + commande `--metrics`.
- Fini quand : un run écrit un `metrics.json` et un test vérifie les bornes.

### 1.3 Mémoire longue et résumés
La mémoire par paire est une fenêtre de 5 sujets. Ajouter un résumé périodique
de la relation (« ils se sont disputés sur X, puis réconciliés ») pour éviter
les répétitions sur les runs longs.
- Où : `_memory_note`, `_remember_conversation`, `state.json`.
- Fini quand : un run de 200 ticks conserve une trace compressée par paire.

### 1.4 Réseau social explicite
Garder un graphe d'interactions (qui a parlé à qui, combien de fois) et s'en
servir pour choisir les partenaires (voisins du réseau + découverte aléatoire),
plutôt que la seule distance spatiale.
- Où : `select_partner`, nouvelles structures dans `Simulation`.
- Fini quand : le réseau est exporté et affichable dans la GUI.

### 1.5 Dynamique des groupes
Les groupes se forment mais ne se dissolvent pas. Ajouter :
- dissolution quand la cohésion tombe sous un seuil ;
- fusion de groupes proches ;
- rôles internes (leader, médiateur) dérivés des attributs.
- Où : `_update_groups`, `_group_cohesion`, `state.json`.

### 1.6 Refus plus expressifs
- Motifs typés et comptés (`occupé`, `désaccord`, `méfiance`), au lieu d'un
  texte libre ;
- réputation par agent : un agent souvent refusé devient moins sollicité ;
- excuser un refus (`--apology-rate`) pour tester la réconciliation.
- Où : `_refusal_reason`, `refusal_probability`, `stats`.

### 1.7 Événements externes
Injecter des événements datés (crise, annonce, météo) qui modifient les sujets
et les opinions pendant le run.
- Où : nouveau `--events FILE` (JSONL) consommé dans `step()`.
- Fini quand : un événement à un tick donné est reproductible à seed égal.

### 1.8 Diffusion de rumeur
Un agent qui apprend une information peut la transmettre aux conversations
suivantes avec une probabilité, et l'information se déforme.
- Où : nouveau champ `knowledge` par agent, propagé dans `_finish_conversation`.

---

## 2. Débit, coût et robustesse

### 2.1 Parallélisme avec limite de débit
`--parallel` lance N conversations simultanément sans contrôle de débit. Ajouter
`--max-inflight` et un backoff quand le provider renvoie 429.
- Où : `_generate_batch`, `chat_agent.respond`.

### 2.2 Retry et reprise sur erreur LLM
Réessayer les appels transitoires (timeout, 429, 5xx) avec backoff exponentiel,
puis marquer l'événement en erreur sans casser le run.
- Où : `chat_agent._open` / `respond`, `_generate_conversation`.
- Test : simuler un provider qui échoue deux fois puis réussit.

### 2.3 Cache disque des répliques
En mode mock ou pour les tests, mettre en cache les réponses par
(agent, historique, modèle) afin d'accélérer les longues suites.
- Où : `chat_agent.respond` ou un wrapper `cache`.

### 2.4 Reprise après arrêt brutal
`--resume` repart d'un `state.json` écrit tous les N ticks. Ajouter un journal
de reprise plus fin (dernier tick complet, événements non appliqués) pour ne
jamais rejouer ni perdre une conversation.
- Où : `run`, `state_every`, `_write_state`.

### 2.5 Streaming en parallèle
Aujourd'hui le flux est coupé quand `--parallel > 1`, car les tokens
s'entrelacent. On pourrait multiplexer les flux avec un préfixe par conversation.
- Où : `_generate_conversation`, `_generate_batch`.

---

## 3. GUI (`tools/gui.py`)

### 3.1 Capture `docs/gui.png` à jour
L'image date d'avant le panneau Simulation. La régénérer avec le panneau visible.
- Où : `docs/gui.png`, script Playwright ou capture manuelle.

### 3.2 Graphique d'évolution
Courbes par tick : conversations, refus, groupes, opinion moyenne, affinité
moyenne. Stocker l'historique dans `state.json` ou le recalculer depuis
`events.jsonl`.
- Où : nouveau panneau + endpoint `/api/sim/history`.

### 3.3 Réseau d'affinités interactif
Graphe force-directed des affinités fortes, avec groupes en couleur. Cliquer un
nœud ouvre la fiche de l'agent.
- Où : canvas dédié ou SVG, données déjà dans `/api/sim/status`.

### 3.4 Interactions dans l'Atlas
Cliquer une bille sélectionne l'agent ; survol affiche nom + affinités ; double
clic recentre. Aujourd'hui le survol sert surtout à l'affinité.
- Où : `drawAtlasFrame`, gestion des hit-tests existante (`ATLAS_HITS`).

### 3.5 Filtres d'événements et recherche
Filtrer le flux par type (conversation, refus, groupe), par agent ou par sujet ;
rechercher dans les transcripts.
- Où : `_sim_events`, panneau Journal.

### 3.6 Comparaison de runs
Sélectionner deux runs et comparer côte à côte : stats, opinions, groupes,
transcripts. Très utile pour régler les taux.
- Où : `/api/runs/compare?ids=...`.

### 3.7 Export CSV / Markdown
Exporter les conversations (participants, sujet, affinités, transcript) en CSV
ou Markdown, en plus du `state.json`.
- Où : `export_state`, `_write_transcript`.

### 3.8 Confort
- Mode sombre ;
- raccourcis clavier (espace = pause/reprise) ;
- indicateur de progression / ETA ;
- reprise automatique du dernier run au démarrage ;
- message explicite quand le provider est injoignable.

### 3.9 Tests E2E
Installer Playwright et couvrir : démarrage, pause, reprise, arrêt, chargement
d'un run, export. Aujourd'hui les tests GUI sont unitaires (HTML + endpoints).

---

## 4. Qualité et outillage

### 4.1 Lint et typage
Ajouter `ruff` (format + lint) et `mypy` sur `tools/` et `tests/`, avec une
configuration dans `pyproject.toml`.
- Fini quand : `uv run ruff check .` et `uv run mypy tools` passent.

### 4.2 CI GitHub Actions
Matrice Python (3.11/3.12), `uv sync`, tests, `node --check` sur le JS extrait de
la GUI, cache uv.
- Fini quand : la CI tourne sur chaque PR.

### 4.3 Couverture
Mesurer la couverture et viser un seuil sur `social_sim.py` et `gui.py`
(`coverage.py`).
- Attention : les parties JS ne sont pas couvertes par les tests Python.

### 4.4 Tests de charge
Un test lent (marqué `slow`) qui lance 500 agents sur 100 ticks et vérifie :
pas d'erreur, positions bornées, taille de `state.json` raisonnable.

### 4.5 Validation des données
Vérifier automatiquement que `agents.jsonl`, `topics.fr.json` et les schémas
JSON restent cohérents (ids uniques, catégories connues, pas de sujet vide).

---

## 5. Analyse et exploitation

### 5.1 Notebook d'analyse
Un notebook qui charge un run et produit : nombre de conversations par agent,
distribution des opinions, graphe d'affinités, sujets les plus discutés.

### 5.2 Détection de coalitions
Repérer les sous-groupes cohérents (affinité + opinions proches) et suivre leur
évolution dans le temps.

### 5.3 Qualité des conversations
Heuristiques sur les transcripts : répétitions, monologues, réponses hors sujet,
longueur moyenne. Servirait à comparer les modèles.

### 5.4 Benchmark de modèles
Étendre le benchmark existant aux conversations multi-agents : temps par tour,
coût, qualité perçue, taux d'échec.

---

## 6. Documentation et packaging

### 6.1 Guide de contribution
`CONTRIBUTING.md` : style, tests, conventions de commit, branches.

### 6.2 Schémas JSON
Documenter et valider `state.json`, `events.jsonl`, `config.json`,
`transcripts/*.json` avec des schémas versionnés.

### 6.3 Exemples prêts à l'emploi
Un dossier `examples/` avec des configurations de runs (petit groupe, foule,
débat polarisé) et les commandes associées.

### 6.4 Packaging
Exposer `social-sim` et `agents-gui` comme points d'entrée console dans
`pyproject.toml`, pour éviter les `uv run python tools/...`.

---

## 7. Idées plus exploratoires

- **Agents qui se souviennent des autres agents**, pas seulement des paires :
  réputation globale, confiance.
- **Sujets générés** par les agents à partir de leur vécu, au lieu d'un catalogue
  fixe.
- **Émotions** (humeur à court terme) modulant le ton et le refus.
- **Cycle jour/nuit** : activité et sujets différents selon l'heure simulée.
- **Migration spatiale** : les agents se déplacent vers les zones de forte
  affinité, formant des quartiers.
- **Contrefactuel** : rejouer un run en changeant un seul paramètre et comparer.
