# Structure du projet

Ce projet est organise pour separer le code applicatif, les donnees metier,
les sorties generees et la documentation. L'objectif est de garder un depot
lisible, facile a relancer et propre pour un depot GitLab.

## Arborescence principale

```text
conops_ai_platform_v2/
|-- app.py                    # Interface Streamlit
|-- src/                      # Code source Python
|-- tests/                    # Tests automatises
|-- data/                     # Regles, exemples et documents de reference
|-- docs/                     # Documentation et diagrammes UML
|-- finetuning/               # Points d'entree pratiques pour C6/C7
|-- lib/                      # Bibliotheques front-end locales
|-- models/                   # Registre et modeles locaux ignores par Git
|-- outputs/                  # Resultats generes localement
|-- requirements*.txt         # Dependances par usage
`-- README_DEMARRAGE.md       # Guide de demarrage
```

## Organisation de `src/`

```text
src/
|-- config.py                 # Configuration centralisee
|-- core/                     # PDF, RAG, embeddings, schemas, vector store
|-- domain/                   # Regles ConOps/OpsCon et ontologie
|-- evaluation/               # Metriques, scoring et syntheses scientifiques
|-- experiments/              # Analyse de repetitions experimentales
|-- extraction/               # Extraction specialisee d'interfaces
|-- graph/                    # API stable pour les graphes
|-- knowledge/                # Contexte de connaissance metier
|-- llm/                      # Providers, prompts et reparation JSON
|-- pipeline/                 # Orchestration extraction, evaluation, exports
|-- postprocessing/           # Nettoyage des exigences
|-- rules/                    # Chargement du repository officiel de regles
|-- sysml/                    # Generation SysML
|-- training/                 # Dataset et fine-tuning LoRA
`-- utils/                    # Fonctions utilitaires partagees
```

## Donnees versionnees

Les fichiers stables utiles au projet restent dans `data/` :

- `data/rules/` contient les regles officielles ConOps/OpsCon V3.0.
- `data/reference_docs/` contient les documents de reference RAG.
- `data/gold_standards/` contient les jeux de validation disponibles.
- `data/examples/` et `data/fine_tuning/` contiennent les exemples de depart.

## Donnees locales ignorees

Les elements suivants sont regenerables ou propres a une machine locale :

- `.venv/`
- `.env`
- `__pycache__/`
- `.pytest_cache/`
- `outputs/`
- `global_results/`
- `data/ocr_cache/`
- `data/chroma_db/`
- les poids de modeles dans `models/`

## Commandes de verification

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```
