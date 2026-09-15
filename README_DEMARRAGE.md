# Plateforme d'evaluation ConOps/OpsCon

## Presentation generale

Ce projet est une plateforme Python dediee a l'analyse de documents
ConOps/OpsCon. Elle combine extraction PDF, connaissances metier, RAG,
evaluation par regles, generation SysML v2, graphes de connaissances et
exports experimentaux.

L'objectif principal est d'obtenir une analyse structuree et tracable d'un
document operationnel, puis de comparer plusieurs strategies d'extraction et
d'evaluation sur une base experimentale commune.

La plateforme ne se limite pas a envoyer un PDF vers un modele de langage. Le
pipeline applique une chaine complete :

```text
PDF -> extraction texte/OCR -> RAG -> prompt -> LLM -> JSON
    -> reparation JSON -> post-traitement -> evaluation par regles
    -> SysML v2 -> graphes -> CSV -> validation humaine
```

## Perimetre fonctionnel

La plateforme couvre les fonctions suivantes :

- chargement d'un document ConOps/OpsCon au format PDF ;
- extraction du texte, avec OCR local lorsque le PDF est scanne ;
- indexation de documents de reference pour le RAG ;
- comparaison de strategies LLM, RAG et evaluation ;
- application des regles officielles ConOps/OpsCon V3.0 ;
- production d'un JSON structure selon le schema du projet ;
- generation d'un modele SysML v2 approximatif ;
- construction d'un graphe de connaissances ;
- export des resultats en CSV, JSON, Markdown et HTML ;
- preparation d'un protocole de fine-tuning conditionnel.

## Strategies et configurations

Le protocole experimental distingue sept configurations :

| Code | Strategie |
| --- | --- |
| C1 | LLM seul |
| C2 | LLM avec template ConOps |
| C3 | LLM avec regles ConOps/OpsCon |
| C4 | LLM avec RAG |
| C5 | LLM avec RAG et evaluateur |
| C6 | modele fine-tune, actif avec adaptateur LoRA valide |
| C7 | modele fine-tune avec RAG, actif avec adaptateur LoRA valide |

C6 et C7 sont volontairement conditionnels. Ils ne sont consideres comme
exploitables que lorsqu'un dataset expert, un decoupage train/validation et un
adaptateur LoRA valide sont disponibles.

## Architecture du projet

Le code est organise autour de modules specialises :

| Dossier | Role |
| --- | --- |
| `src/core/` | services techniques : PDF, OCR, RAG, embeddings, Chroma, schemas |
| `src/domain/` | regles metier ConOps/OpsCon et ontologie |
| `src/evaluation/` | scoring, metriques, explications et syntheses scientifiques |
| `src/graph/` | construction du graphe de connaissances |
| `src/llm/` | providers LLM, prompts, reparation et controle JSON |
| `src/pipeline/` | orchestration extraction, evaluation et exports |
| `src/postprocessing/` | nettoyage des exigences et normalisation finale |
| `src/rules/` | chargement du repository officiel de regles |
| `src/sysml/` | generation SysML v2 |
| `src/training/` | preparation dataset et fine-tuning LoRA |
| `finetuning/` | points d'entree pratiques pour les experimentations C6/C7 |
| `docs/uml/` | diagrammes PlantUML du projet |

Cette organisation separe les responsabilites : le coeur technique reste dans
`src/core`, les regles metier dans `src/domain`, les analyses scientifiques
dans `src/evaluation`, et l'orchestration dans `src/pipeline`.

## Regles metier ConOps/OpsCon V3.0

Le referentiel de regles est stocke dans :

```text
data/rules/conops_opscon_rules_v3.json
```

Il contient les familles D1/D2, C1-C14, O1-O14 et O6a. Le scoring suit la
logique suivante :

- `+W` pour une regle attendue presente ;
- `-alpha*W` pour une regle attendue absente ;
- `-beta*W` pour un element interdit present.

Le module `src/domain/rules_loader.py` controle la coherence du referentiel :
version, identifiants, poids et parametres de penalite. Le module
`src/domain/rule_based_evaluator.py` applique ensuite l'evaluation de maniere
deterministe.

Les regles ne servent pas seulement au score final. Elles participent aussi a
la classification ConOps/OpsCon, au guidage du RAG, a la construction du prompt
et a l'estimation de confiance.

## RAG et documents de reference

Le RAG utilise les documents places dans :

```text
data/reference_docs/
```

L'index persistant est gere avec Chroma dans `data/chroma_db/`. Les embeddings
sont calcules avec `sentence-transformers/all-MiniLM-L6-v2`. La collection
principale est nommee `conops_reference_docs`.

Le RAG conserve la tracabilite des chunks utilises : score, source, page,
extrait et regles associees. Ces informations sont exportees dans
`rag_chunks_used.csv`.

## OCR local

Certains PDF operationnels sont scannes ou contiennent peu de texte exploitable.
Dans ce cas, la plateforme utilise RapidOCR et PyMuPDF. Les pages analysees
sont selectionnees sur le debut, le milieu et la fin du document afin de
limiter le temps de calcul.

Les resultats OCR sont conserves localement dans `data/ocr_cache/`. Ce cache
accelere les analyses suivantes et reste exclu du versionnement.

## Pipeline d'analyse

Le pipeline principal suit une logique progressive :

1. extraction du texte du PDF ;
2. classification ConOps/OpsCon/Hybrid/Neither ;
3. recuperation des chunks RAG utiles ;
4. construction du prompt avec les regles et les preuves ;
5. appel du fournisseur LLM selectionne ;
6. reparation du JSON lorsque la reponse est incomplete ;
7. post-traitement des entites extraites ;
8. evaluation deterministe par regles ;
9. generation SysML v2, graphes et rapports ;
10. export des fichiers experimentaux.

Cette separation permet d'analyser les erreurs a chaque etape : extraction,
RAG, generation, evaluation et export.

## Sorties generees

Chaque execution est stockee dans un dossier `outputs/<run_id>/`. Les fichiers
principaux sont :

| Fichier | Contenu |
| --- | --- |
| `experiments_summary.csv` | scores et statut par configuration |
| `evaluation_criteria.csv` | criteres, poids et scores |
| `extraction_items_detailed.csv` | elements extraits avec preuves |
| `rag_chunks_used.csv` | chunks RAG selectionnes |
| `rule_assessment_detailed.csv` | evaluation regle par regle |
| `graph_edges.csv` | relations du graphe avec preuves |
| `classification_explanation.json` | justification de la classification |
| `expert_validation.csv` | support de validation humaine |
| `run_metadata.json` | metadonnees d'execution |
| `model.sysml` | sortie SysML v2 |
| `graph_2d.html` | graphe interactif 2D |
| `graph_3d.html` | graphe interactif 3D |
| `report.md` | rapport lisible |
| `report.json` | rapport structure |

Les sorties sont regenerables et ne sont pas destinees a etre versionnees.

## Evaluation scientifique

L'evaluation combine plusieurs dimensions :

- conformite aux regles officielles V3.0 ;
- grounding documentaire des elements extraits ;
- couverture des preuves et des pages sources ;
- stabilite sur plusieurs repetitions ;
- comparaison avec le baseline C1 ;
- validation experte optionnelle.

Les metriques precision, recall et F1 demandent un gold standard annote
document par document. Le projet conserve donc une distinction claire entre
score automatique, preuve documentaire et jugement expert.

## SysML v2 et graphe de connaissances

La generation SysML v2 represente les elements extraits sous forme de package,
acteurs, exigences, interfaces, risques et relations operationnelles. Cette
generation reste une approximation exploitable pour l'analyse, pas une
validation formelle par parseur SysML officiel.

Le graphe de connaissances met en relation le document, les stakeholders, les
besoins, les exigences, les services, les capacites, les interfaces, les
risques, les actions futures et les preuves. Les noeuds et les relations
portent des attributs de confiance, de page source et de statut de validation.

## Fine-tuning

Le fine-tuning est prepare pour les configurations C6 et C7. Le dataset est
base sur des sorties corrigees et validees. Il est separe en train et
validation afin d'eviter une evaluation limitee a la memorisation des exemples.

Les fichiers principaux sont :

```text
data/fine_tuning/expert_validated_training_data.jsonl
data/fine_tuning/train_dataset.jsonl
data/fine_tuning/validation_dataset.jsonl
```

Un adaptateur LoRA local peut etre place dans `models/conops-lora/`. Les poids
de modele ne sont pas pousses dans GitLab, sauf mecanisme specifique comme Git
LFS ou depot externe.

## Documentation UML

Les diagrammes PlantUML documentent les vues principales du projet :

| Diagramme | Contenu |
| --- | --- |
| `01_component_architecture.puml` | architecture globale |
| `02_sequence_analysis_c5.puml` | sequence C5 avec RAG, JSON repair et evaluation |
| `03_class_architecture.puml` | classes logiques principales |
| `04_activity_experimental_pipeline.puml` | activite experimentale |
| `05_knowledge_model_conops_opscon.puml` | modele de connaissance |
| `06_deployment_architecture.puml` | deploiement local |
| `07_finetuning_pipeline.puml` | pipeline fine-tuning conditionnel |

Les sources restent dans `docs/uml/` et les rendus PNG dans
`docs/uml/rendered/`.

## Donnees locales et versionnement

Le depot garde uniquement les fichiers utiles a la comprehension, a
l'execution et a la reproductibilite du projet. Les elements generes localement
restent ignores :

- `.env` ;
- `.venv/` ;
- `__pycache__/` ;
- `.pytest_cache/` ;
- `outputs/` ;
- `global_results/` ;
- `data/ocr_cache/` ;
- `data/chroma_db/` ;
- poids de modeles dans `models/`.

Cette regle evite de melanger le code source avec des resultats temporaires,
des caches ou des fichiers propres a une machine.

## Limites actuelles

Le referentiel V3.0 permet une evaluation metier structuree, mais la validation
experte reste necessaire pour confirmer les resultats document par document.

Le RAG ameliore la contextualisation, mais la qualite depend du corpus de
reference, des chunks selectionnes et de la qualite du PDF source.

Le fine-tuning reste conditionnel tant que le volume de donnees expertes n'est
pas suffisant pour produire une comparaison solide avec C1, C3, C4 et C5.

La sortie SysML v2 est utile pour representer les resultats, mais elle ne
remplace pas une validation formelle dans un outil MBSE specialise.

## Etat du projet

La version actuelle met l'accent sur la robustesse du pipeline, la tracabilite
des preuves et la comparaison experimentale. Les priorites restantes sont
l'enrichissement du corpus de validation, la consolidation du fine-tuning et
l'analyse sur plusieurs documents ConOps/OpsCon.
