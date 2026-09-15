# Fine-tuning C6/C7

## Role dans le projet

Ce dossier documente la partie fine-tuning associee aux configurations C6 et
C7. L'objectif est de preparer une extension LoRA du modele de base pour
l'extraction et l'evaluation de documents ConOps/OpsCon.

Dans l'etat actuel du projet, C6 et C7 sont traites comme des configurations
conditionnelles. Elles ne sont pas presentees comme des resultats valides tant
qu'un dataset expert suffisant, un split train/validation et un adaptateur LoRA
fonctionnel ne sont pas disponibles.

## Configurations concernees

| Code | Description |
| --- | --- |
| C6 | modele fine-tune sans RAG |
| C7 | modele fine-tune avec RAG |

C6 permet d'etudier l'effet du fine-tuning seul. C7 permet d'observer
l'association entre un modele adapte au domaine et un contexte documentaire
fourni par le RAG.

## Donnees d'entrainement

Le dataset principal est stocke dans :

```text
data/fine_tuning/expert_validated_training_data.jsonl
```

Le format attendu reste volontairement simple :

```json
{
  "instruction": "Extract and evaluate ConOps/OpsCon elements according to official rules V3.0.",
  "input": "document excerpt + relevant rules + RAG context",
  "output": "valid JSON extraction validated or corrected by expert"
}
```

Les fichiers derives sont :

```text
data/fine_tuning/train_dataset.jsonl
data/fine_tuning/validation_dataset.jsonl
```

La separation train/validation est importante pour verifier que le modele ne
se limite pas a memoriser les exemples d'entrainement.

## Adaptateur LoRA

L'adaptateur local est attendu dans :

```text
models/conops-lora/
```

Un adaptateur exploitable contient au minimum :

- `adapter_config.json` ;
- un fichier de poids `adapter_model.*`.

Le provider `FineTunedLLM` charge le modele de base avec `transformers`, puis
applique l'adaptateur LoRA avec `peft`. Lorsque l'adaptateur n'est pas present
ou que le dataset est insuffisant, la plateforme signale que C6/C7 sont
prepares mais non executes.

## Critere de credibilite

Avant de presenter C6 ou C7 comme resultats experimentaux, je retiens les
conditions suivantes :

- dataset JSONL non vide avec les champs `instruction`, `input` et `output` ;
- exemples corriges ou valides par expertise humaine ;
- split `train_dataset.jsonl` et `validation_dataset.jsonl` disponible ;
- au moins 20 exemples valides pour un premier essai controle ;
- adaptateur LoRA sauvegarde et rechargeable ;
- comparaison avec C1, C3, C4 et C5 sur les memes documents ;
- nombre de repetitions identique entre les configurations comparees ;
- traces exportees dans les CSV et les metadonnees de run.

## Interpretation des resultats

Le fine-tuning n'est pas interprete seul. Son interet se mesure par rapport aux
configurations de reference :

- C1 pour le comportement du LLM seul ;
- C3 pour l'effet des regles dans le prompt ;
- C4 pour l'effet du RAG ;
- C5 pour l'effet combine du RAG et de l'evaluateur.

Une amelioration credible doit apparaitre sur plusieurs dimensions :
conformite aux regles, stabilite des repetitions, grounding documentaire,
qualite du JSON et validation experte.

## Versionnement

Les datasets d'exemple peuvent rester dans le depot lorsqu'ils ne contiennent
pas de donnees sensibles. Les poids de modeles et adaptateurs LoRA restent
locaux par defaut. Leur versionnement demande un mecanisme adapte, par exemple
Git LFS ou un stockage externe documente.
