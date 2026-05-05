# YouTube Spam Detection

Projet de classification de commentaires YouTube (`ham` vs `spam`) avec NLP + Machine Learning.

## Contenu du projet

- `YouTubeSpamDetection.ipynb`: notebook d'exploration initial.
- `spam_detector.py`: version scriptable et reproductible pour entrainer/evaluer/sauvegarder le modele.
- `youtube-dataset/*.csv`: donnees d'entrainement.

## Ameliorations apportees

- Pipeline unique `TF-IDF (1-2 grams) + ComplementNB`.
- Split stratifie et reproductible.
- Rapport de classification clair.
- Sauvegarde du modele entraine (`joblib`).
- Mode prediction en ligne de commande.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Entrainement

```bash
python spam_detector.py
```

Le modele sera sauvegarde dans `artifacts/spam_model.joblib`.

## Prediction rapide

```bash
python spam_detector.py --predict "This song is amazing!" "Win money now click here"
```

## Parametres utiles

```bash
python spam_detector.py --dataset-dir youtube-dataset --test-size 0.2 --random-state 365
```

## Prochaines pistes

- Ajouter une validation croisee et recherche d'hyperparametres.
- Ajouter des tests unitaires (`pytest`) pour fiabiliser le pipeline.
- Ajouter une petite API (FastAPI/Flask) pour servir le modele.

 
