# House Prices – Advanced Regression Techniques

Ma première compétition Kaggle : prédire le prix de vente de maisons à Ames (Iowa) à partir de 79 caractéristiques.
La métrique est la RMSE sur le logarithme du prix.

**Meilleur résultat : RMSE de 0,1090 en validation croisée, 0,122 sur le classement public Kaggle**, avec une combinaison d'un Lasso et d'un LightGBM réglé.

## Résultats

| Soumission | RMSE en CV (5 folds) | Score Kaggle |
| --- | --- | --- |
| Lasso (alpha = 0,005, 66 colonnes gardées) | 0,1141 | 0,12756 |
| 60 % Lasso + 40 % LightGBM | 0,1109 | 0,12237 |
| 40 % Lasso + 60 % LightGBM réglé | **0,1090** | 0,12260 |

L'écart entre les deux dernières soumissions sur Kaggle (0,0002) n'est pas significatif : le classement public ne porte que sur environ la moitié du test. La validation croisée (1 458 maisons, 5 folds) sert de référence pour comparer les modèles.

## Démarche

### 1. Analyse exploratoire (`notebooks/01_eda.ipynb`)

- **Valeurs manquantes** : analyse colonne par colonne. Un NaN qui signifie « absent » (pas de garage, pas de sous-sol) devient une catégorie `None` ; un vrai manque est imputé ; les colonnes presque vides ou sans information sont supprimées (`PoolQC`, `MiscFeature`, `GarageYrBlt`, `Utilities`).
- **Valeurs extrêmes** : deux maisons retirées du train (Id 524 et 1299). Ce ne sont pas des erreurs de saisie mais des ventes anormales : très grandes maisons neuves, vendues bien en dessous de leur valeur (ventes partielles).
- **Asymétrie** : `log1p` sur 16 surfaces à longue traîne, pour rendre linéaire la relation multiplicative entre surface et prix.
- **Feature engineering** : surface totale (`TotalSF`), âges à la vente (`HouseAge`, `RemodAge`, qui remplacent les années), nombre de salles de bain pondéré (`TotalBath`) et indicatrices (`IsNew`, `IsRemodeled`, `Has2ndFloor`, `HasPool`). Chaque feature a été testée ; certaines ont été rejetées (`TotalPorchSF`, dont une composante a un effet opposé aux autres).

### 2. Préparation sans fuite de données (`src/`)

- `src/preprocessing.py` : les transformations déterministes (`clean`, `remove_outliers`, `add_features`), qui n'apprennent rien des données.
- `src/transformers.py` : des transformers scikit-learn sur mesure (`GroupMedianImputer`, `ModeImputer`, `Log1pTransformer`, `OrdinalMapper`).

Tout ce qui apprend des données est dans un **Pipeline scikit-learn de 21 étapes** : 17 imputers (médianes et modes), `log1p`, encodage ordinal de 17 colonnes, one-hot de 25 colonnes nominales, puis standardisation. Le Pipeline est réappris à chaque fold de validation croisée : aucune statistique n'est calculée sur les données d'évaluation.

### 3. Modélisation (`notebooks/02_modeling.ipynb`)

Tous les modèles sont évalués avec la même validation croisée (`KFold`, 5 folds, `random_state=42`), pour que les écarts de score viennent du modèle et non du découpage.

- **Lasso** : modèle de référence. `alpha` réglé sur une grille logarithmique : 0,005, avec 66 colonnes gardées sur 253 (la pénalité L1 élimine surtout des catégories rares du one-hot).
- **LightGBM** : réglé par étapes avec `GridSearchCV` (complexité des arbres, puis échantillonnage des colonnes et des maisons). RMSE de 0,1184 à 0,1118.
- **Ridge** : testé pour diversifier la combinaison, mais ses erreurs ressemblent trop à celles du Lasso (corrélation 0,93) : aucun gain, non retenu.
- **Combinaison** : moyenne pondérée des prédictions en log. Le poids est choisi sur les prédictions *out-of-fold* (`cross_val_predict`), pour ne jamais choisir sur des maisons que le modèle a vues. Le gain vient de la diversité : le Lasso et LightGBM ne se trompent pas sur les mêmes maisons.

## Ce que j'en retiens

- **La diversité compte plus que la force individuelle** : LightGBM, moins bon que le Lasso au départ, l'a amélioré dans la combinaison parce que ses erreurs étaient différentes. Ridge, trop semblable au Lasso, n'a rien apporté.
- **Un gain plus petit que l'écart-type entre folds n'est pas un gain** : pour choisir un paramètre, je compare fold par fold et je cherche une tendance régulière plutôt que la meilleure valeur d'un tableau.
- **Faire confiance à sa validation croisée** plutôt qu'au classement public, calculé sur trop peu de maisons pour départager des écarts de quelques millièmes.

## Structure du projet

```
kaggle-house-prices/
├── data/                  # données Kaggle (non versionnées)
├── notebooks/
│   ├── 01_eda.ipynb       # analyse exploratoire et décisions de préparation
│   └── 02_modeling.ipynb  # Pipeline, modèles, combinaison, soumissions
├── src/
│   ├── preprocessing.py   # transformations déterministes
│   └── transformers.py    # transformers scikit-learn sur mesure
├── submissions/           # fichiers de soumission (non versionnés)
└── requirements.txt
```

## Reproduire

1. Cloner le dépôt et créer un environnement virtuel :
   ```bash
   git clone https://github.com/raybelpro77git/kaggle-house-prices.git
   cd kaggle-house-prices
   python -m venv .venv
   .venv\Scripts\activate        # Windows ; sous macOS / Linux : source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Télécharger les données depuis la [page de la compétition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) et placer `train.csv`, `test.csv`, `sample_submission.csv` et `data_description.txt` dans `data/`.
3. Exécuter `notebooks/01_eda.ipynb`, puis `notebooks/02_modeling.ipynb`. Les fichiers de soumission sont écrits dans `submissions/`.

## Outils

Python, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, transformers sur mesure), LightGBM, matplotlib, seaborn.
