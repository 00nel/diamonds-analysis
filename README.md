# Diamond Price Prediction

Projet de Data Science / Machine Learning ayant pour objectif de prédire le prix d'un diamant à partir de ses caractéristiques physiques et de ses critères de qualité.

## Objectif

L'objectif de ce projet est de construire un modèle capable de prédire le prix d'un diamant à partir de différentes caractéristiques :

* `carat` : poids du diamant en carats
* `cut` : qualité de la taille
* `color` : couleur du diamant
* `clarity` : pureté du diamant
* `depth` : profondeur du diamant
* `table` : largeur de la table du diamant

Le projet suit une démarche complète de Machine Learning, allant de l'analyse exploratoire des données jusqu'à l'optimisation et l'évaluation du modèle final.

---

## Dataset

Le projet utilise le dataset **Diamonds**, contenant des informations sur plusieurs dizaines de milliers de diamants.

Les principales variables utilisées sont :

| Variable  | Description                      |
| --------- | -------------------------------- |
| `carat`   | Poids du diamant                 |
| `cut`     | Qualité de la taille             |
| `color`   | Couleur du diamant               |
| `clarity` | Pureté du diamant                |
| `depth`   | Profondeur du diamant            |
| `table`   | Largeur de la table              |
| `price`   | Prix du diamant — variable cible |

Certaines variables présentes dans le dataset original n'ont pas été conservées lorsqu'elles n'étaient pas pertinentes pour la modélisation.

---

## Analyse exploratoire (EDA)

Une analyse exploratoire a été réalisée afin de comprendre la structure du dataset et les relations entre les différentes variables.

Elle comprend notamment :

* analyse des distributions des variables ;
* détection et analyse des valeurs atypiques ;
* étude des variables catégorielles (`cut`, `color`, `clarity`) ;
* analyse de la relation entre le `carat` et le `price` ;
* analyse du prix selon différentes catégories de carat ;
* étude des diamants les plus chers ;
* analyse métier des caractéristiques associées aux prix élevés.

L'analyse des valeurs atypiques a notamment montré que les diamants présentant des valeurs élevées de `carat`, `x`, `y` et `z` correspondent généralement à des diamants de grande taille et ne constituent donc pas nécessairement des erreurs de données.

---

## Préprocessing

Avant l'entraînement des modèles, les données ont été séparées en variables explicatives (`X`) et variable cible (`y`).

Le dataset a ensuite été séparé en :

* jeu d'entraînement ;
* jeu de test.

Les variables numériques ont été standardisées avec `StandardScaler`, tandis que les variables catégorielles ont été encodées avec `OneHotEncoder`.

Le préprocessing a été intégré dans un `ColumnTransformer`, lui-même intégré dans un `Pipeline` avec chaque modèle.

Cette approche permet notamment d'appliquer exactement les mêmes transformations aux données d'entraînement et aux données de test.

---

## Baseline

Avant de tester différents algorithmes, une baseline simple a été mise en place.

La prédiction utilisée correspond à une valeur constante basée sur les données d'entraînement. Cette baseline sert de point de comparaison minimal afin de vérifier que les modèles de Machine Learning apportent réellement une amélioration.

Résultats de la baseline :

* **MAE : 2 817,07 $**
* **RMSE : 4 293,84 $**
* **R² : -0,15**

Les modèles testés par la suite améliorent donc fortement cette référence.

---

## Modèles testés

Plusieurs modèles de régression ont été comparés :

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**

Les modèles ont été évalués avec trois métriques :

* **MAE (Mean Absolute Error)** : erreur absolue moyenne en dollars ;
* **RMSE (Root Mean Squared Error)** : pénalise davantage les grandes erreurs ;
* **R²** : proportion de la variance du prix expliquée par le modèle.

---

## Résultats

| Modèle                     |    MAE ($) |   RMSE ($) |       R² |
| -------------------------- | ---------: | ---------: | -------: |
| Linear Regression          |     809,24 |   1 169,17 |     0,92 |
| Decision Tree              |     350,76 |     707,15 |     0,97 |
| Random Forest              |     281,08 |     538,27 |     0,98 |
| Gradient Boosting          |     405,44 |     730,56 |     0,97 |
| **Random Forest optimisé** | **278,28** | **530,11** | **0,98** |
| Gradient Boosting optimisé |     290,44 |     577,27 |     0,98 |

Le **Random Forest optimisé** obtient les meilleures performances parmi les modèles testés.

---

## Optimisation avec Optuna

Les modèles **Random Forest** et **Gradient Boosting** ont ensuite été optimisés à l'aide d'**Optuna** afin de rechercher de meilleures configurations d'hyperparamètres.

Pour le Random Forest, plusieurs paramètres ont notamment été explorés :

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

Pour le Gradient Boosting :

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `subsample`

L'optimisation a permis d'améliorer légèrement les performances du Random Forest et de manière plus importante celles du Gradient Boosting.

Cependant, le tuning du Random Forest s'est révélé coûteux en temps de calcul, notamment en raison du nombre d'arbres et de la validation croisée utilisée.

---

## Modèle final

Le modèle retenu est le **Random Forest optimisé**.

### Performances

* **MAE : 278,28 $**
* **RMSE : 530,11 $**
* **R² : 0,98**

Le modèle explique ainsi une grande partie de la variabilité des prix observés.

L'analyse des erreurs montre que l'erreur absolue augmente avec le `carat`. Cependant, lorsque l'on rapporte cette erreur au prix réel, l'erreur relative reste globalement comprise entre **6 et 10 %**, ce qui indique que l'augmentation de l'erreur en dollars est en grande partie liée au prix plus élevé des diamants de grande taille.

Le graphique des prix réels et prédits montre également une forte proximité entre les prédictions et les valeurs réelles.

---

## Structure du projet

```text
diamonds-analysis/
│
├── data/
│   └── diamonds.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── modelisation.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies utilisées

### Langage

* Python

### Analyse et visualisation

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Optuna

### Environnement

* Jupyter Notebook
* Conda
* Git / GitHub

---

## Installation

Cloner le repository :

```bash
git clone https://github.com/00nel/diamonds-analysis.git
cd diamonds-analysis
```

Créer l'environnement Conda :

```bash
conda create -n DIAMONDS python=3.14
```

Activer l'environnement :

```bash
conda activate DIAMONDS
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer Jupyter Notebook :

```bash
jupyter notebook
```

---

## Démarche du projet

Le projet suit globalement les étapes suivantes :

```text
Chargement des données
        ↓
Nettoyage
        ↓
Analyse exploratoire (EDA)
        ↓
Analyse métier
        ↓
Baseline
        ↓
Preprocessing
        ↓
Entraînement des modèles
        ↓
Comparaison des performances
        ↓
Optimisation avec Optuna
        ↓
Évaluation finale
        ↓
Analyse des erreurs
        ↓
Conclusion
```

---

## Conclusion

Ce projet a permis de mettre en pratique les principales étapes d'un projet de Machine Learning supervisé : exploration des données, preprocessing, création d'une baseline, entraînement de plusieurs modèles, comparaison des performances, optimisation des hyperparamètres et analyse des erreurs.

Les résultats montrent que les modèles capables de capturer des **relations non linéaires et des interactions entre variables**, comme les arbres de décision et les méthodes d'ensemble, sont particulièrement adaptés à cette problématique.

Le **Random Forest optimisé** constitue finalement le meilleur modèle parmi ceux testés, avec un **MAE de 278,28 $ et un R² de 0,98**.
