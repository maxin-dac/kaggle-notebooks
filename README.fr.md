<div align="center">

# Portfolio Kaggle : Data Science & Machine Learning

[![Switch to English](https://img.shields.io/badge/Language-English-blue?style=flat&logo=googletranslate&logoColor=white)](README.md)
[![Profil Kaggle](https://img.shields.io/badge/Kaggle-maximendacleu-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/maximendacleu)
[![Version Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Licence : MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Projets](https://img.shields.io/badge/Projets-9%20Notebooks-success?style=flat)]()

<p align="center">
  Une sélection de pipelines de Machine Learning, d'architectures de Deep Learning, de prévisions de séries temporelles et d'analyses exploratoires orientées Business Intelligence, développés et validés sur <b>Kaggle</b>.
</p>

</div>

---

## Table des Matières

- [Vue d'ensemble & Accès Rapide](#matrice-du-portfolio)
- [Projets par Domaine](#projets-par-domaine)
  - [1. Vision par Ordinateur & Deep Learning](#1-vision-par-ordinateur--deep-learning)
    - [Détection des Points Clés Faciaux](#-détection-des-points-clés-faciaux-avec-deep-learning)
    - [Reconnaissance de Chiffres Manuscrits (MNIST CNN)](#-reconnaissance-de-chiffres-manuscrits---mnist-cnn)
  - [2. Machine Learning Tabulaire & Régression Avancée](#2-machine-learning-tabulaire--régression-avancée)
    - [Prédiction des Prix de l'Immobilier](#-prédiction-des-prix-immobiliers-house-prices)
    - [Titanic : Dynamique de Survie & Modélisation Prédictive](#-titanic--dynamiques-de-survie--modélisation-prédictive)
    - [Spaceship Titanic : Pipeline Baseline](#-spaceship-titanic---machine-learning)
  - [3. Séries Temporelles & Prévision](#3-séries-temporelles--prévision)
    - [Prévision des Ventes Journalières (Store Sales Favorita)](#-store-sales---prévision-de-séries-temporelles)
  - [4. Business Intelligence & Audit Exploratoire](#4-business-intelligence--audit-exploratoire)
    - [Audit Commercial & Rentabilité Superstore](#-analyse-des-ventes-superstore)
    - [Analyse de Performance Vidéo YouTube & Viralité](#-performance-de-contenu-youtube--eda-machine-learning--stratégie)
  - [5. NLP & Modèles de Langage (LLM)](#5-nlp--modèles-de-langage-llm)
    - [Classification & Préférences Humaines (Chatbot Arena)](#-classification-et-fine-tuning-llm-chatbot-arena)
- [Vue d'ensemble de la Stack Technique](#vue-densemble-de-la-stack-technique)
- [Structure du Dépôt](#structure-du-dépôt)
- [Installation & Démarrage Local](#installation--démarrage-local)
- [Auteur & Contact](#auteur--contact)

---

## Matrice du Portfolio

| Projet | Domaine | Stack Technique | Notebook Kaggle |
| :--- | :--- | :--- | :---: |
| **Facial Keypoints Detection** | Vision / Repères Faciaux | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/facial-keypoint-detection-with-deep-learning) |
| **House Prices Prediction** | Tabulaire / Régression Avancée | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/house-prices-prediction) |
| **Titanic: Survival Dynamics** | Classification / Ensemble ML | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/titanic-analysis) |
| **Superstore Sales Analysis** | BI / Audit Opérationnel | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/superstore-sales-analysis) |
| **YouTube Content Analytics** | Stratégie / Classification | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/youtube-video-analytics) |
| **Store Sales Forecasting** | Séries Temporelles / Demande | ![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/store-sales-time-series-forecasting) |
| **Spaceship Titanic ML** | Pipeline Classification Binaire | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/spaceship-titanic-machine-learning) |
| **Digit Recognizer (MNIST)** | Deep Learning / CNN | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/digit-recognizer) |
| **LLM Classification Arena** | NLP / Préférences Humaines | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) | [![Ouvrir sur Kaggle](https://img.shields.io/badge/Kaggle-Ouvrir%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/llm-classification-finetuning) |

---

## Projets par Domaine

### 1. Vision par Ordinateur & Deep Learning

#### 👁️ Détection des Points Clés Faciaux avec Deep Learning
> **Objectif** : Prédire précisément 15 points de repère faciaux (soit 30 coordonnées $(x, y)$) sur des images de visages en niveaux de gris de dimensions $96 \times 96$.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Points Forts** :
  - Conception de `Dataset` et `DataLoader` PyTorch personnalisés gérant les annotations partielles / valeurs manquantes.
  - Architecture de régression CNN profonde avec normalisation par lots (BatchNorm), régularisation par Dropout et fonction de coût MSE.
  - Normalisation/dénormalisation des coordonnées et projection graphique des repères prédits sur les visages de test.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/facial-keypoint-detection-with-deep-learning) | [Dossier du projet](computer-vision/01-facial-keypoints-detection/)

---

#### 🔢 Reconnaissance de Chiffres Manuscrits - MNIST CNN
> **Objectif** : Classifier avec une grande précision les chiffres manuscrits (de 0 à 9) à partir d'images en niveaux de gris à l'aide d'un réseau de neurones convolutif (CNN).

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Points Forts** :
  - Pipeline CNN 2D complet avec couches convolutives (`Conv2D`), sous-échantillonnage (`MaxPooling2D`), régularisation par `Dropout` et tête dense avec fonction d'activation `Softmax`.
  - Normalisation des valeurs de pixels et suivi rigoureux de la perte d'entropie croisée catégorielle.
  - Analyse approfondie des prédictions via matrice de confusion et rapport de classification détaillé.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/digit-recognizer) | [Dossier du projet](computer-vision/02-digit-recognizer-mnist/)

---

### 2. Machine Learning Tabulaire & Régression Avancée

#### 🏡 Prédiction des Prix Immobiliers (House Prices)
> **Objectif** : Estimer avec précision le prix de vente final (`SalePrice`) de biens résidentiels grâce à un pipeline de Machine Learning de bout en bout et un modèle de Gradient Boosting.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Top 10 des Variables les Plus Prédictives** (obtenues via `GradientBoostingRegressor`) :

| Rang | Variable | Description | Type d'Impact sur la Valeur |
| :---: | :--- | :--- | :--- |
| **1** | `OverallQual` | Qualité générale des matériaux et des finitions | **Variable Dominante** |
| **2** | `GrLivArea` | Surface habitable hors-sol en pieds carrés | Espace / Volume habitable |
| **3** | `TotalBsmtSF` | Surface totale du sous-sol en pieds carrés | Espace / Surface utile |
| **4** | `GarageCars` | Capacité du garage en nombre de voitures | Commodité / Stationnement |
| **5** | `1stFlrSF` | Surface habitable du premier étage | Espace / Disposition |
| **6** | `GarageArea` | Superficie totale du garage en pieds carrés | Commodité / Stockage |
| **7** | `YearBuilt` | Date de construction originale de la propriété | Âge / Modernité |
| **8** | `FullBath` | Nombre de salles de bain complètes hors-sol | Confort / Équipement |
| **9** | `YearRemodAdd` | Date de dernière rénovation majeure | Rénovation / Modernité |
| **10** | `TotRmsAbvGrd` | Nombre total de pièces hors-sol (hors salles de bain) | Espace / Organisation des pièces |

- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/house-prices-prediction) | [Dossier du projet](tabular-ml/01-house-prices-regression/)

---

#### 🚢 Titanic : Dynamiques de Survie & Modélisation Prédictive
> **Objectif** : Transformer les données historiques de la catastrophe en insights analytiques précis et construire un ensemble de modèles prédictifs haute performance.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Découvertes Analytiques Clés** :
  - **Interaction Sexe & Classe Sociale** : Le protocole « femmes et enfants d'abord » s'est appliqué de façon très inégale. Les passagères de 1ère et 2ème classe ont survécu à **>90%**, contre seulement **50%** en 3ème classe.
  - **L'effet Familial « Boucles d'or » (Goldilocks)** : Relation en U inversé. Les passagers isolés (survie de 30.4%) et les familles nombreuses $\ge 5$ (survie de 22.3%) ont été les plus touchés. Les foyers de 2 à 4 personnes atteignent le taux de survie record de **61.83%**.
  - **Proximité du Pont de Sauvetage (Deck)** : Les ponts B à D affichent un taux de survie supérieur à **75%**, tandis que le pont G s'effondre à **28%**.
  - **Les Quintiles de Tarif comme Proxy de Hiérarchie** : La survie progresse de façon strictement croissante avec le prix du billet ($Q_1 = 21.8\% \rightarrow Q_5 = 64.2\%$).
- **Modélisation** : Ensemble pondéré (`VotingClassifier`) combinant `RandomForest`, `GradientBoosting` et `SVC`, validé par validation croisée stratifiée (`StratifiedKFold`).
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/titanic-analysis) | [Dossier du projet](tabular-ml/02-titanic-survival-dynamics/)

---

#### 🚀 Spaceship Titanic - Machine Learning
> **Objectif** : Concevoir une analyse exploratoire et une baseline de Machine Learning robuste pour prédire si un passager a été transporté dans une dimension alternative lors de l'anomalie spatiale.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Points Forts** :
  - Pipeline modulaire scikit-learn (`ColumnTransformer`, `Pipeline`) assurant l'imputation par médiane, la standardisation des dépenses (`RoomService`, `Spa`, etc.) et l'encodage `OneHotEncoder`.
  - Entraînement d'une Forêt Aléatoire (`RandomForestClassifier`) avec métriques complètes d'évaluation.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/spaceship-titanic-machine-learning) | [Dossier du projet](tabular-ml/03-spaceship-titanic/)

---

### 3. Séries Temporelles & Prévision

#### 📈 Store Sales - Prévision de Séries Temporelles
> **Objectif** : Prédire les ventes quotidiennes par famille de produits pour 54 magasins de la chaîne Favorita en Équateur, en combinant historique, signaux de calendrier et variables macroéconomiques.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Variables et Mécanismes Clés** :
  - **Saisonnalité Hebdomadaire Dominante** : La variable de décalage `lag_7` (ventes du même jour la semaine précédente) et la moyenne mobile `rmean_7` constituent les moteurs prédictifs principaux.
  - **Signaux Exogènes** : Intégration du cours du pétrole brut (baromètre économique équatorien), des jours fériés locaux et nationaux, et des volumes de transactions en caisse.
  - **Stabilisation de la Variance** : Transformation logarithmique de la cible $\log(1 + y)$ (`log1p`) afin de réduire la forte asymétrie des ventes et optimiser l'erreur RMSLE.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/store-sales-time-series-forecasting) | [Dossier du projet](time-series/01-store-sales-forecasting/)

---

### 4. Business Intelligence & Audit Exploratoire

#### 🏬 Analyse des Ventes Superstore
> **Objectif** : Mener un audit financier et opérationnel approfondi sur un jeu de données retail mondial pour identifier les leviers de rentabilité, déceler les fuites de marge et optimiser la logistique.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Constats Stratégiques Majeurs** :
  - **Le Piège des Remises à 20%** : Les remises supérieures à 20% n'augmentent pas proportionnellement les volumes d'achat et détruisent la marge brute ; les remises supérieures à 50% sont systématiquement à perte.
  - **L'Hémorragie de la Catégorie Mobilier (Furniture)** : Forte contributrice au chiffre d'affaires global, cette catégorie détruit de la valeur nette (sous-catégories Tables et Bibliothèques en perte structurelle).
  - **Zones Géographiques Rouges** : Détection d'États à fort volume de vente mais à résultat net négatif, révélant des surcoûts d'expédition ou une politique de remises incontrôlée.
  - **Le Paradoxe de Novembre** : Mois réalisant le plus fort chiffre d'affaires de l'année, mais dégageant un bénéfice inférieur à septembre ou décembre en raison des promotions excessives du Black Friday.
  - **Coûts Logistiques** : Les modes de livraison accélérés (First Class / Same Day) réduisent drastiquement le taux de marge nette comparés à la livraison standard.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/superstore-sales-analysis) | [Dossier du projet](business-intelligence-eda/01-superstore-sales-analysis/)

---

#### 🎬 Performance de Contenu YouTube : EDA, Machine Learning & Stratégie
> **Objectif** : Comprendre la dynamique de performance des vidéos YouTube à partir de 537 vidéos, analyser l'engagement de l'audience et prédire la probabilité de viralité à l'aide du Machine Learning.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Enseignements Clés & Stratégie de Contenu** :
  - **La Durée est Déterminante** : Les contenus viraux durent en moyenne **3 min 35 s** (215 secondes), alors que les vidéos standards durent plus de **1 h 20 min** (4 802 secondes).
  - **L'Effet de Dilution de l'Engagement** : Les vidéos non-virales conservent un taux d'engagement moyen plus élevé (**2.93%**) que les vidéos virales (**1.76%**), car la diffusion de masse vers une audience passive dilue le ratio d'interactions.
  - **Créneau Optimal de Publication** : Pic d'impact maximal observé entre **11h00 et 17h00 UTC**, avec un sommet à **14h00 UTC** (synchronisation des heures d'activité Europe et Amérique du Nord).
  - **Seuil de Viralité Statistique** : Isolé empiriquement au-delà de **85.9 Millions de vues** (95e centile).
  - **Pipeline Prédictif** : Modélisation via `RandomForestClassifier` encapsulée dans une `Pipeline` scikit-learn avec `ColumnTransformer`.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/youtube-video-analytics) | [Dossier du projet](business-intelligence-eda/02-youtube-content-analytics/)

---

### 5. NLP & Modèles de Langage (LLM)

#### 🤖 Classification et Fine-tuning LLM (Chatbot Arena)
> **Objectif** : Réaliser une analyse exploratoire approfondie sur le jeu de données Kaggle LMSYS Chatbot Arena pour comprendre les dynamiques et facteurs influençant la préférence humaine entre deux réponses de modèles d'IA générative.

- **Stack** : ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Points Forts** :
  - Parsing et traitement de structures textuelles imbriquées au format JSON.
  - Analyse du biais de longueur (la verbosité influence-t-elle la probabilité de victoire d'un LLM ?).
  - Établissement d'une baseline prédictive de préférence par Régression Logistique.
- **Notebook** : [Voir sur Kaggle](https://www.kaggle.com/code/maximendacleu/llm-classification-finetuning) | [Dossier du projet](nlp-llm/01-llm-chatbot-arena-classification/)

---

## Vue d'ensemble de la Stack Technique

Tous les badges ci-dessous sont au style Shields.io **flat** :

| Domaine | Outils & Bibliothèques |
| :--- | :--- |
| **Langages & Environnement** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) |
| **Machine Learning & Séries Temporelles** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) |
| **Manipulation de Données** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) |
| **Visualisation de Données & BI** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) |
| **Plateformes** | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) |

---

## Structure du Dépôt

```plaintext
kaggle-notebooks/
├── README.md                                  # Documentation en anglais (par défaut)
├── README.fr.md                               # Documentation en français
├── requirements.txt                           # Dépendances Python globales
├── .gitignore                                 # Exclusion des artefacts et données lourdes
│
├── computer-vision/
│   ├── 01-facial-keypoints-detection/         # Régression de repères faciaux avec PyTorch
│   │   └── README.md
│   └── 02-digit-recognizer-mnist/             # Classification de chiffres manuscrits (CNN)
│       └── README.md
│
├── tabular-ml/
│   ├── 01-house-prices-regression/            # Régression avancée & Gradient Boosting
│   │   └── README.md
│   ├── 02-titanic-survival-dynamics/          # EDA survie & VotingClassifier Ensemble
│   │   └── README.md
│   └── 03-spaceship-titanic/                  # Baseline classification avec scikit-learn
│       └── README.md
│
├── time-series/
│   └── 01-store-sales-forecasting/            # Prévision de ventes LightGBM (Favorita)
│       └── README.md
│
├── business-intelligence-eda/
│   ├── 01-superstore-sales-analysis/          # Audit de rentabilité et logistique Superstore
│   │   └── README.md
│   └── 02-youtube-content-analytics/          # Analyse de viralité YouTube & Random Forest
│       └── README.md
│
└── nlp-llm/
    └── 01-llm-chatbot-arena-classification/   # Analyse des préférences humaines sur LLMs
        └── README.md
```

---

## Installation & Démarrage Local

### 1. Cloner le dépôt
```bash
git clone https://github.com/maximendacleu/kaggle-notebooks.git
cd kaggle-notebooks
```

### 2. Créer et activer un environnement virtuel
```bash
# Sur Linux / macOS :
python3 -m venv venv
source venv/bin/activate

# Sur Windows (PowerShell) :
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Démarrer Jupyter Lab
```bash
jupyter lab
```

---

## Auteur & Contact

**Maxime Ndacleu**  
Data Scientist & Machine Learning Practitioner

- **Kaggle** : [@maximendacleu](https://www.kaggle.com/maximendacleu)
- **GitHub** : [@maximendacleu](https://github.com/maximendacleu)

---

<div align="center">
  <sub>Si vous appréciez ces travaux ou les trouvez utiles, n'hésitez pas à laisser une étoile sur GitHub et un upvote sur Kaggle !</sub>
</div>
