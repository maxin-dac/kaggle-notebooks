# 🚢 Titanic: Survival Dynamics & Predictive Modeling

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/titanic-analysis)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

## Objective
1. Perform deep Exploratory Data Analysis (EDA) to uncover the core historical drivers of survival (gender, class, family size, deck proximity) and translate them into passenger personas and risk profiles.
2. Build an ensemble predictive model using advanced Feature Engineering and Ensemble Learning to forecast test set survival.

## Key Empirical Insights
- **The Gender & Class Interaction**: The "women and children first" protocol was applied asymmetrically. 1st/2nd-class females had near-universal survival (>90%), while 3rd-class females dropped to 50%. Class overrode gender for poorer women.
- **The "Goldilocks" Family Effect**: A clear inverted-U relationship. Solo travelers (30.4% survival) and large families $\ge 5$ (22.3% survival) fared worst. Families of 2–4 achieved the highest survival rates (**61.83%**).
- **Deck Proximity**: Decks B–D (1st class, boat-deck adjacent) showed >75% survival, while Deck G (lowest, 3rd class) dropped to 28%.
- **Fare as a Continuous Proxy**: Survival rises monotonically with fare quintiles ($Q_1 = 21.8\% \rightarrow Q_5 = 64.2\%$).

## Model Architecture
An ensemble `VotingClassifier` combining:
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVC` (Support Vector Classifier with RBF kernel)
- Stratified cross-validation via `StratifiedKFold`.

## Kaggle Link
Direct link to the notebook: [titanic-analysis](https://www.kaggle.com/code/maximendacleu/titanic-analysis)
