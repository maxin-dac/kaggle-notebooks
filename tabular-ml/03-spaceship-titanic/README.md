# 🚀 Spaceship Titanic - Machine Learning Pipeline

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/spaceship-titanic-machine-learning)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

## Objective
Predict whether a passenger was transported to an alternate dimension during the spaceship anomaly in the Kaggle Spaceship Titanic competition.

## Pipeline & Modeling
- **Data Preprocessing**:
  - `ColumnTransformer` bundling numerical and categorical branches.
  - Imputation via `SimpleImputer` (median for numerical amenities, most frequent for categories).
  - Feature scaling with `StandardScaler` for spending metrics (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`).
  - `OneHotEncoder(handle_unknown='ignore')` for categorical fields.
- **Model**: `RandomForestClassifier` with cross-validated performance tuning.
- **Evaluation**: Accuracy score, classification report, and ROC/PR curve inspection.

## Kaggle Link
Direct link to the notebook: [spaceship-titanic-machine-learning](https://www.kaggle.com/code/maximendacleu/spaceship-titanic-machine-learning)
