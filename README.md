<div align="center">

# Kaggle Data Science & Machine Learning Portfolio

[![Switch to French](https://img.shields.io/badge/Langue-Français-0055A5?style=flat&logo=googletranslate&logoColor=white)](README.fr.md)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-maximendacleu-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/maximendacleu)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Projects](https://img.shields.io/badge/Projects-9%20Notebooks-success?style=flat)]()

<p align="center">
  A curated collection of production-grade Machine Learning pipelines, Deep Learning architectures, Time Series forecasting systems, and Business Intelligence exploratory analyses developed and benchmarked on <b>Kaggle</b>.
</p>

</div>

---

## Table of Contents

- [Overview & Quick Access](#quick-portfolio-matrix)
- [Projects by Domain](#projects-by-domain)
  - [1. Computer Vision & Deep Learning](#1-computer-vision--deep-learning)
    - [Facial Keypoints Detection](#-facial-keypoints-detection-with-deep-learning)
    - [Digit Recognizer (MNIST CNN)](#-digit-recognizer---mnist-cnn)
  - [2. Tabular ML & Advanced Regression](#2-tabular-ml--advanced-regression)
    - [House Prices Prediction](#-house-prices-prediction-advanced-regression)
    - [Titanic: Survival Dynamics & Predictive Modeling](#-titanic-survival-dynamics--predictive-modeling)
    - [Spaceship Titanic Baseline Pipeline](#-spaceship-titanic---machine-learning)
  - [3. Time Series Forecasting](#3-time-series-forecasting)
    - [Store Sales - Daily Demand Forecasting](#-store-sales---time-series-forecasting)
  - [4. Business Intelligence & Advanced EDA](#4-business-intelligence--advanced-eda)
    - [Superstore Sales & Profitability Audit](#-superstore-sales-analysis)
    - [YouTube Video Performance & Virality Analytics](#-youtube-content-performance-eda-machine-learning--strategy)
  - [5. NLP & Large Language Models](#5-nlp--large-language-models)
    - [LLM Classification & Human Preference Analysis](#-llm-classification-finetuning-chatbot-arena)
- [Tech Stack Badges](#tech-stack-overview)
- [Repository Structure](#repository-structure)
- [Getting Started Locally](#getting-started-locally)
- [Author & Contact](#author--connect)

---

## Quick Portfolio Matrix

| Project | Domain | Tech Stack | Kaggle Notebook |
| :--- | :--- | :--- | :---: |
| **Facial Keypoints Detection** | Computer Vision / Keypoints | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/facial-keypoint-detection-with-deep-learning) |
| **House Prices Prediction** | Tabular / Advanced Regression | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/house-prices-prediction) |
| **Titanic: Survival Dynamics** | Classification / Ensemble ML | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/titanic-analysis) |
| **Superstore Sales Analysis** | BI / Operational Audit | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/superstore-sales-analysis) |
| **YouTube Content Analytics** | Strategy / Classification | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/youtube-video-analytics) |
| **Store Sales Forecasting** | Time Series / Demand | ![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/store-sales-time-series-forecasting) |
| **Spaceship Titanic ML** | Binary Classification Pipeline | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/spaceship-titanic-machine-learning) |
| **Digit Recognizer (MNIST)** | Deep Learning / CNN | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/digit-recognizer) |
| **LLM Classification Arena** | NLP / Human Preferences | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) | [![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/llm-classification-finetuning) |

---

## Projects by Domain

### 1. Computer Vision & Deep Learning

#### 👁️ Facial Keypoints Detection with Deep Learning
> **Objective**: Accurately predict 15 facial landmarks (30 coordinates $(x, y)$) on $96 \times 96$ grayscale human face images.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Highlights**:
  - Custom PyTorch `Dataset` & `DataLoader` pipelines handling missing ground-truth keypoint annotations.
  - Deep Convolutional Neural Network regressor with batch normalization, dropout, and MSE loss optimization.
  - Coordinate re-scaling and visual validation of predicted landmark overlays.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/facial-keypoint-detection-with-deep-learning) | [Local Project Folder](computer-vision/01-facial-keypoints-detection/)

---

#### 🔢 Digit Recognizer - MNIST CNN
> **Objective**: Classify handwritten digits (0 to 9) from the classic MNIST dataset with high test accuracy using modern deep learning architectures.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Highlights**:
  - End-to-end 2D Convolutional Neural Network with Conv2D, MaxPooling2D, Dropout regularization, and Dense classification head.
  - Image normalization, data reshaping, and categorical cross-entropy loss tracking.
  - Detailed model evaluation with Confusion Matrix analysis and classification reports.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/digit-recognizer) | [Local Project Folder](computer-vision/02-digit-recognizer-mnist/)

---

### 2. Tabular ML & Advanced Regression

#### 🏡 House Prices Prediction (Advanced Regression)
> **Objective**: Forecast the final sales price (`SalePrice`) of residential properties in Ames, Iowa, using an end-to-end ML pipeline with advanced feature engineering and Gradient Boosting.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Top 10 Feature Importances** (from trained `GradientBoostingRegressor`):

| Rank | Feature | Description | Valuation Impact Type |
| :---: | :--- | :--- | :--- |
| **1** | `OverallQual` | Overall material and finish quality of the house | **Dominant Feature** |
| **2** | `GrLivArea` | Above grade (ground) living area square feet | Size / Living Space |
| **3** | `TotalBsmtSF` | Total square feet of basement area | Size / Usable Space |
| **4** | `GarageCars` | Size of garage in car capacity | Utility & Convenience |
| **5** | `1stFlrSF` | First floor square feet | Size / Living Space |
| **6** | `GarageArea` | Size of garage in square feet | Utility & Storage |
| **7** | `YearBuilt` | Original construction date | Age / Modernity |
| **8** | `FullBath` | Full bathrooms above grade | Utility & Comfort |
| **9** | `YearRemodAdd` | Remodel date (indicates recent upgrades) | Age / Modernity |
| **10** | `TotRmsAbvGrd` | Total rooms above grade (excluding bathrooms) | Size / Space Allocation |

- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/house-prices-prediction) | [Local Project Folder](tabular-ml/01-house-prices-regression/)

---

#### 🚢 Titanic: Survival Dynamics & Predictive Modeling
> **Objective**: Conduct in-depth EDA to discover historical survival drivers, translate findings into passenger personas, and build a top-tier ensemble model for survival prediction.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Key Empirical Insights**:
  - **Gender & Class Interaction**: Protocol "women and children first" was asymmetric. 1st/2nd-class females survived at **>90%**, whereas 3rd-class females dropped to **50%**. Socioeconomic status severely attenuated the gender advantage.
  - **The "Goldilocks" Family Effect**: Inverted-U survival relationship. Solo travelers (30.4%) and large families $\ge 5$ (22.3%) suffered high mortality. Nuclear families of 2–4 achieved **61.83% survival**.
  - **Deck Proximity**: Decks B–D (adjacent to the boat deck) showed **>75%** survival, while Deck G dropped to **28%**.
  - **Fare Quintiles as Continuous Hierarchy**: Monotonic survival progression ($Q_1 = 21.8\% \rightarrow Q_5 = 64.2\%$).
- **Modeling**: Weighted `VotingClassifier` combining `RandomForest`, `GradientBoosting`, and `SVC` tuned with `StratifiedKFold`.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/titanic-analysis) | [Local Project Folder](tabular-ml/02-titanic-survival-dynamics/)

---

#### 🚀 Spaceship Titanic - Machine Learning
> **Objective**: Build an exploratory data analysis and a production-grade machine learning baseline to predict passenger transport into an alternate dimension.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Highlights**:
  - Robust preprocessing via `ColumnTransformer` and `Pipeline`: median imputation, `StandardScaler` for numeric amenities (`RoomService`, `Spa`, `VRDeck`), and `OneHotEncoder` for categorical signals.
  - Baseline model using `RandomForestClassifier` with complete validation evaluation (Accuracy Score, Classification Report).
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/spaceship-titanic-machine-learning) | [Local Project Folder](tabular-ml/03-spaceship-titanic/)

---

### 3. Time Series Forecasting

#### 📈 Store Sales - Time Series Forecasting
> **Objective**: Predict daily unit sales across thousands of item families sold in 54 Favorita grocery stores throughout Ecuador, fusing seasonal patterns with macroeconomic indicators.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Key Forecasting Signals & Methodologies**:
  - **Dominant Weekly Seasonality**: Autoregressive evaluation reveals `lag_7` (same day of prior week) and `rmean_7` (7-day rolling mean) as highest-importance features.
  - **Exogenous Features**: Integrated crude oil prices (Ecuador economic proxy), regional holidays, calendar effects, and daily store transaction traffic.
  - **Variance Stabilization**: Target transformation via $\log(1 + y)$ (`log1p`) to dampen extreme variance in volatile retail categories and optimize RMSLE.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/store-sales-time-series-forecasting) | [Local Project Folder](time-series/01-store-sales-forecasting/)

---

### 4. Business Intelligence & Advanced EDA

#### 🏬 Superstore Sales Analysis
> **Objective**: Conduct a deep-dive commercial audit on multinational retail operations, examining sales volumes, profit margins, discounting strategies, and geographical cost drivers.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Key Business Findings**:
  - **The 20% Discount Trap**: Discounts $>20\%$ fail to drive proportional incremental volume; discounts $>50\%$ mathematically guarantee operational losses.
  - **The Furniture Bleed**: High gross revenue masked destructive bottom-line margins in subcategories such as Tables and Bookcases due to unconstrained discounting.
  - **Geographic Red Zones**: Identified high-revenue territories generating net financial deficits caused by regional logistics and localized discounting practices.
  - **The November Paradox**: Highest annual gross sales coincide with squeezed operating profit due to aggressive Black Friday discount slashing.
  - **Logistical Inefficiencies**: First Class & Same Day shipping disproportionately compress margin thresholds compared to Standard Class.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/superstore-sales-analysis) | [Local Project Folder](business-intelligence-eda/01-superstore-sales-analysis/)

---

#### 🎬 YouTube Content Performance: EDA, Machine Learning & Strategy
> **Objective**: Model video performance dynamics and build a predictive classification pipeline for content virality based on audience engagement metrics and metadata.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Key Insights & Strategic Guidelines**:
  - **Duration Sweet Spot**: Viral content averages **3:35** (215 seconds), whereas non-viral content averages **>1 hour 20 minutes** (4,802 seconds). Fast-paced structures correlate strongly with retention.
  - **The Engagement Dilution Effect**: Non-viral videos maintain higher mean engagement rates (**2.93%**) vs viral hits (**1.76%**), as broader passive audiences dilute relative feedback ratios.
  - **Optimal Publishing Window**: Highest velocity momentum is achieved between **11:00 and 17:00 UTC**, peaking at **14:00 UTC** (synchronizing US & European peak hours).
  - **Viral Threshold**: Statistically isolated at **85.9 Million views** (95th percentile).
  - **Predictive Pipeline**: `RandomForestClassifier` trained within an end-to-end `ColumnTransformer` pipeline.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/youtube-video-analytics) | [Local Project Folder](business-intelligence-eda/02-youtube-content-analytics/)

---

### 5. NLP & Large Language Models

#### 🤖 LLM Classification Finetuning (Chatbot Arena)
> **Objective**: Conduct rigorous Exploratory Data Analysis and statistical preference modeling on the Kaggle LLM Chatbot Arena dataset to decode factors influencing human preference between competing generative AI models.

- **Stack**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
- **Highlights**:
  - Systematic parsing of conversation trees and JSON conversational exchanges.
  - Evaluation of model verbosity bias (length vs winning probability) and sentiment correlation.
  - Logistic regression baseline modeling predicting preference labels from engineered textual features.
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/maximendacleu/llm-classification-finetuning) | [Local Project Folder](nlp-llm/01-llm-chatbot-arena-classification/)

---

## Tech Stack Overview

All badges are displayed in Shields.io **flat style**:

| Layer | Tools & Libraries |
| :--- | :--- |
| **Core Languages & Runtime** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) |
| **Deep Learning Frameworks** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) |
| **Machine Learning & Forecasting** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) |
| **Data Manipulation** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) |
| **Data Visualization & BI** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) |
| **Platforms & Deployment** | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) |

---

## Repository Structure

```plaintext
kaggle-notebooks/
├── README.md                                  # English Documentation (Default)
├── README.fr.md                               # French Documentation (Bilingue)
├── requirements.txt                           # Global Python dependencies
├── .gitignore                                 # Git ignore for DS & ML artifacts
│
├── computer-vision/
│   ├── 01-facial-keypoints-detection/         # PyTorch facial landmarks regression
│   │   └── README.md
│   └── 02-digit-recognizer-mnist/             # TensorFlow/Keras CNN digit recognition
│       └── README.md
│
├── tabular-ml/
│   ├── 01-house-prices-regression/            # Advanced Regression & Gradient Boosting
│   │   └── README.md
│   ├── 02-titanic-survival-dynamics/          # Survival EDA & VotingClassifier Ensemble
│   │   └── README.md
│   └── 03-spaceship-titanic/                  # Scikit-learn Pipeline baseline
│       └── README.md
│
├── time-series/
│   └── 01-store-sales-forecasting/            # LightGBM Favorita demand forecasting
│       └── README.md
│
├── business-intelligence-eda/
│   ├── 01-superstore-sales-analysis/          # Commercial & margin audit (Plotly/EDA)
│   │   └── README.md
│   └── 02-youtube-content-analytics/          # Virality threshold & Random Forest
│       └── README.md
│
└── nlp-llm/
    └── 01-llm-chatbot-arena-classification/   # Human preference analysis & NLP baseline
        └── README.md
```

---

## Getting Started Locally

### 1. Clone the repository
```bash
git clone https://github.com/maxin-dac/kaggle-notebooks.git
cd kaggle-notebooks
```

### 2. Create and activate a virtual environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter Lab / Notebook
```bash
jupyter lab
```

---

## Author & Connect

**Maxime Ndacleu**  
Data Science & Machine Learning Practitioner

- **Kaggle**: [@maximendacleu](https://www.kaggle.com/maximendacleu)
- **GitHub**: [@maxin-dac](https://github.com/maxin-dac)

---

<div align="center">
  <sub>If you find these notebooks helpful or educational, consider starring this repository and upvoting on Kaggle!</sub>
</div>
