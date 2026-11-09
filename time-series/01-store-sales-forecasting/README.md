# 📈 Store Sales - Time Series Forecasting

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/store-sales-time-series-forecasting)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-007ACC?style=flat&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

## Objective
Accurately predict daily unit sales for thousands of product families sold across 54 Favorita grocery stores in Ecuador, incorporating store metadata, promotion schedules, crude oil prices, holidays, and transaction traffic.

## Key Insights & Forecasting Dynamics
- **Weekly Seasonality is Dominant**: Feature importance analysis shows that `lag_7` (sales on the same weekday from the preceding week) and `rmean_7` (7-day rolling average) are the strongest predictors.
- **Recent Trends**: Multi-step rolling windows capture localized demand shifts much more effectively than isolated single-day lags.
- **Exogenous Signals**: Product category (`family`), day of year, daily store transactions, and national/local holidays provide crucial non-lag predictive power.
- **Variance Stabilization**: Log-transformation ($\log(1 + y)$) stabilizes high variance on volatile product categories and optimizes LightGBM against RMSLE.

## Kaggle Link
Direct link to the notebook: [store-sales-time-series-forecasting](https://www.kaggle.com/code/maximendacleu/store-sales-time-series-forecasting)
