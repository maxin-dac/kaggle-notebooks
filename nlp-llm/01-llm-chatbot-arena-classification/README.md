# 🤖 LLM Classification & Human Preference Modeling (Chatbot Arena)

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/llm-classification-finetuning)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

## 🎯 Objective
Perform an in-depth Exploratory Data Analysis (EDA) on the Kaggle LMSYS LLM Chatbot Arena dataset to analyze conversational structures and isolate key determinants influencing human preferences between two competing Large Language Model responses.

## 🧠 Approach & Key Elements
- **Conversation Parsing**: Systematic deserialization and extraction of multi-turn conversation payloads from structured JSON entries.
- **Bias Diagnostics**: Investigating common LLM evaluation artifacts, notably the length/verbosity bias (where human raters lean toward longer, structured answers) and conversational tone.
- **Predictive Modeling**: Establishing a text-feature classification baseline with `LogisticRegression` and scikit-learn feature extractors.

## 🔗 Kaggle Link
Direct link to the notebook: [llm-classification-finetuning](https://www.kaggle.com/code/maximendacleu/llm-classification-finetuning)
