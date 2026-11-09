# 👁️ Facial Keypoints Detection with Deep Learning

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/facial-keypoint-detection-with-deep-learning)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

## Objective

Predict 15 facial landmarks (30 coordinates $(x, y)$) on $96 \times 96$ grayscale face images from the Kaggle Facial Keypoints Detection competition.

## Methodology & Architecture

- **Data Handling**: Custom PyTorch `Dataset` with target normalization and missing coordinate masking/imputation.
- **Model**: Deep Convolutional Neural Network (CNN) featuring Conv2D layers, Batch Normalization, ReLU activations, Dropout regularization, and Fully Connected regression heads.
- **Loss Function**: Mean Squared Error (MSE Loss) with Adam optimizer.
- **Inference**: Rescaling coordinate outputs back to pixel space and overlaying keypoints on unseen test samples.

## Kaggle link

Direct link to the notebook: [facial-keypoint-detection-with-deep-learning](https://www.kaggle.com/code/maximendacleu/facial-keypoint-detection-with-deep-learning)
