# 🔢 Digit Recognizer - MNIST CNN

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/maximendacleu/digit-recognizer)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Objective

Classify handwritten digits (between 0 and 9) from $28 \times 28$ grayscale images with high generalization accuracy on the Kaggle Digit Recognizer competition.

## Methodology & Architecture

- **Preprocessing**: Grayscale pixel normalization ($[0, 255] \rightarrow [0.0, 1.0]$), reshaping to $(28, 28, 1)$, and one-hot encoding of target class labels.
- **CNN Architecture**:
  - Multiple `Conv2D` feature extraction layers with ReLU activations.
  - `MaxPooling2D` for spatial downsampling.
  - `Dropout` regularization to prevent overfitting on specific handwriting styles.
  - Dense classification layers terminating in a 10-way `Softmax` output.
- **Evaluation**: Confusion matrix visualization, precision/recall per digit, and classification report.

## Kaggle link

Direct link to the notebook: [digit-recognizer](https://www.kaggle.com/code/maximendacleu/digit-recognizer)
