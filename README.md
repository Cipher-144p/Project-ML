# Fake News Detection using Machine Learning and Deep Learning

## Overview

This project implements a Fake News Detection system using multiple machine learning and deep learning models, including:
- **Bidirectional LSTM**
- **Random Forest Classifier**
- **Gaussian Naive Bayes**
- **K-Nearest Neighbors (KNN)**

The system supports multilingual data (Hindi and English), with appropriate preprocessing for each language, leveraging libraries like `DistilBERT` for feature extraction.

## Features
- **Bidirectional LSTM** for deep learning-based classification.
- Traditional ML models: Random Forest, Gaussian Naive Bayes, and KNN.
- Multilingual support: Detects the language (English or Hindi) and applies preprocessing accordingly.
- Handles class imbalance using `SMOTE`.
- Pretrained `DistilBERT` embeddings for feature representation.
- Robust evaluation with metrics like **Accuracy**, **Precision**, **Recall**, and **AUC-ROC**.

---

## Dataset

### File: `randomized_combined_dataset.csv`
- **Columns**:
  - `title`: The headline or text of the news article.
  - `label`: The classification label (1 for Fake News, 0 for Real News).

### Dataset Statistics:
- Contains both English and Hindi text.
- Imbalanced dataset managed using **SMOTE**.

---

## Installation

### Requirements
- Python 3.10+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tensorflow`
  - `transformers`
  - `scikit-learn`
  - `seaborn`
  - `langdetect`
  - `snowballstemmer`
  - `imbalanced-learn`

### Install Dependencies
```bash
pip install -r requirements.txt
