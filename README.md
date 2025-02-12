# Banknote Authentication Using Machine Learning

## Overview
This project explores different machine learning models for classifying banknotes as genuine or counterfeit. Models include a rule-based classifier, k-Nearest Neighbors (KNN), and Logistic Regression.

## Dataset
The dataset contains features extracted from banknotes:
- f1: Wavelet variance
- f2: Wavelet skewness
- f3: Wavelet kurtosis
- f4: Image entropy

## Methodology
- A simple rule-based classifier was implemented based on feature thresholds.
- KNN models were trained and tested with different k values (3, 5, 7, 9, 11).
- A logistic regression model was implemented for comparison.
- Performance metrics (accuracy, TPR, TNR) were calculated.

## Results
- The simple rule-based classifier achieved **59% accuracy**.
- KNN (k=3) outperformed other models, achieving **100% accuracy**.
- Logistic Regression performed similarly to KNN but had a lower true positive rate.

## Dependencies
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
```bash
python_project_3.677.py
