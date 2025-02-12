# Banknote Authentication Using Machine Learning

## Overview
This project explores machine learning techniques for classifying banknotes as genuine or counterfeit using features extracted from banknotes. The dataset consists of numerical features obtained via wavelet transforms.

## Features
- **f1**: Wavelet variance
- **f2**: Wavelet skewness
- **f3**: Wavelet kurtosis
- **f4**: Image entropy
- **Class**: 0 (genuine) or 1 (counterfeit)

## Models Implemented
- Simple Rule-Based Classifier
- k-Nearest Neighbors (k-NN)
- Logistic Regression

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset
This project uses the `data_banknote_authentication.txt` dataset. Ensure the dataset is placed in the same directory as the script. If missing, download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).

## Usage
### Running the Script
Execute the script using:
```bash
python banknote_authentication.3.677.py
```

The script performs the following:
1. Loads and preprocesses the dataset.
2. Splits the dataset into training and testing sets.
3. Computes basic statistical summaries.
4. Trains and evaluates different models.
5. Displays performance metrics.
6. Generates visualizations for feature distribution.

## Output
- Accuracy and classification metrics for each model.
- Pairplot visualization of features colored by class labels.

## License
This project is open-source and available for modification and use.

