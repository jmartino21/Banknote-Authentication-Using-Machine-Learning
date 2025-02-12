import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load dataset
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}. Please make sure the dataset is in the correct directory.")
    df = pd.read_csv(filename, names=["f1", "f2", "f3", "f4", "Class"])
    df['Color'] = np.where(df['Class'] == 0, 'green', 'red')
    return df

# Split dataset into training and test sets
def split_data(df, test_size=0.5, random_state=100):
    return train_test_split(df, test_size=test_size, random_state=random_state)

# Compute basic statistics
def compute_statistics(df):
    stats = {
        "overall_mean": df[["f1", "f2", "f3", "f4"]].mean(),
        "overall_std": df[["f1", "f2", "f3", "f4"]].std(),
        "class_mean": df.groupby('Class')[["f1", "f2", "f3", "f4"]].mean(),
        "class_std": df.groupby('Class')[["f1", "f2", "f3", "f4"]].std()
    }
    return stats

# Train and evaluate a simple rule-based classifier
def simple_classifier(df):
    df["prediction"] = np.where((df['f1'] > 0) & (df['f2'] > 0) & (df['f3'] < 0), 'green', 'red')
    return df

# Compute classification metrics
def compute_metrics(df):
    size = df.shape[0]
    TP = ((df['Color'] == 'green') & (df['prediction'] == 'green')).sum() / size
    FP = ((df['prediction'] == 'green') & (df['Color'] == 'red')).sum() / size
    TN = ((df['prediction'] == 'red') & (df['Color'] == 'red')).sum() / size
    FN = ((df['prediction'] == 'red') & (df['Color'] == 'green')).sum() / size
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return {"Accuracy": accuracy, "TPR": TP / (TP + FN), "TNR": TN / (TN + FP)}

# Train and evaluate k-NN classifier
def knn_classifier(x_train, y_train, x_test, y_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Train and evaluate logistic regression model
def logistic_regression(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(random_state=0)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Main script
if __name__ == "__main__":
    # Load and split data
    file_path = 'data_banknote_authentication.txt'  # Ensure this file is present
    df = load_data(file_path)
    df_train, df_test = split_data(df)

    # Compute statistics
    stats = compute_statistics(df)
    print("Dataset Statistics:", stats)

    # Evaluate simple classifier
    df_test = simple_classifier(df_test)
    simple_metrics = compute_metrics(df_test)
    print("Simple Classifier Metrics:", simple_metrics)

    # Prepare data for ML models
    x_train, y_train = df_train[["f1", "f2", "f3", "f4"]], df_train["Class"]
    x_test, y_test = df_test[["f1", "f2", "f3", "f4"]], df_test["Class"]

    # Evaluate k-NN
    knn_accuracy, y_pred_knn = knn_classifier(x_train, y_train, x_test, y_test)
    print("KNN Accuracy (k=3):", knn_accuracy)

    # Evaluate Logistic Regression
    lr_accuracy, y_pred_lr = logistic_regression(x_train, y_train, x_test, y_test)
    print("Logistic Regression Accuracy:", lr_accuracy)

    # Visualizing the data
    sns.pairplot(df, hue='Class')
    plt.show()
