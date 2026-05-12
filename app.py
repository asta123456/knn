import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load Iris Dataset

data = load_iris()

X = data.data
y = data.target

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create KNN Classifier

knn_classifier = KNeighborsClassifier(
    n_neighbors=3,
    metric='minkowski',
    p=2
)

# Train Model

knn_classifier.fit(X_train, y_train)

# Predict Test Data

y_pred = knn_classifier.predict(X_test)

# Accuracy Score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report

class_report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(class_report)

# Confusion Matrix Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

# Training Accuracy

train_accuracy = knn_classifier.score(X_train, y_train)

print("\nTraining Accuracy:", train_accuracy)

# Testing Accuracy

test_accuracy = knn_classifier.score(X_test, y_test)

print("Testing Accuracy:", test_accuracy)

# Predict New Sample

new_sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = knn_classifier.predict(new_sample)

print("\nPredicted Flower:")
print(data.target_names[prediction][0])

# Find Best K Value

accuracy_scores = []

k_values = range(1, 21)

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred_k = knn.predict(X_test)

    accuracy_scores.append(
        accuracy_score(y_test, y_pred_k)
    )

# Plot Accuracy vs K Value

plt.figure(figsize=(10,6))

plt.plot(k_values, accuracy_scores, marker='o')

plt.title("Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# Best K Value

best_k = k_values[np.argmax(accuracy_scores)]

print("\nBest K Value:", best_k)
print("Best Accuracy:", max(accuracy_scores))