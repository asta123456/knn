# Iris Flower Classification using KNN

This project is a Machine Learning web application built using Streamlit and K-Nearest Neighbors (KNN) Classification Algorithm.

The application predicts the species of an Iris flower based on flower measurements.

---

# Project Description

The Iris dataset is one of the most popular datasets in Machine Learning.  
This project uses the KNN Classification algorithm to classify Iris flowers into:

- Setosa
- Versicolor
- Virginica

The user enters flower measurements and the model predicts the flower species.

---

# Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

# Machine Learning Algorithm

K-Nearest Neighbors (KNN) Classification Algorithm

Parameters used:

- n_neighbors = 3
- metric = minkowski
- p = 2

---

# Dataset

The project uses the Iris dataset available in Scikit-learn.

Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target:
- Iris Flower Species

---

# Project Workflow

1. Load Iris Dataset
2. Split Dataset into Training and Testing Data
3. Train KNN Model
4. Predict Test Data
5. Evaluate Model Accuracy
6. Build Streamlit Web Application
7. Deploy Application

---

# Model Evaluation

The following evaluation metrics are used:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

# Installation

Install required libraries using:

```bash
pip install -r requirements.txt
