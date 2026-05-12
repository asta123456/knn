import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset

data = load_iris()

X = data.data
y = data.target

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

# Streamlit UI

st.title("Iris Flower Classification App")

st.write("Predict Iris flower species using KNN algorithm")

# User Input

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)

sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.5)

petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)

petal_width = st.slider("Petal Width", 0.1, 3.0, 0.2)

# Prediction

input_data = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
])

prediction = model.predict(input_data)

species = data.target_names[prediction][0]

# Output

st.success(f"Predicted Flower Species: {species}")

# Accuracy

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")
