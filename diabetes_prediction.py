"""
Standalone end-to-end script: load -> scale -> stratified split ->
train SVM -> evaluate accuracy -> predict a new patient.

Run:  python diabetes_prediction.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load dataset
diabetes_dataset = pd.read_csv("data/diabetes.csv")
print(diabetes_dataset.head())
print("\nShape:", diabetes_dataset.shape)
print("\nOutcome counts:\n", diabetes_dataset["Outcome"].value_counts())
print("\nOutcome mean:\n", diabetes_dataset.groupby("Outcome").mean(numeric_only=True))

# 2. Features / target
X = diabetes_dataset.drop(columns="Outcome")
y = diabetes_dataset["Outcome"]

# 3. Standardization (multi-range diagnostic parameters)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Stratified train-test split + SVM training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)

# 5. Accuracy evaluation
print("\nTraining accuracy:", accuracy_score(y_train, classifier.predict(X_train)))
print("Testing accuracy :", accuracy_score(y_test, classifier.predict(X_test)))

# 6. Automated predictive workflow for a new patient
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # one patient, as a tuple
std_data = scaler.transform(np.asarray(input_data).reshape(1, -1))
prediction = classifier.predict(std_data)
print("\nNew patient:", input_data)
print("The person is Diabetic" if prediction[0] == 1 else "The person is not Diabetic")
