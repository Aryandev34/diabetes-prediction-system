"""
Diabetes Prediction System — core binary-classification pipeline.

Uses Python, Pandas, NumPy and Scikit-Learn:
  * features are separated from the ``Outcome`` target,
  * multi-range diagnostic parameters are standardized with StandardScaler,
  * a Support Vector Machine (SVM) is trained on a stratified train/test split,
  * new patient arrays are ingested and returned as a risk classification.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
LABELS = {0: "Non-Diabetic", 1: "Diabetic"}
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "diabetes.csv"
DEFAULT_MODEL = ROOT / "models" / "svm_model.pkl"
DEFAULT_SCALER = ROOT / "models" / "scaler.pkl"


class DiabetesPredictionSystem:
    """SVM pipeline: StandardScaler -> SVC (linear kernel)."""

    def __init__(self, kernel: str = "linear", random_state: int = 2):
        self.scaler = StandardScaler()
        self.classifier = SVC(kernel=kernel)
        self.random_state = random_state
        self.is_trained = False

    # ------------------------------------------------------------------ data
    @staticmethod
    def load_data(csv_path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")
        return df

    # ----------------------------------------------------------------- train
    def train(self, csv_path: str | Path = DEFAULT_DATA, test_size: float = 0.2) -> dict:
        dataset = self.load_data(csv_path)
        X, y = dataset[FEATURES], dataset[TARGET]

        # Stratified split keeps the diabetic / non-diabetic ratio in both sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        # Standardize multi-range parameters (fit on training data only).
        X_train_std = self.scaler.fit_transform(X_train)
        X_test_std = self.scaler.transform(X_test)

        self.classifier.fit(X_train_std, y_train)
        self.is_trained = True

        train_acc = accuracy_score(y_train, self.classifier.predict(X_train_std))
        test_acc = accuracy_score(y_test, self.classifier.predict(X_test_std))
        self.metrics_ = {
            "training_accuracy": round(float(train_acc), 4),
            "testing_accuracy": round(float(test_acc), 4),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "classification_report": classification_report(
                y_test, self.classifier.predict(X_test_std), target_names=["Non-Diabetic", "Diabetic"]
            ),
            "confusion_matrix": confusion_matrix(y_test, self.classifier.predict(X_test_std)).tolist(),
        }
        return self.metrics_

    # --------------------------------------------------------------- predict
    def predict_patient(self, patient_data) -> dict:
        """Ingest one patient's 8 metrics and output its risk classification."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained/loaded. Run train() or load() first.")
        values = np.asarray(patient_data, dtype=float)
        if values.shape != (len(FEATURES),):
            raise ValueError(
                f"Expected {len(FEATURES)} values in order {FEATURES}, got shape {values.shape}"
            )
        standardized = self.scaler.transform(
            pd.DataFrame(values.reshape(1, -1), columns=FEATURES)
        )
        prediction = int(self.classifier.predict(standardized)[0])
        return {
            "input": dict(zip(FEATURES, values.tolist())),
            "prediction": prediction,
            "classification": LABELS[prediction],
            "message": (
                "The person is Diabetic" if prediction == 1
                else "The person is not Diabetic"
            ),
        }

    # ----------------------------------------------------------- persistence
    def save(self, model_path=DEFAULT_MODEL, scaler_path=DEFAULT_SCALER) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, model_path=DEFAULT_MODEL, scaler_path=DEFAULT_SCALER) -> "DiabetesPredictionSystem":
        system = cls()
        with open(model_path, "rb") as f:
            system.classifier = pickle.load(f)
        with open(scaler_path, "rb") as f:
            system.scaler = pickle.load(f)
        system.is_trained = True
        return system
