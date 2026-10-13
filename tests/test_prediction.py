import numpy as np
import pytest

from src.diabetes_prediction_system import FEATURES, DiabetesPredictionSystem


@pytest.fixture(scope="module")
def system():
    s = DiabetesPredictionSystem()
    s.train()
    return s


def test_training_accuracy_is_reasonable(system):
    assert system.metrics_["testing_accuracy"] > 0.70
    assert system.metrics_["train_size"] + system.metrics_["test_size"] == 768


def test_predict_diabetic_patient(system):
    result = system.predict_patient((5, 166, 72, 19, 175, 25.8, 0.587, 51))
    assert result["prediction"] == 1
    assert result["classification"] == "Diabetic"


def test_predict_rejects_wrong_number_of_features(system):
    with pytest.raises(ValueError):
        system.predict_patient((1, 85, 66))


def test_feature_order():
    assert FEATURES == [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
