"""Automated prediction workflow for a new patient.

Usage:
    python -m src.predict --input 5,166,72,19,175,25.8,0.587,51
    python -m src.predict            # interactive prompts

Feature order: Pregnancies, Glucose, BloodPressure, SkinThickness,
               Insulin, BMI, DiabetesPedigreeFunction, Age

Note: educational ML demo only — not a medical diagnosis.
"""
import argparse

from src.diabetes_prediction_system import FEATURES, DiabetesPredictionSystem


def parse_args():
    p = argparse.ArgumentParser(description="Predict diabetes status for one patient.")
    p.add_argument("--input", help="Comma-separated 8 values, e.g. 5,166,72,19,175,25.8,0.587,51")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        patient = [float(v) for v in args.input.split(",")]
    else:
        patient = [float(input(f"Enter {name}: ")) for name in FEATURES]

    try:
        system = DiabetesPredictionSystem.load()
    except FileNotFoundError:
        print("Trained model not found — training first ...\n")
        system = DiabetesPredictionSystem()
        system.train()
        system.save()

    result = system.predict_patient(patient)
    print("\nInput data        :", tuple(patient))
    print("Prediction        :", result["classification"])
    print(result["message"])


if __name__ == "__main__":
    main()
