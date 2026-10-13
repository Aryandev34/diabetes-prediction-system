"""Train and evaluate the SVM model, then persist model + scaler.

Usage:  python -m src.train
"""
from src.diabetes_prediction_system import DEFAULT_MODEL, DEFAULT_SCALER, DiabetesPredictionSystem


def main() -> None:
    system = DiabetesPredictionSystem()
    metrics = system.train()
    system.save()

    print("Diabetes Prediction System — SVM (linear kernel)")
    print(f"Training data size : {metrics['train_size']}")
    print(f"Testing data size  : {metrics['test_size']}")
    print(f"Training accuracy  : {metrics['training_accuracy'] * 100:.2f}%")
    print(f"Testing accuracy   : {metrics['testing_accuracy'] * 100:.2f}%")
    print(f"Confusion matrix   : {metrics['confusion_matrix']}")
    print("\nClassification report:\n", metrics["classification_report"])
    print(f"Saved model  -> {DEFAULT_MODEL}")
    print(f"Saved scaler -> {DEFAULT_SCALER}")


if __name__ == "__main__":
    main()
