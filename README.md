# Diabetes Prediction System (Machine Learning Project)

Binary classification pipeline using **Python, Pandas, NumPy and Scikit-Learn**
to predict diabetes status from patient health metrics, using a
**Support Vector Machine (SVM)**.

## Overview

- Loads the Pima Indians Diabetes dataset (768 patients, 8 diagnostic features).
- Separates features from the `Outcome` target and inspects class balance.
- Standardizes multi-range diagnostic parameters with `StandardScaler`.
- Trains an SVM (`SVC`, linear kernel) on a **stratified** train–test split.
- Evaluates accuracy (training/testing), classification report & confusion matrix.
- Provides an automated prediction workflow: a new patient's input array is
  standardized with the *fitted* scaler and classified as
  **Diabetic / Non-Diabetic**.

> ⚠️ Educational ML project only — its output is not a medical diagnosis.

## Dataset

`data/diabetes.csv` — Pima Indians Diabetes Database, originally from the
National Institute of Diabetes and Digestive and Kidney Diseases
(via https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset, Apache-2.0,
see `data/LICENSE.txt`). 768 women, 500 non-diabetic / 268 diabetic.

| Feature | Meaning |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2h OGTT) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (µU/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood |
| `Age` | Age (years) |
| `Outcome` | Target — `1` diabetic, `0` non-diabetic |

## Project Structure

```
├── data/diabetes.csv                    # dataset
├── notebooks/diabetes_prediction.ipynb  # step-by-step notebook
├── src/
│   ├── diabetes_prediction_system.py    # reusable DiabetesPredictionSystem pipeline
│   ├── train.py                         # train, evaluate, save model + scaler
│   └── predict.py                       # CLI prediction for a new patient
├── diabetes_prediction.py               # standalone end-to-end script
├── tests/test_prediction.py
├── models/                              # svm_model.pkl & scaler.pkl (generated)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
pip install pytest                                   # only for tests
```

## Usage

```bash
# 1. Train + evaluate (saves models/svm_model.pkl and models/scaler.pkl)
python -m src.train

# 2. Predict for a new patient (8 comma-separated values, feature order as above)
python -m src.predict --input 5,166,72,19,175,25.8,0.587,51
# -> The person is Diabetic

python -m src.predict --input 1,85,66,29,0,26.6,0.351,31
# -> The person is not Diabetic

python -m src.predict   # interactive prompts (trains first if no model found)

# Or run the all-in-one script / open the notebook
python diabetes_prediction.py
jupyter notebook notebooks/diabetes_prediction.ipynb

# Tests
python -m pytest -q
```

## Results

Reproduced with `python -m src.train` (stratified 80/20 split, `random_state=2`):

| Metric | Score |
|---|---|
| Training accuracy | ~78.6% |
| Testing accuracy | ~77.3% |

(Exact numbers are printed by the training script and may vary slightly
across Scikit-Learn versions.)

## Tech Stack

Python · Pandas · NumPy · Scikit-Learn (`StandardScaler`, `SVC`, `train_test_split`)
