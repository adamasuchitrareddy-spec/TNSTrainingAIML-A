# Diabetes Risk Prediction Using Supervised Learning (Streamlit)

A beginner-friendly machine-learning project with a Streamlit interface, a CSV dataset, and a Random Forest classifier.

## Important dataset note
The included `data/diabetes.csv` contains **synthetic/artificial demonstration records** generated for this project. It is not an official Pima Indians Diabetes dataset, not real patient data, and not suitable for clinical use. The model output is for learning and project demonstration only—not diagnosis or medical advice.

## Project structure

```text
diabetes_risk_prediction_streamlit/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── diabetes.csv
└── models/
    └── (model file is created after training)
```

## Requirements
- Python 3.7 (the requirements are pinned for Python 3.7 compatibility)
- Internet connection for installing packages
- VS Code or any Python editor

## Windows setup (PowerShell)

Open PowerShell in this project folder and run:

```powershell
py -3.7 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`).

## How it works
1. `data/diabetes.csv` supplies eight input features and the target column `Outcome` (0 or 1).
2. `train_model.py` loads the CSV, splits it into training and test sets, trains a Random Forest classifier, prints evaluation metrics, and saves the model using Joblib.
3. `app.py` loads the saved model and displays input fields in a Streamlit web app.
4. On clicking **Predict risk**, the app displays the predicted class and the model-estimated probability.

## Dataset columns
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = class 0, 1 = class 1)

## Common fixes
- If you see `ModuleNotFoundError: No module named 'sklearn'`, activate the virtual environment and run `pip install -r requirements.txt`.
- If `streamlit` is not recognized, run `python -m streamlit run app.py`.
- Always use the same Python interpreter/virtual environment to install packages and run the project.

## Disclaimer
This is a student learning project. The synthetic dataset and model are not validated for real-world healthcare decisions.
