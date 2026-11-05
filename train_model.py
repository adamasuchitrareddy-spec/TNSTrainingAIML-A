import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.joblib")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=4,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Test accuracy: {:.2f}%".format(accuracy_score(y_test, predictions) * 100))
    print("\nClassification report:\n", classification_report(y_test, predictions))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
