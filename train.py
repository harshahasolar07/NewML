


import json
import joblib
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logging setup
logging.basicConfig(filename="training.log", level=logging.INFO)

print("Starting training pipeline...")

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# -----------------------------
# Train Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_f1 = f1_score(y_test, log_preds)

print("Logistic Regression F1:", log_f1)

# -----------------------------
# Train Random Forest
# -----------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_f1 = f1_score(y_test, rf_preds)

print("Random Forest F1:", rf_f1)

# -----------------------------
# Select Best Model
# -----------------------------
if log_f1 > rf_f1:
    best_model = log_model
    best_f1 = log_f1
    best_name = "LogisticRegression"
else:
    best_model = rf_model
    best_f1 = rf_f1
    best_name = "RandomForest"

print(f"Best Model: {best_name} with F1: {best_f1}")

logging.info(f"Logistic F1: {log_f1}")
logging.info(f"RandomForest F1: {rf_f1}")
logging.info(f"Best Model: {best_name} | F1: {best_f1}")

# -----------------------------
# Load Production Baseline
# -----------------------------
with open("registry.json", "r") as f:
    registry = json.load(f)

baseline_f1 = registry["production"]["f1_score"]

print("Baseline Production F1:", baseline_f1)

# -----------------------------
# CONDITIONAL DEPLOYMENT GATE
# -----------------------------
if best_f1 >= baseline_f1:
    print("✅ Model approved for deployment.")

    joblib.dump(best_model, "model_v2.pkl")

    registry["production"] = {
        "version": "v2",
        "model_name": best_name,
        "f1_score": best_f1
    }

    with open("registry.json", "w") as f:
        json.dump(registry, f, indent=4)

    logging.info("Model deployed successfully.")

else:
    print("❌ Model rejected. Below production baseline.")
    logging.info("Model rejected.")