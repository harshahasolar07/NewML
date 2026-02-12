# ML-Assessment – MLOps Lifecycle Extension

## 📌 Overview

This project simulates a simplified production-grade MLOps workflow.

The goal was to extend a basic machine learning training pipeline with:

- Multi-model experimentation  
- Automated model selection  
- Performance-based deployment gating  
- Model version tracking  
- Logging for observability  
- API-based inference deployment  

The system mimics how models are promoted to production in real-world ML systems.

---

## 🚀 Model Training Pipeline

The training pipeline performs the following steps:

1. Loads the Breast Cancer dataset from scikit-learn.
2. Splits the dataset into training and validation sets.
3. Trains two candidate models:
   - Logistic Regression
   - Random Forest
4. Evaluates both models using F1-score.
5. Automatically selects the better-performing model.
6. Compares the selected model against the current production baseline stored in `registry.json`.
7. Deploys the model only if:

   **New F1-score ≥ Production F1-score**

8. Updates the registry with:
   - New model version
   - Model name
   - F1-score

This simulates a real-world CI/CD promotion gate for machine learning models.

---

## 📁 Model Registry

A lightweight JSON-based registry (`registry.json`) is used to simulate production model tracking.

It stores:
- Current production version  
- Model name  
- F1-score  

This enables performance-based deployment decisions.

---

## 📊 Logging & Observability

Training metrics and deployment decisions are logged to:

```
training.log
```

This simulates monitoring and audit logging in production environments.

---

## 🌐 API Deployment

The production model is served using FastAPI.

### Run the API

```
uvicorn app:app --reload
```

### Access Swagger UI

```
http://127.0.0.1:8000/docs
```

### POST `/predict` Request Format

```json
{
  "features": [ ... 30 numeric values ... ]
}
```

The API loads the currently deployed production model and returns a prediction.

---

## ⚙️ Setup Instructions

1. Create a virtual environment:

```
python -m venv venv
```

2. Activate (Windows):

```
venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Train and evaluate models:

```
python train.py
```

5. Start the API server:

```
uvicorn app:app --reload
```

---

## 🧠 Assumptions

- A local JSON file is used as a lightweight model registry.
- A single-model production environment is assumed.
- No external experiment tracking tools (e.g., MLflow) were integrated due to time constraints.
- No containerization (Docker) was implemented for simplicity.

---

## 🚧 Limitations

- No automated CI/CD integration.
- No historical experiment tracking.
- No containerized deployment.
- No cloud hosting.

---

## 🤖 Reflection on Using a Coding Assistant

A coding assistant was used to accelerate boilerplate development and API structuring. It was particularly helpful in scaffolding the multi-model training logic and FastAPI integration.

All architectural decisions — including model evaluation strategy, performance gating condition, registry design, and deployment structure — were intentionally validated to simulate a realistic MLOps lifecycle.

The assistant improved development speed while maintaining full control over design and implementation choices.