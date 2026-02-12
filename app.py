from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model_v2.pkl")

class Features(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(data: Features):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}