from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Classifier API")
model = joblib.load("spam_model.joblib")

class InferenceRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: InferenceRequest):
    prediction = model.predict([request.text])[0]
    
    # Calculate how confident the model is
    probabilities = model.predict_proba([request.text])[0]
    confidence = max(probabilities)
    
    return {
        "text": request.text, 
        "prediction": prediction,
        "confidence_score": round(float(confidence), 4)
    }
