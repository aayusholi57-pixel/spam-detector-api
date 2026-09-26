from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Classifier API")

# --- NEW: Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites to fetch. You can restrict this later to just your website's URL.
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"], # Allows all headers
)
# ------------------------

model = joblib.load("spam_model.joblib")

class InferenceRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: InferenceRequest):
    prediction = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]
    confidence = max(probabilities)
    
    return {
        "text": request.text, 
        "prediction": prediction,
        "confidence_score": round(float(confidence), 4)
    }