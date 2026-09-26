from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("spam_model.joblib")

app = FastAPI(
    title="Spam Detector API",
    description="Spam classification API using TF-IDF and Logistic Regression.",
    version="1.0.0",
)


class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence_score: float


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run python train.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: InferenceRequest):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    probabilities = model.predict_proba([request.text])[0]
    prediction = str(model.classes_[probabilities.argmax()])
    confidence = float(probabilities.max())

    return PredictionResponse(
        text=request.text,
        prediction=prediction,
        confidence_score=round(confidence, 4),
    )
