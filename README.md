# Spam Detector API

A FastAPI service that classifies SMS messages as spam or ham using a TF-IDF + Logistic Regression pipeline.

## Features

- FastAPI REST API with interactive Swagger documentation
- TF-IDF unigram and bigram features
- Reproducible Logistic Regression training
- Stratified train/test evaluation
- Confidence score from model probabilities
- Health endpoint for deployment checks
- Docker build that creates the model artifact
- Automated GitHub Actions tests

## Structure

spam-detector-api/
  .github/workflows/ci.yml
  tests/test_api.py
  .gitignore
  Dockerfile
  main.py
  requirements.txt
  train.py

## Local setup

Python 3.11+ is recommended.

1. Create and activate a virtual environment.
2. Install dependencies with: pip install -r requirements.txt
3. Train the model with: python train.py
4. Start the API with: uvicorn main:app --reload
5. Open Swagger at: http://localhost:8000/docs

## API

GET /health

Reports service status and whether the model artifact exists.

POST /predict

Example request:

{
  "text": "Congratulations! You have won a free prize."
}

Example response:

{
  "text": "Congratulations! You have won a free prize.",
  "prediction": "spam",
  "confidence_score": 0.9876
}

The confidence score is the highest class probability returned by the classifier. It is not a calibrated probability of correctness.

## Docker

Build and run:

docker build -t spam-detector-api .
docker run --rm -p 8080:8080 spam-detector-api

Then open: http://localhost:8080/docs

## Dataset and model

train.py downloads the public SMS Spam Collection dataset from the URL defined in the script. The generated spam_model.joblib file is ignored by Git because it is a binary build artifact.

For production, a versioned dataset and external model artifact storage would improve reproducibility.

## Tests

Run:

pytest -q

CI also compiles the Python source and runs the API tests on pushes and pull requests.
