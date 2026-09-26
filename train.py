from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_PATH = Path(__file__).with_name("spam_model.joblib")


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def main() -> None:
    print("Downloading the SMS spam dataset...")
    df = pd.read_csv(DATA_URL, sep="\t", header=None, names=["label", "text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print("Training TF-IDF + Logistic Regression pipeline...")
    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
