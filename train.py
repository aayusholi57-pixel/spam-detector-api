import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

print("Downloading the standard 5,500+ SMS dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

print("Building TF-IDF & Logistic Regression pipeline...")
model = make_pipeline(
    TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), 
    LogisticRegression(max_iter=1000)
)

print("Training model...")
model.fit(df['text'], df['label'])

predictions = model.predict(df['text'])
print(f"Training Accuracy: {accuracy_score(df['label'], predictions):.4f}")

joblib.dump(model, "spam_model.joblib")
print("Stronger model saved as spam_model.joblib!")
