from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "resumes.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "resume_classifier.joblib"

df = pd.read_csv(DATA)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.2%}")
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(pipeline, MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")
