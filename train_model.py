from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "resumes.csv"
MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "resume_classifier.joblib"
REFERENCE_PATH = MODEL_DIR / "reference_data.joblib"


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Total records: {len(df)}")
print("\nJob categories:")
print(df["label"].value_counts())


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)


# --------------------------------------------------
# MACHINE LEARNING PIPELINE
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)

print(f"\nModel saved to:")
print(MODEL_PATH)


# --------------------------------------------------
# SAVE REFERENCE DATA
# --------------------------------------------------
# Used by Task 3 for similarity checking.

reference_data = {
    "texts": X_train.tolist(),
    "labels": y_train.tolist()
}

joblib.dump(
    reference_data,
    REFERENCE_PATH
)

print("\nReference data saved to:")
print(REFERENCE_PATH)

print("\nModel training completed successfully.")