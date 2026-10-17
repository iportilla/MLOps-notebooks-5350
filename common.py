"""
Shared helpers for the MLOps 101 lab.

Every script in this folder uses these few functions, so the
"how we train" and "how we measure" logic lives in ONE place.
"""

import hashlib
import string

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

FEATURES = ["length", "punct"]
LABEL = "label"


def extract_features(text: str) -> dict:
    """Turn a raw SMS message into the two numbers our model understands."""
    return {
        "length": len(text),
        "punct": sum(1 for ch in text if ch in string.punctuation),
    }


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV with columns: label, length, punct."""
    return pd.read_csv(path)


def file_fingerprint(path: str) -> str:
    """Short SHA-256 hash of a file. Same data -> same fingerprint."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def train(train_df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression spam classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[FEATURES], train_df[LABEL])
    return model


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Return accuracy, spam recall and the confusion matrix."""
    y_true = test_df[LABEL]
    y_pred = model.predict(test_df[FEATURES])
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "spam_recall": round(recall_score(y_true, y_pred, pos_label="spam", zero_division=0), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).tolist(),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics in a student-friendly way."""
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print(f"  Accuracy    : {metrics['accuracy']:.1%}")
    print(f"  Spam recall : {metrics['spam_recall']:.1%}   (share of real spam we caught)")
    print("  Confusion matrix:")
    print("                    predicted ham   predicted spam")
    print(f"    actual ham      {tn:>13}   {fp:>14}")
    print(f"    actual spam     {fn:>13}   {tp:>14}")
