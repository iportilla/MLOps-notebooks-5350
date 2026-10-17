"""
Create the solution models for the MLOps notebooks.

    python make_solution_models.py           # skips models that already exist
    python make_solution_models.py --force   # replaces them

Stuck on train_model.ipynb? This script trains v1 and v2 exactly the way that
notebook's solution (Part B) does, and saves the four files that
mlops_basics.ipynb and mlops_pipeline.ipynb start from:

    models/model_nb_v1.pkl  + models/model_nb_v1.json   (imbalanced data)
    models/model_nb_v2.pkl  + models/model_nb_v2.json   (balanced data)

The models are retrained rather than shipped as .pkl files, so they work with
whatever scikit-learn version you have installed.
"""

import hashlib
import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["length", "punct"]
LABEL = "label"
TEST_SET = "data/test_set.csv"
MODELS = [
    ("nb_v1", "data/sms_real_1k.csv"),   # real, imbalanced: 85% ham
    ("nb_v2", "data/balanced.csv"),      # balanced: 1,000 ham + 1,000 spam
]


def file_fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def evaluate(model, df):
    y_true = df[LABEL]
    y_pred = model.predict(df[FEATURES])
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, pos_label="spam", zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, pos_label="spam", zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, pos_label="spam", zero_division=0), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).tolist(),
    }


def train(data_file):
    """Same steps as train_model.ipynb: 75/25 stratified split with random_state=42, then logistic regression."""
    df = pd.read_csv(data_file)
    train_df, _ = train_test_split(df, test_size=0.25, random_state=42, stratify=df[LABEL])
    return LogisticRegression(max_iter=1000).fit(train_df[FEATURES], train_df[LABEL]), len(df)


def main():
    force = "--force" in sys.argv[1:]
    for f in [TEST_SET] + [data_file for _, data_file in MODELS]:
        if not os.path.exists(f):
            sys.exit(f"{f} is missing. Run this script from the folder that contains make_data.py.")

    os.makedirs("models", exist_ok=True)
    fixed_test = pd.read_csv(TEST_SET)

    for version, data_file in MODELS:
        pkl, card_path = f"models/model_{version}.pkl", f"models/model_{version}.json"
        if os.path.exists(pkl) and not force:
            print(f"{pkl} already exists, skipped (use --force to replace it)")
            continue
        model, rows = train(data_file)
        joblib.dump(model, pkl)
        card = {
            "version": version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_file": data_file,
            "data_fingerprint": file_fingerprint(data_file),
            "rows": rows,
            "algorithm": "LogisticRegression",
            "metrics": evaluate(model, fixed_test),
        }
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
        m = card["metrics"]
        print(f"saved {pkl} and {card_path}: accuracy {m['accuracy']:.1%}, recall {m['recall']:.1%} on {TEST_SET}")

    print("\nDone. You can now run mlops_basics.ipynb or mlops_pipeline.ipynb.")


if __name__ == "__main__":
    main()
