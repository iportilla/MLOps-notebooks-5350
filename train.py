"""
Train ONE model and save it with a "model card" (metadata).

    python train.py <data.csv> <version>

Examples:
    python train.py data/sms_real_1k.csv v1
    python train.py data/balanced.csv    v2

Creates:
    models/model_<version>.pkl    the trained model
    models/model_<version>.json   the model card: what data, what metrics, when
"""

import json
import os
import sys
from datetime import datetime

import joblib
from sklearn.model_selection import train_test_split

from common import evaluate, file_fingerprint, load_data, print_metrics, train


def main():
    if len(sys.argv) != 3:
        print("Usage: python train.py <data.csv> <version>")
        sys.exit(1)
    data_path, version = sys.argv[1], sys.argv[2]

    # 1. Load data
    df = load_data(data_path)
    print(f"\nTraining {version} on {data_path} ({len(df)} rows)")
    print(f"  Label mix: {df['label'].value_counts(normalize=True).round(2).to_dict()}")

    # 2. Split 75% train / 25% test (random_state makes it repeatable)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label"])

    # 3. Train and evaluate
    model = train(train_df)
    metrics = evaluate(model, test_df)
    print_metrics(metrics)

    # 4. Save the model AND its model card
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/model_{version}.pkl")
    card = {
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": data_path,
        "data_fingerprint": file_fingerprint(data_path),
        "rows": len(df),
        "algorithm": "LogisticRegression",
        "metrics": metrics,
    }
    with open(f"models/model_{version}.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"\nSaved models/model_{version}.pkl and models/model_{version}.json")


if __name__ == "__main__":
    main()
