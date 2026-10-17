"""
An automated retraining pipeline with a QUALITY GATE.

    python pipeline.py <new_data.csv>

Example:
    python pipeline.py data/new_week_ok.csv

Steps:
    1. Ingest   - combine our base training data with the new batch
    2. Retrain  - train a new candidate model
    3. Evaluate - score it on a FIXED test set (data/test_set.csv)
    4. Version  - save it with a timestamp, log it in models/registry.csv
    5. Promote  - replace models/production.pkl ONLY if the gate passes
"""

import csv
import os
import shutil
import sys
from datetime import datetime

import joblib
import pandas as pd

from common import evaluate, load_data, print_metrics, train

BASE_DATA = "data/balanced.csv"
TEST_SET = "data/test_set.csv"
PRODUCTION_MODEL = "models/production.pkl"
REGISTRY = "models/registry.csv"

# ---------------- Quality gate ----------------
ACCURACY_MIN = 0.85
# TODO (Part 4): add a minimum spam recall, for example 0.80
# SPAM_RECALL_MIN = ...


def passes_gate(metrics: dict) -> bool:
    """Return True if the candidate model is good enough for production."""
    ok = metrics["accuracy"] >= ACCURACY_MIN
    # TODO (Part 4): also require metrics["spam_recall"] >= SPAM_RECALL_MIN
    return ok
# ----------------------------------------------


def log_to_registry(row: dict) -> None:
    """Append one line per pipeline run to models/registry.csv."""
    new_file = not os.path.exists(REGISTRY)
    with open(REGISTRY, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <new_data.csv>")
        sys.exit(1)
    new_data_path = sys.argv[1]
    os.makedirs("models", exist_ok=True)

    print("\n[1/5] Ingest")
    data = pd.concat([load_data(BASE_DATA), load_data(new_data_path)], ignore_index=True)
    print(f"  {BASE_DATA} + {new_data_path} = {len(data)} rows")

    print("\n[2/5] Retrain")
    candidate = train(data)
    print("  Candidate model trained")

    print(f"\n[3/5] Evaluate on fixed test set ({TEST_SET})")
    metrics = evaluate(candidate, load_data(TEST_SET))
    print_metrics(metrics)

    print("\n[4/5] Version")
    version = "model_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(candidate, f"models/{version}.pkl")
    print(f"  Saved models/{version}.pkl")

    print("\n[5/5] Quality gate")
    promoted = passes_gate(metrics)
    if promoted:
        shutil.copy(f"models/{version}.pkl", PRODUCTION_MODEL)
        print(f"  PASSED -> promoted to {PRODUCTION_MODEL}")
    else:
        print(f"  FAILED -> {PRODUCTION_MODEL} was NOT changed")

    log_to_registry({
        "version": version,
        "new_data": new_data_path,
        "train_rows": len(data),
        "accuracy": metrics["accuracy"],
        "spam_recall": metrics["spam_recall"],
        "promoted": promoted,
    })
    print(f"  Logged in {REGISTRY}")


if __name__ == "__main__":
    main()
