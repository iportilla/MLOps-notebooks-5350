"""
Evaluate an MLflow model (for example an AutoML model) on a labeled test set.

    python evaluate.py --model <mlflow model folder> --test_data <csv> --metrics_out <folder>

Writes <metrics_out>/metrics.json with accuracy, precision, recall and F1 for
the positive class (spam), the confusion matrix and the share predicted positive.
Used as a step of the Azure ML pipeline in pipelines/evaluate-and-gate.yml.
"""

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def predict_labels(model, X):
    """Call an MLflow pyfunc model and return a flat array of string labels."""
    predictions = model.predict(X)
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.iloc[:, 0]
    return np.asarray(predictions).astype(str).ravel()


def compute_metrics(y_true, y_pred, positive_label="spam"):
    labels = sorted(set(y_true) | set(y_pred) | {positive_label})
    negative = [lab for lab in labels if lab != positive_label]
    order = negative[:1] + [positive_label]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)), 3),
        "predicted_positive_share": round(float(np.mean(y_pred == positive_label)), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=order).tolist(),
        "confusion_matrix_labels": order,
        "rows": int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="folder of an MLflow model")
    parser.add_argument("--test_data", required=True, help="CSV with the features and a label column")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--positive_label", default="spam")
    parser.add_argument("--metrics_out", required=True)
    args = parser.parse_args()

    model = mlflow.pyfunc.load_model(args.model)
    test = pd.read_csv(args.test_data)
    y_true = test[args.label_column].astype(str).to_numpy()
    y_pred = predict_labels(model, test.drop(columns=[args.label_column]))
    metrics = compute_metrics(y_true, y_pred, args.positive_label)

    os.makedirs(args.metrics_out, exist_ok=True)
    with open(os.path.join(args.metrics_out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    try:  # shows up on the step's Metrics tab in Azure ML studio
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
    except Exception as exc:
        print(f"(MLflow logging skipped: {exc})")


if __name__ == "__main__":
    main()
