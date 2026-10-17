"""
Quality gate and champion/challenger decision.

One copy of the rules, used by the Azure ML pipeline step, by CI and by the notebook:

    python gate.py --champion_metrics <folder> --challenger_metrics <folder> --decision <folder>

Each metrics folder holds a metrics.json written by evaluate.py. The decision
folder gets a decision.json: {"promote": true/false, "reasons": [...], ...}.
"""

import argparse
import json
import os

ACCURACY_MIN = 0.85     # absolute rule 1
RECALL_MIN = 0.80       # absolute rule 2: catch at least 80% of spam
TOLERANCE = 0.01        # no-regression rule: at most 1 point worse than the champion
NO_REGRESSION_ON = ("recall", "f1")   # the metrics users care about (not precision: see lab section 3c)


def passes_gate(metrics, accuracy_min=ACCURACY_MIN, recall_min=RECALL_MIN):
    """Absolute thresholds: is the model good enough at all?"""
    return metrics["accuracy"] >= accuracy_min and metrics["recall"] >= recall_min


def decide(champion, challenger, accuracy_min=ACCURACY_MIN, recall_min=RECALL_MIN, tolerance=TOLERANCE):
    """Should the challenger replace the champion?

    It must pass the absolute gate AND must not be worse than the champion on
    recall or F1 by more than `tolerance`.
    """
    reasons = []
    if challenger["accuracy"] < accuracy_min:
        reasons.append(f"accuracy {challenger['accuracy']:.3f} < {accuracy_min}")
    if challenger["recall"] < recall_min:
        reasons.append(f"recall {challenger['recall']:.3f} < {recall_min}")
    for metric in NO_REGRESSION_ON:
        if challenger[metric] < champion[metric] - tolerance:
            reasons.append(f"{metric} {challenger[metric]:.3f} is worse than the champion's {champion[metric]:.3f}")
    return {
        "promote": not reasons,
        "reasons": reasons or ["passes the gate and does not regress on " + ", ".join(NO_REGRESSION_ON)],
        "rules": {"accuracy_min": accuracy_min, "recall_min": recall_min, "tolerance": tolerance},
        "champion": champion,
        "challenger": challenger,
    }


def read_metrics(folder):
    with open(os.path.join(folder, "metrics.json")) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion_metrics", required=True)
    parser.add_argument("--challenger_metrics", required=True)
    parser.add_argument("--accuracy_min", type=float, default=ACCURACY_MIN)
    parser.add_argument("--recall_min", type=float, default=RECALL_MIN)
    parser.add_argument("--decision", required=True)
    args = parser.parse_args()

    decision = decide(read_metrics(args.champion_metrics), read_metrics(args.challenger_metrics),
                      accuracy_min=args.accuracy_min, recall_min=args.recall_min)
    os.makedirs(args.decision, exist_ok=True)
    with open(os.path.join(args.decision, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))

    try:  # also visible in Azure ML studio, on the gate step's Metrics tab
        import mlflow
        mlflow.log_metric("promote", int(decision["promote"]))
    except Exception as exc:  # gate still works without MLflow
        print(f"(MLflow logging skipped: {exc})")


if __name__ == "__main__":
    main()
