"""
Monitoring: does new data still look like our training data?

    python check_drift.py <new_data.csv>

For each feature we compare the average value in the new data with the
average in the training data. A big change (more than 25%) = DRIFT.
If the new data has labels, we also check how the production model does on it.
"""

import sys

import joblib

from common import FEATURES, evaluate, load_data, print_metrics

TRAINING_DATA = "data/balanced.csv"
PRODUCTION_MODEL = "models/production.pkl"
DRIFT_THRESHOLD = 0.25  # 25% change in the average


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_drift.py <new_data.csv>")
        sys.exit(1)
    reference = load_data(TRAINING_DATA)
    new = load_data(sys.argv[1])

    print(f"\nDrift check: {sys.argv[1]} vs {TRAINING_DATA}")
    print("  We look at SPAM messages only (that is what spammers change).")
    ref_spam = reference[reference["label"] == "spam"]
    new_spam = new[new["label"] == "spam"]

    drift_found = False
    for feature in FEATURES:
        before, after = ref_spam[feature].mean(), new_spam[feature].mean()
        change = (after - before) / before
        flag = "DRIFT" if abs(change) > DRIFT_THRESHOLD else "ok"
        drift_found = drift_found or flag == "DRIFT"
        print(f"  {feature:<7} training avg {before:6.1f} | new avg {after:6.1f} | change {change:+.0%}  -> {flag}")

    print("\nHow does the production model do on this new data?")
    print_metrics(evaluate(joblib.load(PRODUCTION_MODEL), new))

    if drift_found:
        print("\nALERT: drift detected. Time to collect new labels and retrain.")
    else:
        print("\nNo drift detected.")


if __name__ == "__main__":
    main()
