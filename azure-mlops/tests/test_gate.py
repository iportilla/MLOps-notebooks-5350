"""Unit tests for the quality gate. Run with: python -m pytest -q tests"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gate import decide, passes_gate  # noqa: E402

# Numbers from the local lab (train_model.ipynb), on data/test_set.csv
V1_IMBALANCED = {"accuracy": 0.856, "precision": 1.0, "recall": 0.04, "f1": 0.077}
V2_BALANCED = {"accuracy": 0.990, "precision": 0.938, "recall": 1.0, "f1": 0.968}
BAD_WEEK = {"accuracy": 0.868, "precision": 1.0, "recall": 0.12, "f1": 0.214}


def test_gate_rejects_high_accuracy_low_recall():
    assert not passes_gate(V1_IMBALANCED)          # the accuracy trap


def test_gate_accepts_balanced_model():
    assert passes_gate(V2_BALANCED)


def test_challenger_with_low_recall_is_not_promoted():
    decision = decide(champion=V2_BALANCED, challenger=V1_IMBALANCED)
    assert decision["promote"] is False
    assert any("recall" in reason for reason in decision["reasons"])


def test_bad_data_week_is_blocked():
    assert decide(champion=V2_BALANCED, challenger=BAD_WEEK)["promote"] is False


def test_better_challenger_is_promoted():
    assert decide(champion=V1_IMBALANCED, challenger=V2_BALANCED)["promote"] is True


def test_precision_drop_alone_does_not_block():
    # v2's precision is lower than v1's 100%, but precision is not a no-regression metric
    assert decide(champion=V1_IMBALANCED, challenger=V2_BALANCED)["promote"] is True


def test_small_regression_within_tolerance_is_allowed():
    slightly_worse = dict(V2_BALANCED, recall=0.995, f1=0.960)
    assert decide(champion=V2_BALANCED, challenger=slightly_worse)["promote"] is True


def test_regression_beyond_tolerance_is_blocked():
    worse = dict(V2_BALANCED, recall=0.95)
    assert decide(champion=V2_BALANCED, challenger=worse)["promote"] is False


def test_decision_is_json_serialisable():
    json.dumps(decide(V2_BALANCED, V1_IMBALANCED))
