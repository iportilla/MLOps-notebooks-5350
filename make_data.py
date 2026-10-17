"""
Create the synthetic datasets used in this lab.

    python make_data.py

Files created in data/:
    balanced.csv        1,000 ham + 1,000 spam  (our main training data)
    test_set.csv        a FIXED test set, 85% ham / 15% spam (like real life)
    new_week_ok.csv     a new batch of data that looks normal
    new_week_bad.csv    a new batch with a labeling bug (spam labeled as ham)
    new_week_drift.csv  a new batch where spammers send SHORTER messages

The real (imbalanced) dataset, data/sms_real_1k.csv, is already provided.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def make_messages(n, label, length_mean, length_sd, punct_mean, punct_sd):
    """Generate n fake messages described only by their two features."""
    length = np.maximum(1, rng.normal(length_mean, length_sd, n).round()).astype(int)
    punct = np.maximum(0, rng.normal(punct_mean, punct_sd, n).round()).astype(int)
    return pd.DataFrame({"label": label, "length": length, "punct": punct})


def ham(n):
    return make_messages(n, "ham", 50, 20, 3, 2)


def spam(n):
    return make_messages(n, "spam", 150, 30, 12, 5)


def save(df, name):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv(f"data/{name}", index=False)
    counts = df["label"].value_counts().to_dict()
    print(f"  data/{name:<20} {len(df):>5} rows  {counts}")


if __name__ == "__main__":
    print("Creating datasets...")
    save(pd.concat([ham(1000), spam(1000)]), "balanced.csv")
    save(pd.concat([ham(425), spam(75)]), "test_set.csv")
    save(pd.concat([ham(250), spam(250)]), "new_week_ok.csv")

    # Labeling bug: an upstream job marked most spam as "ham"
    bad_spam = spam(1500).assign(label="ham")
    save(pd.concat([ham(250), bad_spam, spam(50)]), "new_week_bad.csv")

    # Drift: spammers change tactics and send short messages
    short_spam = make_messages(250, "spam", 45, 15, 6, 3)
    save(pd.concat([ham(250), short_spam]), "new_week_drift.csv")
    print("Done.")
