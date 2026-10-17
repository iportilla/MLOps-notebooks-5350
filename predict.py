"""
Use a saved model to classify a message.

    python predict.py "Your message here"                 # uses models/production.pkl
    python predict.py "Your message here" models/model_v2.pkl
"""

import sys

import joblib
import pandas as pd

from common import extract_features


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py "message" [model.pkl]')
        sys.exit(1)
    text = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/production.pkl"

    model = joblib.load(model_path)
    features = extract_features(text)
    prediction = model.predict(pd.DataFrame([features]))[0]

    print(f"Model      : {model_path}")
    print(f"Message    : {text}")
    print(f"Features   : {features}")
    print(f"Prediction : {prediction.upper()}")


if __name__ == "__main__":
    main()
