"""
A tiny web app that serves the production model.

    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from common import extract_features

st.title("SMS Spam Classifier")
model = joblib.load("models/production.pkl")

message = st.text_area("Message", "WINNER!! Claim your free prize now!!!")
if st.button("Classify"):
    features = extract_features(message)
    prediction = model.predict(pd.DataFrame([features]))[0]
    st.write("Features:", features)
    if prediction == "spam":
        st.error("SPAM")
    else:
        st.success("HAM (not spam)")
