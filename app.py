"""
Spam Classifier - Streamlit App
--------------------------------
Loads spam_pipeline.pkl (a scikit-learn Pipeline: CountVectorizer -> RandomForestClassifier)
and serves a simple web UI for classifying text as spam / not spam.

Run:
    pip install streamlit joblib scikit-learn
    streamlit run streamlit_app.py
"""

import re
import string
import sys

import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# IMPORTANT: this function must be defined here, under this exact name,
# because the pickled CountVectorizer stores its `preprocessor` as a
# reference to `__main__.wordopt`. If this function is missing (or its
# logic differs from training time), predictions won't match what the
# model actually learned.
# ---------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Make wordopt visible as __main__.wordopt for unpickling.
sys.modules["__main__"].wordopt = wordopt

MODEL_PATH = "spam_pipeline.pkl"
LABELS = {0: "Not Spam", 1: "Spam"}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.set_page_config(page_title="Spam Classifier", page_icon="✉️")
st.title("✉️ Spam Classifier")
st.write("Paste a message below and classify it as spam or not spam.")

text = st.text_area("Message", height=160, placeholder="Paste a message to classify...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        pred = int(pipeline.predict([text])[0])
        proba = pipeline.predict_proba([text])[0]
        label_name = LABELS.get(pred, str(pred))
        confidence = float(proba[pred])

        if pred == 1:
            st.error(f"**{label_name}**  ·  Confidence: {confidence * 100:.1f}%")
        else:
            st.success(f"**{label_name}**  ·  Confidence: {confidence * 100:.1f}%")

        with st.expander("Full probability breakdown"):
            for i, p in enumerate(proba):
                st.write(f"{LABELS.get(i, str(i))}: {p * 100:.2f}%")