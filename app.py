import streamlit as st
import numpy as np
import re
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="⚡",
    layout="centered"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
    <style>
        body {
            background-color: #f2f4f7;
        }

        /* Center container */
        .main-container {
            max-width: 800px;
            margin: auto;
            padding: 20px 40px;
        }

        /* Title */
        .app-title {
            font-size: 45px;
            font-weight: 900;
            text-align: center;
            color: #1a1a1a;
            letter-spacing: -1px;
            margin-bottom: 5px;
        }

        .app-subtitle {
            font-size: 18px;
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }

        /* Card */
        .card {
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            margin-bottom: 25px;
        }

        /* Result box */
        .result-card {
            background: white;
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.07);
            border-left: 8px solid #1a73e8;
            margin-top: 20px;
        }

        .positive {
            font-weight: 700;
            color: #0f9d58;
            font-size: 22px;
        }

        .negative {
            font-weight: 700;
            color: #d93025;
            font-size: 22px;
        }

        /* Button */
        .stButton button {
            width: 100%;
            padding: 12px;
            background-color: #1a73e8 !important;
            color: white !important;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            transition: 0.2s;
        }

        .stButton button:hover {
            background-color: #1558b0 !important;
            transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)


# ===================== CONSTANTS =====================
MAX_WORDS = 10000
MAX_LEN = 200


# ===================== LOAD MODEL & TOKENIZER =====================
@st.cache_resource
def load_tokenizer_and_model():
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    model = load_model("sentiment_gru.h5")
    return tokenizer, model


tokenizer, model = load_tokenizer_and_model()


# ===================== CLEANING =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    return text


def preprocess_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded


# ===================== UI LAYOUT =====================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<div class='app-title'>Sentiment Analyzer ⚡</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Analyze customer reviews using an advanced GRU neural network</div>", unsafe_allow_html=True)


# ===================== INPUT CARD =====================
st.markdown("<div class='card'>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter your review:",
    height=150,
    placeholder="Example: The product quality was amazing and delivery was super fast! ❤️"
)

analyze_btn = st.button("Analyze Sentiment 🔍")

st.markdown("</div>", unsafe_allow_html=True)


# ===================== RESULT =====================
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a review first!")
    else:
        X_input = preprocess_text(user_input)
        prob = model.predict(X_input)[0][0]

        label = "Positive 😀" if prob >= 0.5 else "Negative 😞"
        sentiment_class = "positive" if prob >= 0.5 else "negative"

        st.markdown(f"""
            <div class="result-card">
                <h3 style='margin-bottom:10px;'>Prediction Result</h3>
                <p class="{sentiment_class}">{label}</p>
                <p style='font-size:17px; margin-top:-8px;'>
                    <strong>Confidence:</strong> {prob:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
