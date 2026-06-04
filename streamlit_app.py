# streamlit_app.py
import os
import json
import time
import streamlit as st

from app import (
    ARTIFACT_DIR, MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH,
    DEFAULT_CONFIG, train_and_save, load_artifacts, predict_label
)

st.set_page_config(page_title="Indonesian Sentiment LSTM", page_icon="💬", layout="centered")

st.title("💬 Welcome to Sentiment Analysis")

with st.expander("About this app", expanded=False):
    st.write(
        "- Trains a Bidirectional LSTM on your dataset (data/dataset.csv). "
        "- After training, type any review/sentence to get a sentiment prediction."
    )

# Sidebar: training / config
st.sidebar.header("Training")
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load or edit config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = DEFAULT_CONFIG.copy()

with st.sidebar.form("train_form"):
    st.caption("Model hyperparameters")
    max_words = st.number_input("Max words (vocab size)", 1000, 100000, cfg["max_words"], step=1000)
    max_len = st.number_input("Max sequence length", 10, 300, cfg["max_len"], step=5)
    embedding_dim = st.number_input("Embedding dim", 16, 512, cfg["embedding_dim"], step=16)
    lstm_units = st.number_input("LSTM units", 16, 512, cfg["lstm_units"], step=16)
    dropout = st.slider("Dropout", 0.0, 0.8, float(cfg["dropout"]), 0.05)
    batch_size = st.number_input("Batch size", 8, 256, cfg["batch_size"], step=8)
    epochs = st.number_input("Epochs", 1, 50, cfg["epochs"])
    val_split = st.slider("Validation split", 0.05, 0.3, float(cfg["validation_split"]), 0.01)
    patience = st.number_input("Early stopping patience", 1, 10, cfg["patience"])
    submitted = st.form_submit_button("Train / Retrain")

if submitted:
    new_cfg = {
        "max_words": int(max_words),
        "max_len": int(max_len),
        "embedding_dim": int(embedding_dim),
        "lstm_units": int(lstm_units),
        "dropout": float(dropout),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "validation_split": float(val_split),
        "patience": int(patience),
    }
    with st.spinner("Training LSTM on data/dataset.csv..."):
        summary = train_and_save(new_cfg)
    st.success("Training complete.")
    st.json(summary)

# Inference section
st.header("Try it")
if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)):
    st.warning("Model not found. Please train the model from the sidebar first.")
else:
    # Lazy load artifacts to confirm available
    _ = load_artifacts()

    default_example = "makanannya enak porsinya banyak tempatnya bersih pelayanan cepat"
    text = st.text_area("Enter text (Indonesian):", value=default_example, height=120)
    if st.button("Predict"):
        if text.strip():
            with st.spinner("Predicting..."):
                res = predict_label(text.strip())
                time.sleep(0.2)
            label = "Positive 👍" if res["label"] == 1 else "Negative 👎"
            st.subheader(label)
            st.write(f"Confidence (positive): {res['proba_positive']:.4f}")

            st.progress(min(max(res["proba_positive"], 0.0), 1.0))
        else:
            st.info("Please enter some text.")

st.markdown("---")
st.caption("Built with Keras, TensorFlow, and Streamlit.")


