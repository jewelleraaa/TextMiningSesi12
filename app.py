"""
Flask App - Sentiment Analysis
==============================
Memuat model ML (.pkl) dan DL (.h5) hasil train.py, menyediakan UI web.

Jalankan:
    python app.py
Buka: http://127.0.0.1:5000
"""
import os
import pickle

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

from preprocessing import preprocess

app = Flask(__name__)
MODEL_DIR = "models"

# --- Load artifact ML ---
ml_model = joblib.load(os.path.join(MODEL_DIR, "ml_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# --- Load artifact DL (lazy import tensorflow agar startup ML tetap cepat) ---
_dl_model = None
_tokenizer = None
_config = None


def load_dl():
    global _dl_model, _tokenizer, _config
    if _dl_model is None:
        from tensorflow.keras.models import load_model
        _dl_model = load_model(os.path.join(MODEL_DIR, "dl_model.h5"))
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            _tokenizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
            _config = pickle.load(f)
    return _dl_model, _tokenizer, _config


def predict_ml(text):
    clean = preprocess(text)
    vec = tfidf.transform([clean])
    proba = float(ml_model.predict_proba(vec)[0][1])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


def predict_dl(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tok, cfg = load_dl()
    clean = preprocess(text)
    seq = pad_sequences(tok.texts_to_sequences([clean]),
                        maxlen=cfg["MAX_LEN"], padding="post", truncating="post")
    proba = float(model.predict(seq, verbose=0)[0][0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    model_type = data.get("model", "ml")

    if not text:
        return jsonify({"error": "Teks kosong"}), 400

    if model_type == "dl":
        label, proba = predict_dl(text)
    else:
        label, proba = predict_ml(text)

    return jsonify({
        "model": model_type.upper(),
        "label": label,
        "confidence": round(proba if label == "Positif" else 1 - proba, 4),
        "proba_positif": round(proba, 4),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
