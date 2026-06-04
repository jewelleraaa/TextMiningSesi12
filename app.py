# app.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")
LABEL_INDEX_PATH = os.path.join(ARTIFACT_DIR, "label_index.json")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.h5")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "config.json")

# Default config (you can tune in requirements of your machine)
DEFAULT_CONFIG = {
    "max_words": 20000,     # vocab size
    "max_len": 40,          # sequence length
    "embedding_dim": 64,
    "lstm_units": 64,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 6,
    "validation_split": 0.1,
    "patience": 2
}

def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_dataset(path: str = DATA_PATH) -> Tuple[pd.Series, pd.Series]:
    # Expecting CSV with headers: text,label
    df = pd.read_csv(path)
    # Tolerate various headers by normalizing
    if "text" not in df.columns or "label" not in df.columns:
        # Try to infer from first two columns
        df.columns = [c.strip().lower() for c in df.columns]
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    # Cast to int labels if needed
    df["label"] = df["label"].astype(int)
    return df["text"], df["label"]

def build_tokenizer(texts: pd.Series, max_words: int) -> Tokenizer:
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(texts.tolist())
    return tok

def texts_to_padded(tok: Tokenizer, texts, max_len: int) -> np.ndarray:
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

def build_model(config: Dict[str, Any]) -> tf.keras.Model:
    inputs = layers.Input(shape=(config["max_len"],), dtype="int32")
    x = layers.Embedding(config["max_words"], config["embedding_dim"])(inputs)
    x = layers.SpatialDropout1D(config["dropout"])(x)
    x = layers.Bidirectional(layers.LSTM(config["lstm_units"], return_sequences=False))(x)
    x = layers.Dropout(config["dropout"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_and_save(config: Dict[str, Any] = None) -> Dict[str, Any]:
    ensure_dirs()
    if config is None:
        config = DEFAULT_CONFIG

    texts, labels = load_dataset(DATA_PATH)
    tok = build_tokenizer(texts, config["max_words"])
    X = texts_to_padded(tok, texts, config["max_len"])
    y = labels.values.astype("float32")

    model = build_model(config)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True)
    ]
    history = model.fit(
        X, y,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=config["validation_split"],
        callbacks=cbs,
        verbose=2
    )

    # Save artifacts
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tok, f)
    with open(LABEL_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"negative": 0, "positive": 1}, f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f)
    model.save(MODEL_PATH)

    return {
        "train_size": len(texts),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

def load_artifacts():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tok = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return config, tok, model

def predict_proba(texts_list):
    config, tok, model = load_artifacts()
    X = texts_to_padded(tok, texts_list, config["max_len"])
    probs = model.predict(X, verbose=0).flatten()
    return probs  # probability of positive

def predict_label(text: str) -> Dict[str, Any]:
    prob = float(predict_proba([text])[0])
    label = 1 if prob >= 0.5 else 0
    return {"text": text, "proba_positive": prob, "label": label}

if __name__ == "__main__":
    # Optional: CLI training when running `python app.py`
    summary = train_and_save(DEFAULT_CONFIG)
    print("Training done:", summary)
