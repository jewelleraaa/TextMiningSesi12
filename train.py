"""
Sentiment Analysis - Training Script
=====================================
Pipeline lengkap:
  1. Load data
  2. Cleaning data
  3. Preprocessing (case folding, tokenizing, stopword removal, stemming)
  4. Text representation (TF-IDF untuk ML, sequence/embedding untuk DL)
  5. Split data
  6. Train: 1 model Machine Learning (Logistic Regression) + 1 model Deep Learning (LSTM)
  7. Extract / export model (joblib .pkl + Keras .h5)

Jalankan: python train.py
Output ke folder ./models
"""

import os
import re
import string
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# --- Stopword & Stemmer (Bahasa Indonesia) ---
# Pakai Sastrawi jika tersedia, kalau tidak fallback ke list manual.
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _stemmer = StemmerFactory().create_stemmer()
    _stopwords = set(StopWordRemoverFactory().get_stop_words())
    USE_SASTRAWI = True
except Exception:
    USE_SASTRAWI = False
    _stemmer = None
    _stopwords = set("""
        yang di ke dari dan atau ini itu untuk dengan pada adalah ada akan
        saya kamu dia kami kita mereka aku anda nya tak tidak bukan juga saja
        sudah belum masih bisa harus mau ingin agar supaya karena sebab maka
        sehingga namun tetapi tapi sangat lebih paling akan telah pun lah kah
    """.split())

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(path=None):
    """
    Format dataset: kolom 'text' dan 'label' (0 = negatif, 1 = positif).
    Jika path None / file tidak ada, pakai dummy dataset untuk demo.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        assert "text" in df.columns and "label" in df.columns, \
            "CSV harus punya kolom 'text' dan 'label'"
        return df[["text", "label"]].dropna()

    print("[!] Dataset tidak ditemukan, memakai dummy dataset demo.")
    pos = [
        "barang bagus banget kualitas mantap recommended seller",
        "pelayanan ramah pengiriman cepat saya puas sekali",
        "produk sesuai deskripsi harga murah worth it",
        "suka banget sama kualitasnya keren dan awet",
        "mantap jiwa cepat sampai packing rapi terima kasih",
        "aplikasinya enak dipakai fiturnya lengkap dan membantu",
        "makanannya enak porsinya banyak tempatnya nyaman",
        "filmnya seru alurnya bagus aktingnya memukau",
    ] * 30
    neg = [
        "barang jelek tidak sesuai gambar mengecewakan sekali",
        "pengiriman lama banget pelayanan buruk tidak ramah",
        "produk rusak pas datang kualitas murahan kecewa",
        "aplikasinya lemot sering error bikin kesal",
        "makanannya hambar mahal pelayanan lambat",
        "filmnya membosankan jalan cerita berantakan tidak rekomen",
        "barang palsu tidak original parah banget penipuan",
        "kecewa berat produk cacat tidak bisa dipakai",
    ] * 30
    df = pd.DataFrame({
        "text": pos + neg,
        "label": [1] * len(pos) + [0] * len(neg),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================
# 2 & 3. CLEANING + PREPROCESSING
# ============================================================
def clean_text(text):
    text = str(text).lower()                       # case folding
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # hapus URL
    text = re.sub(r"@\w+|#\w+", " ", text)         # hapus mention/hashtag
    text = re.sub(r"\d+", " ", text)               # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(r"(.)\1{2,}", r"\1", text)       # normalisasi huruf berulang (mantaaap->mantap)
    text = re.sub(r"\s+", " ", text).strip()       # hapus spasi berlebih
    return text


def preprocess(text):
    text = clean_text(text)
    tokens = text.split()                          # tokenizing
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]  # stopword removal
    if USE_SASTRAWI:
        tokens = [_stemmer.stem(t) for t in tokens]  # stemming
    return " ".join(tokens)


# ============================================================
# MAIN
# ============================================================
def main(data_path=None):
    print("=" * 50)
    print("1. Load data")
    df = load_data(data_path)
    print(f"   total data: {len(df)} | distribusi label:\n{df['label'].value_counts()}")

    print("2-3. Cleaning + Preprocessing")
    df["clean"] = df["text"].apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    X_text = df["clean"].values
    y = df["label"].values

    print("5. Split data (80/20)")
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========================================================
    # MODEL 1: MACHINE LEARNING (TF-IDF + Logistic Regression)
    # ========================================================
    print("\n[ML] 4. Text representation: TF-IDF")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_txt)
    X_test_tfidf = tfidf.transform(X_test_txt)

    print("[ML] 6. Train Logistic Regression")
    ml_model = LogisticRegression(max_iter=1000, C=1.0)
    ml_model.fit(X_train_tfidf, y_train)

    ml_pred = ml_model.predict(X_test_tfidf)
    print(f"[ML] Akurasi: {accuracy_score(y_test, ml_pred):.4f}")
    print(classification_report(y_test, ml_pred, target_names=["negatif", "positif"]))

    print("[ML] 7. Extract model -> joblib")
    joblib.dump(ml_model, os.path.join(MODEL_DIR, "ml_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    # ========================================================
    # MODEL 2: DEEP LEARNING (Embedding + LSTM)
    # ========================================================
    print("\n[DL] 4. Text representation: tokenizer + padding")
    MAX_WORDS = 5000
    MAX_LEN = 30
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_txt)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_txt),
                                maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_txt),
                               maxlen=MAX_LEN, padding="post", truncating="post")

    print("[DL] 6. Train LSTM")
    dl_model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=64),
        LSTM(64),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    dl_model.compile(loss="binary_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     metrics=["accuracy"])
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                             restore_best_weights=True)
    dl_model.fit(X_train_seq, y_train, validation_split=0.1,
                 epochs=15, batch_size=32, callbacks=[early], verbose=2)

    dl_eval = dl_model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"[DL] Akurasi: {dl_eval[1]:.4f}")

    print("[DL] 7. Extract model -> .h5 + tokenizer")
    dl_model.save(os.path.join(MODEL_DIR, "dl_model.h5"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # simpan config
    with open(os.path.join(MODEL_DIR, "config.pkl"), "wb") as f:
        pickle.dump({"MAX_LEN": MAX_LEN, "MAX_WORDS": MAX_WORDS}, f)

    print("\nSelesai. Semua artifact tersimpan di ./models")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
