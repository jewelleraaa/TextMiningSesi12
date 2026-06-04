import streamlit as st
import numpy as np
import json
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Analisis Sentimen LSTM", page_icon="📝", layout="centered")

# Fungsi untuk memuat Model dan Tokenizer secara efisien (Caching)
@st.cache_resource
def load_sentiment_artifacts():
    # Memuat model
    model = load_model('lstm_model.h5')
    
    # Memuat tokenizer
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    return model, tokenizer

try:
    model, tokenizer = load_sentiment_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat artefak model. Pastikan 'lstm_model.h5' dan 'tokenizer.json' berada di direktori yang sama. Error: {e}")

# Konstanta Preprocessing (Harus sama dengan saat training)
MAX_LEN = 50

# --- Tampilan UI Streamlit ---
st.title("WELCOME TO: 📝 Aplikasi Analisis Sentimen Teks")
st.subheader("Made by: Zahra Annisa A - 2702284086")
st.write("Masukkan teks atau ulasan di bawah ini untuk mendeteksi apakah sentimennya bernilai **Positif** atau **Negatif**.")

# Form Input Teks
user_input = st.text_area("Input Teks / Ulasan:", placeholder="Tulis ulasan Anda di sini... (contoh: barangnya bagus banget pelayanan cepat)")

if st.button("Analisis Sentimen"):
    if not model_loaded:
        st.warning("Aplikasi tidak dapat memproses karena model belum siap.")
    elif user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        # 1. PREPROCESSING (Wajib sama dengan train.py)
        text_clean = user_input.lower()
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_clean)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # 2. Tokenisasi & Padding
        sequences = tokenizer.texts_to_sequences([text_clean])
        padded = pad_sequences(sequences, maxlen=35, padding='post', truncating='post') # maxlen ganti ke 35
        
        # 3. Prediksi
        prediction = model.predict(padded)[0][0]
        
        # 4. Tampilkan Hasil
        st.write("---")
        st.markdown("### Hasil Analisis:")
        if prediction >= 0.5:
            st.success(f"**Sentimen: POSITIF** 🟢 (Skor Keyakinan: {prediction:.2%})")
        else:
            st.error(f"**Sentimen: NEGATIF** 🔴 (Skor Keyakinan: {(1 - prediction):.2%})")
            
# Informasi Tambahan di Sidebar
st.sidebar.title("Info Model")
st.sidebar.info(
    """
    - **Algoritma:** Long Short-Term Memory (LSTM)
    - **Framework:** TensorFlow / Keras & Streamlit
    - **Fitur:** Word Embedding dengan padding sekuensial.
    """
)