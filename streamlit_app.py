import os
import pickle
import joblib
import streamlit as st
from preprocessing import preprocess

# Konstanta
MODEL_DIR = "models"

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Analisis Sentimen", 
    page_icon="💬", 
    layout="centered"
)


@st.cache_resource
def load_ml():
    """Memuat model Machine Learning (Logistic Regression) dan TF-IDF Vectorizer."""
    model = joblib.load(os.path.join(MODEL_DIR, "ml_model.pkl"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return model, tfidf


@st.cache_resource
def load_dl():
    """Memuat model Deep Learning (LSTM) beserta Tokenizer dan Konfigurasinya."""
    import tensorflow as tf
    
    try:
        model_path = os.path.join(MODEL_DIR, "dl_model.h5")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            tok = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
            cfg = pickle.load(f)
        return model, tok, cfg, None
    except Exception as e:
        # Jika gagal load, kembalikan pesan errornya agar bisa kita baca di UI
        return None, None, None, str(e)


def predict_ml(text):
    """Prediksi menggunakan model Machine Learning."""
    model, tfidf = load_ml()
    vec = tfidf.transform([preprocess(text)])
    proba = float(model.predict_proba(vec)[0][1])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


def predict_dl(text):
    """Prediksi menggunakan model Deep Learning (LSTM) dengan deteksi error internal."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    model, tok, cfg, error_msg = load_dl()
    
    # Jika saat load model/pickle terjadi error, lemparkan ke luar
    if error_msg:
        raise RuntimeError(error_msg)
        
    seq = pad_sequences(
        tok.texts_to_sequences([preprocess(text)]),
        maxlen=cfg["MAX_LEN"], 
        padding="post", 
        truncating="post"
    )
    proba = float(model.predict(seq, verbose=0)[0][0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


# --- UI INTERFACE ---

# Bagian Header & Identitas Mahasiswa
st.title("💬 Analisis Sentimen")

# Kotak Informasi Identitas
st.info(f"""
👤 **Nama:** Zahra Annisa  
🆔 **NIM:** 2702284086  
""")

st.caption("Klasifikasi teks menjadi sentimen positif atau negatif secara otomatis.")
st.write("---")

# Input Pilihan Model
model_choice = st.radio(
    "Pilih Model Analisis:",
    ["Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"],
    horizontal=False,
)

# Input Teks dari Pengguna
text = st.text_area(
    "Masukkan teks yang ingin dianalisis:", 
    height=140,
    placeholder="Contoh: Barangnya bagus banget, pengirimannya super cepat..."
)

# Tombol Aksi
if st.button("Analisis Sentimen", type="primary"):
    if not text.strip():
        st.warning("Teks tidak boleh kosong. Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses teks..."):
            try:
                if model_choice.startswith("Deep"):
                    label, proba = predict_dl(text)
                else:
                    label, proba = predict_ml(text)
                
                # Menghitung nilai confidence score
                conf = proba if label == "Positif" else 1 - proba
                
                # Menampilkan Hasil Utama
                st.write("### Hasil Analisis:")
                if label == "Positif":
                    st.success(f"😊 **Sentimen Terdeteksi: {label}**")
                else:
                    st.error(f"😞 **Sentimen Terdeteksi: {label}**")

                # Menampilkan Metrik dan Visualisasi Probabilitas
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Confidence Score", value=f"{conf * 100:.1f}%")
                with col2:
                    st.caption(f"Probabilitas ke arah Positif: P(positif) = {proba:.4f}")
                    st.progress(proba)
                    
            except Exception as e:
                st.error("🚨 **Terjadi kesalahan internal pada Model LSTM:**")
                st.code(str(e), language="text")
                st.warning("Catatan: Jika errornya berisi 'Magic number', artinya file .pkl kamu harus di-export ulang menggunakan versi python yang sama dengan server.")