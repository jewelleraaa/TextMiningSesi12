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
    """Memuat model Deep Learning (LSTM) dengan membersihkan semua bug Keras 3 secara global."""
    import tensorflow as tf
    import h5py
    import json
    
    try:
        model_path = os.path.join(MODEL_DIR, "dl_model.h5")
        
        # --- GLOBAL PATCH UNTUK KORUP CONFIG KERAS 3 ---
        def clean_quantization_config(config):
            """Menghapus parameter quantization_config di semua layer secara rekursif."""
            if isinstance(config, dict):
                config.pop('quantization_config', None)
                for key, value in config.items():
                    clean_quantization_config(value)
            elif isinstance(config, list):
                for item in config:
                    clean_quantization_config(item)
            return config

        # Buka file .h5 secara manual untuk mengekstrak dan memperbaiki arsitektur model
        with h5py.File(model_path, 'r') as f:
            model_config_raw = f.attrs.get('model_config')
            if model_config_raw is None:
                raise ValueError("Format file .h5 tidak mengenali metadata model_config.")
            
            if isinstance(model_config_raw, bytes):
                model_config_raw = model_config_raw.decode('utf-8')
                
            model_config = json.loads(model_config_raw)
            
            # Bersihkan parameter jahat dari semua layer secara massal
            cleaned_config = clean_quantization_config(model_config)
            
            # FIX AKSES: Menggunakan tf.keras.models langsung atau saving.deserialize_keras_object
            if hasattr(tf.keras.models, 'model_from_config'):
                model = tf.keras.models.model_from_config(cleaned_config)
            elif hasattr(tf.keras.saving, 'deserialize_keras_object'):
                model = tf.keras.saving.deserialize_keras_object(cleaned_config)
            else:
                # Fallback terakhir jika versi tensor berubah lagi
                from tensorflow.keras.layers import deserialize
                model = tf.keras.models.Sequential.from_config(cleaned_config)
            
            # Salin bobot (weights) asli dari file .h5 ke dalam arsitektur baru
            for layer in model.layers:
                layer_name = layer.name
                if f"model_weights/{layer_name}" in f:
                    weight_names = f[f"model_weights/{layer_name}"].attrs.get('weight_names')
                    weights = []
                    for weight_name in weight_names:
                        if isinstance(weight_name, bytes):
                            weight_name = weight_name.decode('utf-8')
                        weights.append(f[f"model_weights/{layer_name}/{weight_name}"][()])
                    layer.set_weights(weights)

        # Load Tokenizer dan Config pendukung
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            tok = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
            cfg = pickle.load(f)
            
        return model, tok, cfg, None
    except Exception as e:
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