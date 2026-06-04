# Sentiment Analysis — ML + DL, Flask + Streamlit

Pipeline lengkap: cleaning → preprocessing (stopword removal + stemming) → text representation → split → train ML (Logistic Regression) & DL (LSTM) → extract model → deploy.

## Struktur
```
sentiment-project/
├── train.py              # training: hasilkan artifact ke models/
├── preprocessing.py      # cleaning/preprocessing bersama (train + inference)
├── app.py                # Flask + UI (templates/index.html)
├── templates/index.html  # UI Flask
├── streamlit_app.py      # untuk deploy di Streamlit
├── requirements.txt
└── models/               # dibuat oleh train.py
    ├── ml_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── dl_model.h5
    ├── tokenizer.pkl
    └── config.pkl
```

## 1. Install
```bash
pip install -r requirements.txt
```

## 2. Training
Pakai dataset sendiri (CSV dengan kolom `text` dan `label`; 0=negatif, 1=positif):
```bash
python train.py data.csv
```
Tanpa argumen akan memakai dummy dataset demo:
```bash
python train.py
```

## 3. Jalankan Flask (lokal)
```bash
python app.py
# buka http://127.0.0.1:5000
```

## 4. Deploy Streamlit
```bash
streamlit run streamlit_app.py        # lokal
```
Untuk Streamlit Cloud: push folder ini ke GitHub **termasuk folder `models/`**, lalu di share.streamlit.io pilih repo dan set main file `streamlit_app.py`.

## Catatan
- Stopword & stemming Bahasa Indonesia memakai **Sastrawi** (otomatis fallback ke list manual bila tidak terinstall).
- Akurasi pada dummy dataset rendah karena data sintetis kecil; ganti dengan dataset asli untuk hasil nyata.
- File `dl_model.h5` cukup besar; jika repo GitHub bermasalah dengan ukuran, gunakan Git LFS atau format `.keras`.
