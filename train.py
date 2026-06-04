import pandas as pd
import numpy as np
import json
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. LOAD & CLEAN DATASET
# ==========================================
df = pd.read_csv('data.csv')

# Fungsi membersihkan teks dari tanda baca, angka, dan spasi berlebih
def clean_text(text):
    text = str(text).lower() # Ubah ke huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus angka dan tanda baca
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

df['text_clean'] = df['text'].apply(clean_text)

texts = df['text_clean'].values
labels = df['label'].values

# ==========================================
# 2. PARAMETER TEXT MINING (DIOPTIMALKAN)
# ==========================================
max_words = 3000  # Dikurangi agar model fokus pada kata yang sering muncul
max_len = 35      # Disesuaikan dengan rata-rata panjang kalimat ulasan Anda
embedding_dim = 64 # Dimensi lebih kecil untuk mencegah overfitting

# ==========================================
# 3. TOKENISASI & PADDING
# ==========================================
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Splitting Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# ==========================================
# 4. ARSITEKTUR LSTM YANG LEBIH AKURAT
# ==========================================
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim),
    SpatialDropout1D(0.3), # Mencegah overfitting pada embedding kata
    LSTM(32, dropout=0.2, recurrent_dropout=0.2), # Mengurangi unit ke 32 agar lebih general
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # Output biner (0-1)
])

# Menggunakan optimizer Adam dengan learning rate standard
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ==========================================
# 5. PROSES TRAINING DENGAN MONITORING
# ==========================================
# Early stopping akan menghentikan training jika akurasi data validasi tidak naik lagi
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nMemulai pelatihan model...")
history = model.fit(
    X_train, y_train,
    epochs=15,          # Meningkatkan epoch karena ada Early Stopping
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Evaluasi Akhir
loss, accuracy = model.evaluate(X_test, y_test)
print('\n=== HASIL EVALUASI MODEL ===')
print(f"Akurasi Akhir pada Data Pengujian: {accuracy*100:.2f}%")

# ==========================================
# 6. SIMPAN ARTEFAK
# ==========================================
model.save('lstm_model.h5')

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("Model dan Tokenizer versi baru berhasil disimpan!")