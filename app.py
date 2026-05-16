import streamlit as st
import os
import gdown
import joblib

# Konfigurasi File & ID Google Drive
MODEL_PATH = "indobert_model.pkl"

# MASUKKAN ID FILE YANG SUDAH ANDA SALIN PADA LANGKAH 1
GOOGLE_DRIVE_ID = "1X1_G56_gyC7m7dYL2nGetKdCXArYxnxM"

@st.cache_resource
def load_model_from_gdrive():
    # Periksa apakah file model sudah ada di server lokal Streamlit Cloud
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Sedang mengunduh model dari Google Drive (Proses ini hanya berjalan sekali saat startup)..."): url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Unduhan model berhasil selesai!")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None
            
    # Memuat model ke memori menggunakan joblib (sesuaikan jika menggunakan keras/torch)
    return joblib.load(MODEL_PATH)

# Memanggil fungsi pemuatan model
model = load_model_from_gdrive()

# --- BAGIAN ANTARMUKA UTAMA (UI) STREAMLIT ---
st.title("Aplikasi Analisis Sentimen Teks ")
st.write("Aplikasi ini menggunakan model machine learning seukuran 400 MB yang di-host via Google Drive.")

user_input = st.text_area("Masukkan Kalimat untuk Dianalisis:", "Saya sangat puas dengan performa aplikasi ini!")

if st.button("Analisis Sentimen"):
    if user_input:
        if model is not None:
            # Contoh logika prediksi (Sesuaikan dengan format preprocessing modelAnda)
            # prediksi = model.predict([user_input])[0]

            # Simulasi hasil prediksi demi demonstrasi antarmuka
            prediksi = "Positif"

            if prediksi == "Positif":
                st.success(f"Hasil Analisis: Sentimen **{prediksi}** ")
            else:
                st.error(f"Hasil Analisis: Sentimen **{prediksi}** ")
        else:
            st.error("Model gagal dimuat. Periksa kembali konfigurasi ID Google Drive Anda.")
else:
    st.warning("Silakan isi teks terlebih dahulu.")
