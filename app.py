import streamlit as st
import numpy as np

from preprocessing.inference import (
    load_csv,
    create_windows,
    load_scaler,
    normalize_windows
)

from model.inference import (
    load_trained_model,
    predict_windows,
    aggregate_prediction
)

# =========================
# KONFIGURASI
# =========================
MODEL_PATH = "model/cnn_lstm_attention.h5"
SCALER_PATH = "preprocessing/scaler.pkl"

st.set_page_config(page_title="Stress Level Detection", layout="centered")

st.title("Deteksi Tingkat Stres Berbasis Sinyal Fisiologis")
st.write(
    "Sistem ini mengklasifikasikan tingkat stres "
    "berdasarkan sinyal ECG, EDA, dan ACC "
    "menggunakan model CNN–LSTM dengan attention mechanism."
)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "Unggah file CSV atau ZIP sinyal fisiologis",
    type=["csv", "zip"]
)

if uploaded_file is not None:
    try:
        # =========================
        # PREPROCESSING
        # =========================
        df = load_csv(uploaded_file)
        windows = create_windows(df)

        scaler = load_scaler(SCALER_PATH)
        X_input = normalize_windows(windows, scaler)

        st.success(f"Data berhasil diproses ({X_input.shape[0]} window)")

        # =========================
        # LOAD MODEL
        # =========================
        model = load_trained_model(MODEL_PATH)

        # =========================
        # PREDIKSI
        # =========================
        if st.button("Prediksi Tingkat Stres"):
            preds = predict_windows(model, X_input)
            final_label, counts = aggregate_prediction(preds)

            # =========================
            # OUTPUT
            # =========================
            st.subheader("Hasil Prediksi")
            st.write(f"**Tingkat Stres Dominan:** {final_label}")

            st.write("Distribusi prediksi window:")
            st.write(
                {
                    "Low Stress": int(counts[0]),
                    "Medium Stress": int(counts[1]),
                    "High Stress": int(counts[2]),
                }
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
