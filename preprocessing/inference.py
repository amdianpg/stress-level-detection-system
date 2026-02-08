# preprocessing/inference.py

import numpy as np
import pandas as pd
import pickle
import zipfile


WINDOW_SIZE = 3500
STEP_SIZE = WINDOW_SIZE // 2
REQUIRED_COLUMNS = ['ECG', 'EDA', 'ACC_X', 'ACC_Y', 'ACC_Z']

def load_csv(file):
    # =========================
    # CASE 1: FILE ZIP
    # =========================
    if file.name.endswith(".zip"):
        with zipfile.ZipFile(file) as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise ValueError("File ZIP tidak mengandung CSV")

            with z.open(csv_files[0]) as f:
                try:
                    df = pd.read_csv(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    df = pd.read_csv(f, encoding="latin1")

    # =========================
    # CASE 2: FILE CSV LANGSUNG
    # =========================
    else:
        file.seek(0)
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1")

    # =========================
    # VALIDASI STRUKTUR
    # =========================
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        raise ValueError(
            "Kolom CSV harus berisi: ECG, EDA, ACC_X, ACC_Y, ACC_Z"
        )

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Jumlah data ({len(df)}) tidak mencukupi untuk satu window ({WINDOW_SIZE})"
        )

    return df[REQUIRED_COLUMNS]


def create_windows(df):
    signals = df.values
    windows = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        window = signals[start:start + WINDOW_SIZE]
        windows.append(window)

    return np.array(windows)  # (n_window, 3500, 5)


def load_scaler(path):
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def normalize_windows(windows, scaler):
    n_window, time_steps, n_features = windows.shape

    reshaped = windows.reshape(-1, n_features)
    scaled = scaler.transform(reshaped)

    return scaled.reshape(n_window, time_steps, n_features)
