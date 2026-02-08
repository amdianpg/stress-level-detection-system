# Stress Level Detection System

This repository contains a stress level detection system based on
physiological signals using a CNN–LSTM model with an attention mechanism.

The system classifies stress into three levels:
Low Stress, Medium Stress, and High Stress.

## Model Overview
The model was trained using the WESAD dataset and utilizes:
- ECG (Electrocardiogram)
- EDA (Electrodermal Activity)
- ACC (Accelerometer)

Signals are segmented into fixed-length windows and normalized using
a pre-fitted scaler before inference.

## How to Run the Application

1. Install dependencies:
pip install -r requirements.txt

2. Run the Streamlit app:
streamlit run app.py

## Input Format
The system accepts CSV files containing physiological signals with
the following columns:

- ECG
- EDA
- ACC_X
- ACC_Y
- ACC_Z

Each row represents one time step.

## Output
The system outputs:
- Dominant stress level
- Distribution of window-based predictions

## Notes
- The pretrained model (`.h5`) and scaler (`.pkl`) are included to allow
  direct inference without retraining.
- This repository focuses on inference and system deployment, not training.


