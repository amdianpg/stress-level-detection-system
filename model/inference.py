# model/inference.py

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Attention,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

LABEL_MAP = {
    0: "Low Stress",
    1: "Medium Stress",
    2: "High Stress"
}

def build_cnn_lstm_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(32, 5, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)

    x = LSTM(64, return_sequences=True)(x)

    attention = Attention()([x, x])
    x = GlobalAveragePooling1D()(attention)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def load_trained_model(weight_path):
    model = build_cnn_lstm_attention(
        input_shape=(3500, 5),
        num_classes=3
    )

    model.load_weights(weight_path)
    return model

def predict_windows(model, X):
    """
    X shape: (n_window, 3500, 5)
    """
    preds = model.predict(X)
    return preds


def aggregate_prediction(preds):
    labels = np.argmax(preds, axis=1)
    counts = np.bincount(labels, minlength=3)
    final_label = np.argmax(counts)

    return LABEL_MAP[final_label], counts
