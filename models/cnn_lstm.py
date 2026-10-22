import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPooling1D, LSTM, Dropout, Dense

def build_cnn_lstm_model(sequence_length, num_features, num_classes, filters=64, lstm_units=64):
    """
    Build 1D CNN + LSTM Deep Learning model for machine fault classification.
    
    Architecture:
    Input (sequence_length, num_features)
    → Conv1D(filters=32, kernel_size=3, padding='same')
    → BatchNormalization
    → ReLU
    → MaxPooling1D(pool_size=2)
    → Conv1D(filters=filters, kernel_size=3, padding='same')
    → BatchNormalization
    → ReLU
    → LSTM(lstm_units, return_sequences=False)
    → Dropout(0.3)
    → Dense(64, activation='relu')
    → Dense(num_classes, activation='softmax')
    """
    model = Sequential([
        # First 1D Conv Block
        Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(sequence_length, num_features)),
        BatchNormalization(),
        ReLU(),
        MaxPooling1D(pool_size=2),

        # Second 1D Conv Block
        Conv1D(filters=filters, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),

        # LSTM Layer for temporal feature extraction
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.3),

        # Dense Classifier Block
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
