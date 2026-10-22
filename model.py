import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_cnn_lstm_model(sequence_length, num_features, num_classes, learning_rate=0.001):
    """
    Build 1D CNN + LSTM Deep Learning Model for equipment condition fault prediction.
    Dynamically adapts input shape (sequence_length, num_features) and softmax output layer (num_classes).
    
    Architecture:
    Input (sequence_length, num_features)
    ↓ Conv1D (32 filters, kernel_size=3, padding='same')
    ↓ BatchNormalization
    ↓ ReLU
    ↓ MaxPooling1D (pool_size=2)
    ↓ Conv1D (64 filters, kernel_size=3, padding='same')
    ↓ BatchNormalization
    ↓ ReLU
    ↓ LSTM (64 units)
    ↓ Dense (64, relu)
    ↓ Dropout (0.3)
    ↓ Dense (num_classes, softmax)
    """
    model = Sequential([
        # 1st Conv Block
        Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(sequence_length, num_features)),
        BatchNormalization(),
        ReLU(),
        MaxPooling1D(pool_size=2),

        # 2nd Conv Block
        Conv1D(filters=64, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),

        # Temporal Sequence Learning Layer
        LSTM(64, return_sequences=False),

        # Fully Connected Classification Block
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
