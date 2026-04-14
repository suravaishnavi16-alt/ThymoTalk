import os
import numpy as np
import pickle
import logging
import librosa
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Import local utility
from utils import extract_mel_spectrogram

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = "dataset"
SAVED_MODEL_PATH = "ml_model/saved"
MODEL_NAME = "emotion_model.h5"
SCALER_NAME = "scaler.pkl"
ENCODER_NAME = "label_encoder.pkl"

# --- AUGMENTATION HELPERS ---
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def pitch_shift(data, sr=16000, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate=1.1):
    return librosa.effects.time_stretch(data, rate=rate)

def load_dataset(dataset_path):
    """
    Loads audio files and standardizes them.
    Returns raw audio data and labels.
    """
    audio_data, labels = [], []
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path '{dataset_path}' not found.")
        return None, None

    logger.info("Starting dataset loading...")
    
    for emotion_dir in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion_dir)
        if not os.path.isdir(emotion_path):
            continue
            
        logger.info(f"Processing emotion: {emotion_dir}")
        
        for file_name in os.listdir(emotion_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(emotion_path, file_name)
                try:
                    # Load raw audio (fixed sr=16000)
                    y, sr = librosa.load(file_path, sr=16000, duration=3, offset=0.5)
                    audio_data.append(y)
                    labels.append(emotion_dir)
                except Exception as e:
                    logger.warning(f"Error loading {file_name}: {e}")

    return audio_data, labels

def build_cnn_model(input_shape, num_classes):
    """
    Final optimized CNN: Reduced filters and dense units to maximize generalization.
    Target: 80-88% Val Accuracy.
    """
    model = Sequential([
        # Layer 1: Downsized to 16 filters
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Layer 2: Downsized to 32 filters
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        
        # Fully Connected: Reduced to 64 units
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    # 1. Load Data
    raw_audio, raw_labels = load_dataset(DATASET_PATH)
    if not raw_audio:
        logger.error("No data found to train on.")
        return

    # 2. Split Data FIRST (Stratified to ensure equal class distribution)
    # Convert labels strings for stratification
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        raw_audio, raw_labels, test_size=0.2, random_state=42, shuffle=True, stratify=raw_labels
    )

    # 3. Augment ONLY Training Data
    logger.info("Applying focused augmentation to training set...")
    X_train_final, y_train_final = [], []
    
    for audio, label in zip(X_train_orig, y_train_orig):
        # Original
        X_train_final.append(extract_mel_spectrogram(audio))
        y_train_final.append(label)
        
        # Augmentation 1: Moderate Noise
        X_train_final.append(extract_mel_spectrogram(add_noise(audio, noise_factor=0.003)))
        y_train_final.append(label)
        
        # Augmentation 2: Pitch Shift
        X_train_final.append(extract_mel_spectrogram(pitch_shift(audio, n_steps=2)))
        y_train_final.append(label)

    # Prepare Validation Data (Clean Spectrograms)
    X_val_final = [extract_mel_spectrogram(audio) for audio in X_val_orig]
    
    X_train = np.array(X_train_final)
    X_val = np.array(X_val_final)
    
    # 4. Preprocessing Labels
    label_encoder = LabelEncoder()
    y_train_encoded = to_categorical(label_encoder.fit_transform(y_train_final))
    y_val_encoded = to_categorical(label_encoder.transform(y_val_orig))
    num_classes = len(label_encoder.classes_)

    # 5. Normalization (using MinMaxScaler for [0, 1] range)
    num_samples = X_train.shape[0]
    height, width, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    
    scaler = MinMaxScaler()
    X_train_flat = X_train.reshape(num_samples, -1)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(num_samples, height, width, channels)
    
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], height, width, channels)

    # 6. Build and Train Model
    logger.info(f"Initializing final model training with {num_samples} augmented samples...")
    model = build_cnn_model((height, width, channels), num_classes)
    
    # Callbacks (Tighter Control)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    ]

    history = model.fit(
        X_train_scaled, y_train_encoded,
        epochs=20,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_encoded),
        callbacks=callbacks,
        verbose=1
    )

    # 7. Final Report
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    logger.info(f"\nOptimization Results:")
    logger.info(f"Training Accuracy: {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Accuracy Gap: {abs(train_acc - val_acc):.4f}")

    # 8. Save Artifacts
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)
        
    model.save(os.path.join(SAVED_MODEL_PATH, MODEL_NAME))
    pickle.dump(scaler, open(os.path.join(SAVED_MODEL_PATH, SCALER_NAME), "wb"))
    pickle.dump(label_encoder, open(os.path.join(SAVED_MODEL_PATH, ENCODER_NAME), "wb"))
    
    logger.info(f"✅ Training complete. Artifacts saved in {SAVED_MODEL_PATH}")

if __name__ == "__main__":
    train()
