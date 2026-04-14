import os
import pickle
import numpy as np
import logging
from tensorflow.keras.models import load_model

# Import local utility
try:
    from .utils import extract_mel_spectrogram
except ImportError:
    from utils import extract_mel_spectrogram

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SAVED_MODEL_PATH = "ml_model/saved"
MODEL_NAME = "emotion_model.h5"
SCALER_NAME = "scaler.pkl"
ENCODER_NAME = "label_encoder.pkl"

class AudioEmotionPredictor:
    def __init__(self, model_dir=SAVED_MODEL_PATH):
        self.model_path = os.path.join(model_dir, MODEL_NAME)
        self.scaler_path = os.path.join(model_dir, SCALER_NAME)
        self.encoder_path = os.path.join(model_dir, ENCODER_NAME)
        
        self.model = None
        self.scaler = None
        self.encoder = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.scaler = pickle.load(open(self.scaler_path, "rb"))
                self.encoder = pickle.load(open(self.encoder_path, "rb"))
                logger.info("ML artifacts loaded successfully.")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Please run training first.")
        except Exception as e:
            logger.error(f"Error loading ML artifacts: {e}")

    def predict(self, file_path):
        """
        Predicts the emotion of a given audio file.
        """
        if self.model is None:
            return {"error": "Model not loaded."}

        # 1. Feature Extraction
        feature = extract_mel_spectrogram(file_path)
        if feature is None:
            return {"error": "Feature extraction failed."}

        # 2. Preprocessing
        # Reshape to (1, Height*Width*Channels) for scaler
        feature_flat = feature.reshape(1, -1)
        feature_scaled_flat = self.scaler.transform(feature_flat)
        
        # Reshape back for CNN (1, Height, Width, Channels)
        feature_scaled = feature_scaled_flat.reshape(1, *feature.shape)

        # 3. Prediction
        predictions = self.model.predict(feature_scaled, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        emotion = self.encoder.inverse_transform([class_idx])[0]

        return {
            "emotion": str(emotion),
            "confidence": float(confidence),
            "all_probabilities": {str(k): float(v) for k, v in zip(self.encoder.classes_, predictions[0])}
        }

# Singleton instance for easy import
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = AudioEmotionPredictor()
    return predictor

def predict_audio_emotion(file_path):
    """
    High-level API for integration.
    """
    return get_predictor().predict(file_path)

if __name__ == "__main__":
    # Test block
    print(predict_audio_emotion("ml_model/test.wav"))
