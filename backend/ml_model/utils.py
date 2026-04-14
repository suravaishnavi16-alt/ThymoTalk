import librosa
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_mel_spectrogram(file_path_or_y, sr=16000, n_mels=128, max_time_steps=130):
    """
    Extracts a Mel Spectrogram from an audio file or raw audio array.
    
    Args:
        file_path_or_y (str or np.ndarray): Path to the .wav file OR raw audio array.
        sr (int): Sample rate (only if file_path_or_y is raw data).
        ...
    """
    try:
        if isinstance(file_path_or_y, str):
            # Load audio (3 seconds max, offset by 0.5s to skip silence)
            y, sr = librosa.load(file_path_or_y, sr=sr, duration=3, offset=0.5)
        else:
            y = file_path_or_y
        
        # Extract Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        
        # Convert to log scale (dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Pad or truncate to ensure consistent time steps
        if mel_db.shape[1] < max_time_steps:
            pad_width = max_time_steps - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_time_steps]
            
        # Add channel dimension for CNN (Height, Width, Channel)
        return mel_db[..., np.newaxis]
        
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None
