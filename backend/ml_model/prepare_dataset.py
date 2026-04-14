import os
import shutil
import zipfile
import requests
import random
import logging
from collections import Counter

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
RAW_DIR = os.path.join(BACKEND_DIR, "dataset_raw")
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
SAMPLES_PER_EMOTION = 250

TARGET_EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprised', 'disgust']
FALLBACK_EMOTIONS = ['anxiety', 'guilt', 'jealous']

# Dataset URLs (Zenodo mirrors for stability)
DATASETS = {
    "RAVDESS": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
    "CREMA-D": "https://zenodo.org/record/7373954/files/AudioWAV.zip?download=1",
    "TESS": "https://zenodo.org/record/4743723/files/TESS.zip?download=1"
}

# Mapping Logic
RAVDESS_MAP = {
    '01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad', 
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprised'
}
CREMAD_MAP = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_folders():
    """Create necessary directories."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for emotion in TARGET_EMOTIONS:
        os.makedirs(os.path.join(DATASET_DIR, emotion), exist_ok=True)
    logger.info("✅ Directories initialized.")

def download_file(url, target_path):
    """Download a file with progress logging."""
    if os.path.exists(target_path):
        logger.info(f"Skipping download, file exists: {target_path}")
        return
    
    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"✅ Downloaded to {target_path}")

def extract_zip(zip_path, extract_path):
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    logger.info(f"✅ Extracted to {extract_path}")

def process_ravdess():
    """Parse and move RAVDESS files."""
    rav_path = os.path.join(RAW_DIR, "RAVDESS")
    os.makedirs(rav_path, exist_ok=True)
    
    # RAVDESS sometimes extracts into Actor_XX folders
    for root, _, files in os.walk(RAW_DIR):
        if "Actor_" in root:
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        emotion = RAVDESS_MAP.get(emotion_code)
                        if emotion in TARGET_EMOTIONS:
                            shutil.copy(os.path.join(root, file), os.path.join(DATASET_DIR, emotion, f"ravdess_{file}"))

def process_cremad():
    """Parse and move CREMA-D files."""
    for file in os.listdir(RAW_DIR):
        if file.endswith(".wav") and "_" in file:
            parts = file.split("_")
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = CREMAD_MAP.get(emotion_code)
                if emotion in TARGET_EMOTIONS:
                    shutil.copy(os.path.join(RAW_DIR, file), os.path.join(DATASET_DIR, emotion, f"cremad_{file}"))

def process_tess():
    """Parse and move TESS files (folders are named by emotion)."""
    for root, dirs, files in os.walk(RAW_DIR):
        for folder in dirs:
            folder_lower = folder.lower()
            emotion = None
            if "happy" in folder_lower: emotion = "happy"
            elif "sad" in folder_lower: emotion = "sad"
            elif "angry" in folder_lower: emotion = "angry"
            elif "neutral" in folder_lower: emotion = "neutral"
            elif "fear" in folder_lower: emotion = "fear"
            elif "surprised" in folder_lower or "ps" in folder_lower: emotion = "surprised"
            elif "disgust" in folder_lower: emotion = "disgust"
            
            if emotion:
                folder_path = os.path.join(root, folder)
                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        shutil.copy(os.path.join(folder_path, file), os.path.join(DATASET_DIR, emotion, f"tess_{file}"))

def balance_dataset():
    """Enforce the 250 samples per emotion rule."""
    logger.info(f"Balancing dataset to {SAMPLES_PER_EMOTION} samples per class...")
    for emotion in TARGET_EMOTIONS:
        emotion_path = os.path.join(DATASET_DIR, emotion)
        files = os.listdir(emotion_path)
        
        if len(files) > SAMPLES_PER_EMOTION:
            random.shuffle(files)
            files_to_remove = files[SAMPLES_PER_EMOTION:]
            for file in files_to_remove:
                os.remove(os.path.join(emotion_path, file))
            logger.info(f"  - {emotion}: Downsampled {len(files)} -> {SAMPLES_PER_EMOTION}")
        else:
            logger.info(f"  - {emotion}: Kept all {len(files)} samples")

def print_report():
    """Summarize the final dataset."""
    print("\n" + "="*40)
    print("      DATASET PREPARATION REPORT")
    print("="*40)
    
    total = 0
    for emotion in TARGET_EMOTIONS:
        count = len(os.listdir(os.path.join(DATASET_DIR, emotion)))
        print(f" {emotion.capitalize():<12}: {count} samples")
        total += count
    
    print("-" * 40)
    print(f" Total Samples : {total}")
    print("="*40)
    
    print("\nNOTE ON SECONDARY EMOTIONS:")
    for m in FALLBACK_EMOTIONS:
        print(f" - {m.capitalize():<10}: These will be handled by LLM fallback.")
    print("=" * 40 + "\n")

def run_pipeline():
    setup_folders()
    
    # Download and Extract
    for name, url in DATASETS.items():
        zip_path = os.path.join(RAW_DIR, f"{name}.zip")
        extract_path = os.path.join(RAW_DIR, name)
        
        try:
            download_file(url, zip_path)
            extract_zip(zip_path, RAW_DIR) # Extract all to RAW_DIR for simplicity
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")

    # Organize
    logger.info("Organizing files into structured folders...")
    process_ravdess()
    process_cremad()
    process_tess()
    
    # Balance
    balance_dataset()
    
    # Report
    print_report()

if __name__ == "__main__":
    run_pipeline()
