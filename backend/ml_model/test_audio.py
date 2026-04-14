import os
import sys
import logging
import json

# Add parent directory to path to allow importing audio_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_model import get_predictor
except ImportError:
    from ml_model.audio_model import get_predictor

# Set up logging for debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_inference(file_path):
    """
    Tests the audio emotion model on a specific file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"\n[DEBUG] Testing file: {file_path}")
    
    predictor = get_predictor()
    result = predictor.predict(file_path)

    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
        return None

    # Clear print output as requested
    print("-" * 30)
    print(f"RESULT:")
    print(f"  Emotion    : {result['emotion'].upper()}")
    print(f"  Confidence : {result['confidence']:.4f}")
    print("-" * 30)
    
    # Return as requested for utility
    return {
        "emotion": result['emotion'],
        "confidence": result['confidence']
    }

if __name__ == "__main__":
    # Check if user provided an argument
    if len(sys.argv) > 1:
        test_inference(sys.argv[1])
    else:
        # Default to test.wav in the current directory if it exists
        test_file = "test.wav"
        if os.path.exists(test_file):
            test_inference(test_file)
        else:
            print("Usage: python ml_model/test_audio.py <path_to_wav>")
            print("Or place a 'test.wav' file in the current directory.")
