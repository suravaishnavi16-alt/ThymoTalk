from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
import numpy as np
from datetime import datetime
from collections import deque
import torch
import librosa
from transformers import pipeline
import logging
import subprocess
import httpx 
import asyncio

# -------------------------------------------------------------------
# 🔥 CONFIGURATION: Local Fusion Stage (No External APIs)
# -------------------------------------------------------------------
DEEPGRAM_API_KEY = "50b77470cca24804115d97cd5c074b980df7f0dc"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&punctuate=true"

# Noise Gate Threshold
NOISE_GATE_RMS = 0.005 

# Logging setup
logging.basicConfig(filename="backend_debug.log", level=logging.INFO, format='%(asctime)s - %(message)s')

def log_debug(msg):
    logging.info(msg)
    print(msg, flush=True)

# FFmpeg check
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"

app = FastAPI(title="ThymoTalk Local Fusion API", version="12.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 LOCAL MODEL INITIALIZATION (Audio + Text)
device = 0 if torch.cuda.is_available() else -1
audio_model = None
text_model = None

@app.on_event("startup")
async def startup_event():
    global audio_model, text_model
    log_debug("Loading Local Models from cache...")
    try:
        # Check local first to prevent redundant HF hub checks
        audio_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device, local_files_only=True)
        text_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1, device=device, local_files_only=True)
        log_debug("Local models loaded successfully using local_files_only=True.")
    except Exception as e:
        log_debug(f"Local cache not found or incomplete, downloading models... (Error: {e})")
        audio_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
        text_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1, device=device)
        log_debug("Models downloaded and loaded successfully.")

    log_debug("Ready for inference.")

SESSION_STATE = {} # user_id -> { "probs": deque, "audio": np.array }
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "user_memory.json")

def load_memory():
    if not os.path.exists(MEMORY_PATH): return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_debug(f"Memory load error: {e}")
        return {}

def save_memory(memory):
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        log_debug(f"Memory save error: {e}")

def update_user_habits(user_id: str, emotion: str):
    memory = load_memory()
    if user_id not in memory:
        memory[user_id] = {"emotion_counts": {}, "active_hours": [], "recent_emotions": [], "sessions": 0, "last_seen": ""}
    
    profile = memory[user_id]
    profile["sessions"] = int(profile.get("sessions", 0)) + 1
    counts = profile.get("emotion_counts", {})
    counts[emotion] = int(counts.get(emotion, 0)) + 1
    profile["emotion_counts"] = counts
    profile["last_seen"] = datetime.now().isoformat()
    save_memory(memory)
    return profile

# -------------------------------------------------------------------
# LOCAL FUNCTIONS: Async-ready Emotion & Response
# -------------------------------------------------------------------

async def get_audio_results(combined_audio):
    """Wraps CPU-bound audio model in thread for async speed."""
    return await asyncio.to_thread(audio_model, combined_audio, top_k=8)

async def get_text_results(transcript):
    """Wraps local text model in thread."""
    if not transcript or len(transcript.strip()) < 3:
        return [{"label": "neutral", "score": 0.0}]
    return await asyncio.to_thread(text_model, transcript)

def detect_emotion_from_text(results) -> dict:
    top_res = results[0][0] if isinstance(results[0], list) else results[0]
    label_map = {"joy": "happy", "sadness": "sad", "anger": "angry", "fear": "fear", "surprise": "surprised"}
    raw_label = top_res['label'].lower()
    return {"emotion": label_map.get(raw_label, raw_label), "confidence": float(top_res['score'])}

def get_aggregated_audio_emotion(user_id: str, current_results: list):
    if user_id not in SESSION_STATE:
        SESSION_STATE[user_id] = {"probs": deque(maxlen=6), "audio": np.array([], dtype=np.float32)}
    
    current_map = {r['label'].lower(): r['score'] for r in current_results}
    SESSION_STATE[user_id]["probs"].append(current_map)
    labels = ["happy", "sad", "angry", "fear", "disgust", "surprised", "neutral", "calm"]
    history = SESSION_STATE[user_id]["probs"]
    
    aggregated_scores = {}
    for label in labels:
        weights = [h.get(label, 0.0)**1.5 for h in history]
        aggregated_scores[label] = sum(weights) / (len(history) if history else 1.0)
    
    top_emotion = max(aggregated_scores, key=aggregated_scores.get)
    win_scores = [h.get(top_emotion, 0.0) for h in history]
    avg_score = sum(win_scores) / (len(win_scores) if win_scores else 1.0)
    return top_emotion, avg_score

def generate_human_response(emotion: str, transcript: str) -> str:
    responses = {
        "happy": "That sounds very exciting! I'm glad to hear that.",
        "sad": "You sound a bit down. I'm here for you and listening.",
        "angry": "I can sense some frustration in your voice. Want to talk about it?",
        "fear": "It sounds like you're feeling a bit anxious. I'm here to support you.",
        "surprised": "Oh, that sounds like quite a surprise! Tell me more about it.",
        "neutral": "I'm listening. Tell me more about what's on your mind.",
        "calm": "I appreciate your calm energy. I'm here for you."
    }
    return responses.get(emotion.lower(), "I'm listening.")

async def transcribe_with_deepgram(contents):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            DEEPGRAM_URL, 
            content=contents, 
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/webm"}, 
            timeout=5.0
        )
        return response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "") if response.status_code == 200 else ""

# -------------------------------------------------------------------
# UNIFIED ENDPOINT: Local Fusion (Speed Optimized, No APIs)
# -------------------------------------------------------------------
@app.post("/analyze_chunk")
async def analyze_chunk(file: UploadFile = File(...), user_id: str = Form("default_user")):
    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    
    tmp_path = None
    wav_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        wav_path = tmp_path + ".wav"
        # Convert to 16kHz mono WAV using ffmpeg
        subprocess.run([FFMPEG_EXE, "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path], check=True, capture_output=True)

        # 🚀 1. Simultaneous Parallel processing 🚀 
        current_samples, sr = librosa.load(wav_path, sr=16000)
        
        if user_id not in SESSION_STATE:
            SESSION_STATE[user_id] = {"probs": deque(maxlen=6), "audio": np.array([], dtype=np.float32)}
        
        combined_audio = np.concatenate([SESSION_STATE[user_id]["audio"], current_samples])
        overlap_size = int(sr * 1.5)
        SESSION_STATE[user_id]["audio"] = current_samples[-overlap_size:] if len(current_samples) > overlap_size else current_samples
        
        # Parallel Execution: Transcription + Audio Model
        res_transcript, res_audio_raw = await asyncio.gather(
            transcribe_with_deepgram(contents),
            get_audio_results(combined_audio)
        )
        
        # 2. Sequential Processing of Local Audio
        audio_emotion, audio_score = get_aggregated_audio_emotion(user_id, res_audio_raw)
        
        # 🔥 3. LOCAL Text Emotion Analysis 🔥 
        text_raw = await get_text_results(res_transcript)
        text_res = detect_emotion_from_text(text_raw)
        text_emotion, text_score = text_res["emotion"], text_res["confidence"]

        # 🚀 4. FUSION LOGIC (Local-Only) 🚀 
        # Text dominates if signal is clear, otherwise Audio fallback
        fused_emotion = text_emotion if text_score > 0.5 else audio_emotion
        final_confidence = text_score if text_score > 0.5 else audio_score
        
        human_reply = generate_human_response(fused_emotion, res_transcript)
        update_user_habits(user_id, fused_emotion)

        # Final Response Schema (Consistent UI Format)
        return {
            "emotion": fused_emotion.capitalize(),
            "confidence": float(final_confidence),
            "response": human_reply,
            "transcript": res_transcript.strip(), # Only current chunk to allow frontend clean-start
            "interrupt": True,
            "urgent": fused_emotion in ["angry", "fear", "sad"] and final_confidence > 0.65
        }

    except Exception as e:
        log_debug(f"Local Fusion Error: {e}")
        return {"emotion": "Neutral", "confidence": 0.0, "response": "I'm listening.", "transcript": "", "interrupt": False, "urgent": False}
    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p): 
                try: os.unlink(p)
                except: pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
