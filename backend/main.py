from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import os
import json
from datetime import datetime
from collections import Counter
import torch
import torchaudio
from transformers import pipeline
import logging
import subprocess
import base64
import edge_tts
import asyncio

logging.basicConfig(filename="backend_debug.log", level=logging.INFO, format='%(asctime)s - %(message)s')
def log_debug(msg):
    logging.info(msg)
    print(msg, flush=True)

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    log_debug(f"FFmpeg path: {FFMPEG_EXE}")
except Exception as e:
    FFMPEG_EXE = "ffmpeg"
    log_debug(f"Could not find imageio_ffmpeg: {e}")

app = FastAPI(title="Voice Emotion Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
log_debug(f"Loading model {MODEL_NAME}...")
device = 0 if torch.cuda.is_available() else -1
emotion_classifier = pipeline("audio-classification", model=MODEL_NAME, device=device)
log_debug("Model loaded.")

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "user_memory.json")
SESSION_STATE = {}


def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return {}
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
        memory[user_id] = {
            "emotion_counts": {},
            "active_hours": [],
            "recent_emotions": [],
            "sessions": 0,
            "last_seen": ""
        }

    profile = memory[user_id]
    profile["sessions"] = int(profile.get("sessions", 0)) + 1

    counts = profile.get("emotion_counts", {})
    counts[emotion] = int(counts.get(emotion, 0)) + 1
    profile["emotion_counts"] = counts

    current_hour = datetime.now().hour
    active_hours = profile.get("active_hours", [])
    active_hours.append(current_hour)
    profile["active_hours"] = active_hours[-100:]

    recent = profile.get("recent_emotions", [])
    recent.append(emotion)
    profile["recent_emotions"] = recent[-8:]
    profile["last_seen"] = datetime.now().isoformat()

    memory[user_id] = profile
    save_memory(memory)
    return profile


def habit_hint(profile):
    counts = profile.get("emotion_counts", {})
    if not counts:
        return ""

    top_emotion = max(counts, key=counts.get)
    hours = profile.get("active_hours", [])
    time_hint = ""
    if hours:
        most_active_hour = Counter(hours).most_common(1)[0][0]
        if 5 <= most_active_hour < 12:
            time_hint = "You often check in during mornings."
        elif 12 <= most_active_hour < 17:
            time_hint = "You often check in in the afternoons."
        elif 17 <= most_active_hour < 22:
            time_hint = "You often check in in the evenings."
        else:
            time_hint = "You often check in late at night."

    emotion_hint_map = {
        "happy": "You usually sound upbeat when we talk.",
        "calm": "You usually sound calm when we talk.",
        "neutral": "You usually keep a steady tone in our chats.",
        "sad": "You often open up when you are feeling low.",
        "fear": "I notice you bring worry here when you need support.",
        "angry": "You often come here to release frustration."
    }
    emotion_hint = emotion_hint_map.get(top_emotion, "")
    return f"{emotion_hint} {time_hint}".strip()


def stabilized_emotion(user_id: str, predicted_emotion: str, confidence: float):
    state = SESSION_STATE.get(user_id, {"history": []})
    history = state.get("history", [])
    history.append({"emotion": predicted_emotion, "confidence": float(confidence)})
    history = history[-6:]
    state["history"] = history
    SESSION_STATE[user_id] = state

    weighted = {}
    for item in history:
        weighted[item["emotion"]] = weighted.get(item["emotion"], 0.0) + item["confidence"]

    best_emotion = max(weighted, key=weighted.get)
    best_weight = weighted[best_emotion]
    total_weight = sum(weighted.values()) or 1.0
    agreement = best_weight / total_weight

    # Only trust unstable windows less; fall back to neutral-ish response behavior.
    if agreement < 0.45 and confidence < 0.55:
        return "neutral"
    return best_emotion

def convert_to_wav(input_path):
    output_path = input_path + ".wav"
    try:
        # Simple ffmpeg conversion to 16kHz mono wav
        cmd = [FFMPEG_EXE, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        log_debug(f"FFmpeg failed (Code {e.returncode}):\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return None
    except Exception as e:
        log_debug(f"Conversion system error: {e}")
        return None

@app.post("/analyze_chunk")
async def analyze_chunk(file: UploadFile = File(...), user_id: str = Form("default_user")):
    if not file.content_type.startswith("audio/") and not file.filename.endswith((".wav", ".webm", ".ogg")):
        raise HTTPException(status_code=400, detail="Not an audio file.")

    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
        log_debug(f"--- Audio received: {len(contents)} bytes ---")

    wav_path = None
    try:
        # Convert to WAV first
        wav_path = convert_to_wav(tmp_path)
        if not wav_path:
            raise Exception("Failed to convert audio to wav using FFmpeg")
        log_debug(f"WAV converted: {wav_path}")

        # Predict emotion
        log_debug("Starting prediction...")
        
        # Load audio manually to skip transformers' internal ffmpeg check
        try:
            import librosa
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            log_debug(f"Audio loaded via librosa: {speech_array.shape}, SR={sampling_rate}")
        except Exception as e:
            log_debug(f"Librosa load failure: {e}")
            raise e
        
        # librosa.load returns (samples,) for mono, so no squeeze needed usually
        # But let's be safe
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=0)
        
        try:
            results = emotion_classifier(speech_array, top_k=3)
            log_debug(f"Classification results: {results}")
        except Exception as e:
            log_debug(f"Emotion classifier failure: {e}")
            raise e
        
        predicted_emotion = results[0]['label'].lower()
        confidence = float(results[0]['score'])
        second_confidence = float(results[1]['score']) if len(results) > 1 else 0.0
        margin = confidence - second_confidence

        # Reduce false positives by requiring clearer separation.
        if confidence < 0.45 or margin < 0.10:
            predicted_emotion = "neutral"

        # Angry is often over-predicted; require stronger evidence.
        if predicted_emotion == "angry" and (confidence < 0.75 or margin < 0.18):
            predicted_emotion = "neutral"
        stable_emotion = stabilized_emotion(user_id, predicted_emotion, confidence)
        profile = update_user_habits(user_id, stable_emotion)
        user_habit_summary = habit_hint(profile)

        # Deeper, more empathetic conversational mapping
        # Tuned to be much more descriptive and supportive
        responses = {
            "happy": [
                "That's wonderful! I can truly hear the joy in your voice. What's making you feel so great?",
                "I love that positive energy! You sound absolutely fantastic today.",
                "Your voice is full of life right now! I'm so glad things are going well for you."
            ],
            "surprised": [
                "Wow, you sound genuinely excited! Did something great happen just now?",
                "I hear that spark! You sound like you've had a wonderful surprise.",
                "That sounds like an amazing positive shock! Tell me more about it!"
            ],
            "sad": [
                "I can hear that things are a bit heavy for you right now. I'm here for you.",
                "Your voice sounds a little tired. Do you want to share what's on your mind?",
                "I'm here to listen if you're feeling down. Take your time."
            ],
            "fear": [
                "I hear some worry in your tone. Take a deep breath; I'm right here with you.",
                "It sounds like you're feeling a bit anxious. What's causing this worry?",
                "I hear the tension. You're safe speaking with me; tell me what's happening."
            ],
            "angry": [
                "I hear the frustration. Let it out if you need to; I'm here to listen.",
                "You sound a bit upset. What happened to spark this feeling?",
                "I can hear that heat in your voice. It's okay to feel this way; tell me more."
            ],
            "disgust": [
                "You sound bothered by something. I can hear the distaste in your voice.",
                "That sounds unpleasant to deal with. What exactly happened?",
                "I hear that recoil in your tone. What's bothering you?"
            ],
            "neutral": [
                "I'm listening.",
                "I hear you.",
                "Go on, I'm here."
            ],
            "calm": [
                "You sound very peaceful. It's nice to hear such a calm tone.",
                "I can hear that you're quite relaxed. How has your day been?",
                "A very serene voice! I'm here listening."
            ]
        }

        # Speak naturally: only interrupt for urgent emotional states.
        interrupt = True
        urgent = False
        message = ""
        display_emotion = predicted_emotion.capitalize()

        import random

        if stable_emotion in responses:
            # For distress states, reply urgently (can interrupt user in under ~2s chunks)
            distress_emotions = {"fear", "sad", "angry"}
            if stable_emotion in distress_emotions and confidence > 0.40:
                urgent = True
                message = random.choice(responses[stable_emotion])
            else:
                # For all others, we speak after a pause/silence.
                urgent = False
                message = random.choice(responses[stable_emotion])
        else:
            display_emotion = "Calm"
            message = "I'm here and I'm listening to your voice. Please continue."
        
        display_emotion = stable_emotion.capitalize()

        log_debug(f"Prediction: {display_emotion} ({confidence:.2f}) -> {message} (Urgent: {urgent})")

        # Human-like TTS generation
        audio_base64 = ""
        log_debug(f"Generating TTS for message: {message}")
        try:
            communicate = edge_tts.Communicate(message, "en-US-AvaNeural")
            temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_mp3_path = temp_mp3.name
            temp_mp3.close()
            
            await communicate.save(temp_mp3_path)
            log_debug(f"TTS MP3 saved: {temp_mp3_path}")
            
            with open(temp_mp3_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            log_debug(f"TTS base64 success: {len(audio_base64)} bytes")
            
            os.unlink(temp_mp3_path)
        except Exception as e:
            log_debug(f"TTS Error in edge_tts: {e}")

        return {
            "emotion": display_emotion,
            "confidence": float(confidence),
            "interrupt": interrupt,
            "urgent": urgent,
            "message": message,
            "memory": {
                "sessions": profile.get("sessions", 0),
                "habit_hint": user_habit_summary
            },
            "audio": audio_base64
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        log_debug(f"analyze_chunk error: {e}\n{error_msg}")
        return {
            "emotion": "Neutral",
            "confidence": 0.0,
            "interrupt": False,
            "message": ""
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
