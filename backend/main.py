from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import time
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
from groq import Groq # 🔥 Stage 5 Advanced Orchestration
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# 🔥 CONFIGURATION: STAGE 5 (Groq LLM + Advanced Fusion)
# -------------------------------------------------------------------
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&punctuate=true"

# 🔑 ADD YOUR GROQ API KEY HERE 🔑 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

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

app = FastAPI(title="ThymoTalk Advanced Studio API", version="13.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://thymotalk.netlify.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 LOCAL MODEL INITIALIZATION (Fallback Signals)
device = 0 if torch.cuda.is_available() else -1
audio_model = None
text_model = None

@app.on_event("startup")
async def startup_event():
    global audio_model, text_model
    log_debug("Loading Local Models from cache...")
    try:
        audio_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device, local_files_only=True)
        text_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1, device=device, local_files_only=True)
        log_debug("Local models loaded successfully.")
    except Exception as e:
        audio_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
        text_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1, device=device)
        log_debug("Models downloaded.")

# -------------------------------------------------------------------
# STAGE 5 FUNCTIONS: Groq & Fusion Logic
# -------------------------------------------------------------------

async def detect_emotion_with_llm(transcript: str) -> dict:
    """Stage 5: Deep semantic analysis using Groq's high-speed inference."""
    if not transcript or len(transcript.strip()) < 3:
        return {"emotion": "neutral", "confidence": 0.0, "reason": "No speech detected."}
    
    try:
        completion = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "Analyze the emotional tone of the sentence. Return exactly a JSON object: { \"emotion\": \"angry|sad|happy|neutral|fear|surprised|calm\", \"confidence\": <0.0-1.0>, \"reason\": \"short reason\" }"
                 },
                {"role": "user", "content": f"Sentence: \"{transcript}\""}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data
    except Exception as e:
        log_debug(f"Stage 5 Groq Error: {e}")
        return {"emotion": "neutral", "confidence": 0.0, "reason": "LLM Fallback."}

async def get_audio_results(combined_audio):
    """Async CPU-bound audio processing."""
    return await asyncio.to_thread(audio_model, combined_audio, top_k=8)

def get_aggregated_audio_emotion(user_id: str, current_results: list):
    """Stage 4 logic: handles subtle emotions and aggregation history."""
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
    """Stage 5 Target: Supportive and natural, not just a label."""
    responses = {
        "happy": "That sounds very exciting! I'm glad to hear that.",
        "sad": "You sound a bit down. I'm here for you and listening.",
        "angry": "I can sense some frustration in what you're saying. I'm here to listen.",
        "fear": "It sounds like you're feeling a bit anxious. I'm here to support you.",
        "surprised": "Oh, that sounds like quite a surprise! Tell me more about it.",
        "neutral": "I'm listening. Tell me more about what's on your mind.",
        "calm": "I appreciate your calm energy. I'm here for you."
    }
    return responses.get(emotion.lower(), "I'm listening.")

async def transcribe_with_deepgram(contents):
    """Deepgram STT Task."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            DEEPGRAM_URL, 
            content=contents, 
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/webm"}, 
            timeout=5.0
        )
        return response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "") if response.status_code == 200 else ""

# -------------------------------------------------------------------
# UNIFIED ENDPOINT: Advanced Stage (Parallel Orchestration)
# -------------------------------------------------------------------
SESSION_STATE = {}

# 🔥 --- SILENCE DETECTION (VAD) ---
def is_speech(audio, sr, threshold=0.0005):  # 🔥 LOWERED THRESHOLD
    rms = np.sqrt(np.mean(audio**2))
    return rms > threshold


# 🔥 --- CLEAN TRANSCRIPT ---
def clean_transcript(text: str) -> str:
    import re
    
    # remove repeated phrases
    text = re.sub(r'\b(\w+(?: \w+){0,5})\b(?: \1\b)+', r'\1', text, flags=re.IGNORECASE)
    
    return text.strip()
@app.post("/analyze_chunk")
async def analyze_chunk(file: UploadFile = File(...), user_id: str = Form("default_user")):
    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".webm"

    tmp_path, wav_path = None, None

    try:
        # ---------- SAVE ----------
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        wav_path = tmp_path + ".wav"

        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True
        )

        current_samples, sr = librosa.load(wav_path, sr=16000)

        # ---------- SESSION ----------
        if user_id not in SESSION_STATE:
            SESSION_STATE[user_id] = {
                "text_buffer": "",
                "last_time": time.time()
            }

        state = SESSION_STATE[user_id]

        # ---------- TRANSCRIPT ----------
        raw_transcript = await transcribe_with_deepgram(contents) or ""
        cleaned_chunk = clean_transcript(raw_transcript)

        print("🎤 RAW:", raw_transcript)

        # ---------- FILE MODE ----------
        is_file = len(current_samples) > sr * 2

        if is_file:
            final_transcript = cleaned_chunk.strip()

            if not final_transcript:
                return {
                    "emotion": "Neutral",
                    "confidence": 0.5,
                    "response": "Couldn't understand audio clearly.",
                    "transcript": "",
                    "interrupt": False,
                    "urgent": False
                }

            try:
                res_llm = await detect_emotion_with_llm(final_transcript)
            except:
                res_llm = {"emotion": "neutral", "confidence": 0.5}

            emotion = res_llm.get("emotion", "neutral").lower()
            confidence = float(res_llm.get("confidence", 0.5) or 0.5)

            if confidence < 0.2:
                confidence = 0.85

            reply = generate_human_response(emotion, final_transcript)

            return {
                "emotion": emotion.capitalize(),
                "confidence": confidence,
                "response": reply,
                "transcript": final_transcript,
                "interrupt": True,
                "urgent": emotion in ["angry", "sad", "fear"] and confidence > 0.65
            }

        # ---------- MIC MODE ----------
        # prevent duplicate
        if cleaned_chunk and not state["text_buffer"].endswith(cleaned_chunk):
            state["text_buffer"] += " " + cleaned_chunk

        buffer_text = state["text_buffer"].strip()

        time_gap = time.time() - state["last_time"]

        # wait until user stops speaking
        if time_gap < 1.8:
            state["last_time"] = time.time()
            return {
                "emotion": "Listening",
                "confidence": 0.0,
                "response": "",
                "transcript": buffer_text,
                "interrupt": False,
                "urgent": False
            }

        # finalize
        final_transcript = buffer_text.strip()
        state["text_buffer"] = ""
        state["last_time"] = time.time()

        if not final_transcript:
            return {
                "emotion": "Listening",
                "confidence": 0.0,
                "response": "",
                "transcript": "",
                "interrupt": False,
                "urgent": False
            }

        try:
            res_llm = await detect_emotion_with_llm(final_transcript)
        except:
            res_llm = {"emotion": "neutral", "confidence": 0.5}

        emotion = res_llm.get("emotion", "neutral").lower()
        confidence = float(res_llm.get("confidence", 0.5) or 0.5)

        if confidence < 0.2:
            confidence = 0.85

        reply = generate_human_response(emotion, final_transcript)

        return {
            "emotion": emotion.capitalize(),
            "confidence": confidence,
            "response": reply,
            "transcript": final_transcript,
            "interrupt": True,
            "urgent": emotion in ["angry", "sad", "fear"] and confidence > 0.65
        }

    except Exception as e:
        print("❌ ERROR:", e)
        return {
            "emotion": "Neutral",
            "confidence": 0.5,
            "response": "System error, retrying...",
            "transcript": "",
            "interrupt": False,
            "urgent": False
        }

    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass
   

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

#----------------------------------------------------

