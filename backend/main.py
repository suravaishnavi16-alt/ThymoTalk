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
import random
import logging
import subprocess
import httpx 
import asyncio
import librosa
from groq import Groq 
from dotenv import load_dotenv
from ml_model.audio_model import predict_audio_emotion

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

# -------------------------------------------------------------------
# STARTUP: Pre-load System
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    log_debug("Starting Hybrid AI Fusion Engine...")
    try:
        from ml_model.audio_model import get_predictor
        await asyncio.to_thread(get_predictor)
    except Exception as e:
        log_debug(f"Pre-load warning: {e}")

# -------------------------------------------------------------------
# HEALTH ENDPOINT (For Keep-Alive/Uptime)
# -------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

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


# --- EMOTION MAPPING (Optimized for Speed) ---
def generate_human_response(emotion: str, transcript: str) -> str:
    """Lightweight response fetcher. No heavy regex or aliasing in real-time."""
    primary = str(emotion).lower().strip()
    
    responses = {
        "happy": "That sounds great! I'm so happy for you.",
        "sad": "You sound a bit down. I'm here for you and listening.",
        "angry": "I can understand your frustration. I'm here to listen.",
        "fear": "It sounds like you're feeling a bit anxious. You're safe here.",
        "surprised": "Wow, that sounds like quite a surprise!",
        "neutral": "I’m listening. Tell me more about that.",
        "calm": "I appreciate your calm energy. I'm here for you.",
        "anxiety": "I can sense some tension. You're safe here, take your time.",
        "guilt": "It sounds like you're feeling guilty. It's okay, we all make mistakes.",
        "jealousy": "That sounds like a difficult feeling. I'm here to listen."
    }
    
    return responses.get(primary, responses["neutral"])

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
    # Only remove obvious repeated words within 3-word range to preserve flow
    text = re.sub(r'\b(\w+)(?:\s+\1\b){1,2}', r'\1', text, flags=re.IGNORECASE)
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

        print("RAW:", raw_transcript)

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
                # 1. Start with LLM (Primary intelligence)
                res_llm = await detect_emotion_with_llm(final_transcript)
                
                # 2. Conditional Audio CNN
                audio_res = None
                word_count = len(final_transcript.split())
                # Run audio ONLY if text is short OR LLM is unsure
                if word_count < 3 or res_llm.get("confidence", 0) < 0.6:
                    audio_res = await asyncio.to_thread(predict_audio_emotion, wav_path)
                    audio_res["emotion"] = audio_res.get("emotion", "neutral").lower()

                # 3. Smart Fusion Logic
                if res_llm.get("emotion", "").lower() in ["anxiety", "guilt", "jealousy"]:
                    emotion = res_llm["emotion"]
                    confidence = res_llm.get("confidence", 0.7)
                elif audio_res and audio_res.get("confidence", 0) >= 0.70:
                    emotion = audio_res["emotion"]
                    confidence = audio_res["confidence"]
                elif res_llm.get("confidence", 0) >= 0.55:
                    emotion = res_llm["emotion"]
                    confidence = res_llm["confidence"]
                else:
                    if audio_res and audio_res.get("emotion") == res_llm.get("emotion"):
                        emotion = audio_res["emotion"]
                        confidence = max(audio_res.get("confidence", 0), res_llm.get("confidence", 0))
                    else:
                        emotion = res_llm.get("emotion", "neutral")
                        confidence = res_llm.get("confidence", 0.5)

                if audio_res: print("Audio Signal (CNN Used):", audio_res)
                print("LLM Signal (Groq Used):", res_llm)
                print("Final Decision:", emotion)

            except Exception as e:
                log_debug(f"Fusion Error File Mode: {e}")
                emotion = "neutral"
                confidence = 0.5

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
        # Trigger "First Speech" time reset
        if not state["text_buffer"] and cleaned_chunk:
            state["last_time"] = time.time()

        # prevent duplicate
        if cleaned_chunk and not state["text_buffer"].endswith(cleaned_chunk):
            state["text_buffer"] += " " + cleaned_chunk

        buffer_text = state["text_buffer"].strip()
        word_count = len(buffer_text.split())
        time_gap = time.time() - state["last_time"]

        # EAGER TRIGGER: If user speaks ≥ 3 words, they want an answer soon.
        if word_count < 3 and time_gap < 0.3:
            state["last_time"] = time.time()
            return {
                "emotion": "Listening",
                "confidence": 0.0,
                "response": random.choice(["Listening...", "I'm listening...", "Go on..."]),
                "transcript": buffer_text,
                "interrupt": False,
                "urgent": False
            }
        
        # If user spoke 3+ words but we're still within the quick-gap, show natural filler
        if word_count >= 3 and time_gap < 0.3:
            return {
                "emotion": "Processing",
                "confidence": 0.0,
                "response": random.choice(["Got it...", "Thinking...", "Hmm...", "Let me think...", "I hear you..."]),
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
            # 1. Start with LLM
            res_llm = await detect_emotion_with_llm(final_transcript)
            
            # 2. Conditional Audio CNN
            audio_res = None
            word_count = len(final_transcript.split())
            if word_count < 3 or res_llm.get("confidence", 0) < 0.6:
                audio_res = await asyncio.to_thread(predict_audio_emotion, wav_path)
                audio_res["emotion"] = audio_res.get("emotion", "neutral").lower()

            # 3. Smart Fusion Logic
            if res_llm.get("emotion", "").lower() in ["anxiety", "guilt", "jealousy"]:
                emotion = res_llm["emotion"]
                confidence = res_llm.get("confidence", 0.7)
            elif audio_res and audio_res.get("confidence", 0) >= 0.70:
                emotion = audio_res["emotion"]
                confidence = audio_res["confidence"]
            elif res_llm.get("confidence", 0) >= 0.55:
                emotion = res_llm["emotion"]
                confidence = res_llm["confidence"]
            else:
                if audio_res and audio_res.get("emotion") == res_llm.get("emotion"):
                    emotion = audio_res["emotion"]
                    confidence = max(audio_res.get("confidence", 0), res_llm.get("confidence", 0))
                else:
                    emotion = res_llm.get("emotion", "neutral")
                    confidence = res_llm.get("confidence", 0.5)

            if audio_res: print("Audio Signal (CNN Used):", audio_res)
            print("LLM Signal (Groq Used):", res_llm)
            print("Final Decision:", emotion)

        except Exception as e:
            log_debug(f"Fusion Error Mic Mode: {e}")
            emotion = "neutral"
            confidence = 0.5

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
        print("ERROR:", e)
        return {
            "emotion": "Neutral",
            "confidence": 0.5,
            "response": "I'm here, please continue.",
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

