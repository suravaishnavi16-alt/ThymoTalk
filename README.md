# ThymoTalk.AI (Advanced Hybrid Emotion Engine)

ThymoTalk is a real-time, highly empathetic Voice Assistant powered by a dual-signal Hybrid AI Architecture. It listens to your voice and analyzes both the **acoustic tone** (how you sound) and the **semantic context** (what you say) to provide natural, supportive, and instantaneous responses.

## 🚀 Features

- **Real-Time Voice Streaming**: Processes audio chunks instantly without waiting for sentence completion.
- **Hybrid "Smart Fusion" Engine**:
  - **Acoustic Layer**: A custom 1D-CNN trained on MFCC voice features to detect raw tonal emotion (Happy, Sad, Angry, Fear, etc.).
  - **Semantic Layer (LLM)**: Integration with the blazing-fast **Groq API (Llama 3)** to understand complex context and nuances like *Anxiety*, *Guilt*, and *Jealousy*.
- **Ultra-Fast Turn-Taking**: A conversational silence threshold of **300ms**, making it feel like a natural human interaction.
- **Zero-Latency UX**: Immediate frontend visual feedback ("Listening...", "Thinking...") to eliminate perceived latency.

## 🛠️ Technology Stack

- **Frontend**: Next.js 14, React, TailwindCSS, Web Audio API
- **Backend**: FastAPI (Python), Uvicorn, asyncio
- **Machine Learning**: TensorFlow/Keras (CNN), Librosa, Scikit-Learn
- **External APIs**: Deepgram (Speech-to-Text), Groq (LLM Inference)
- **Deployment**: Netlify (Frontend), Hugging Face Spaces (Backend/Docker)

## 💻 Local Development

### 1. Start the Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
python main.py
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Environment Variables
You need a `.env` file in the `backend/` directory:
```env
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
```

## 🧠 Model Training
The CNN model was trained on a custom synthesized dataset using the Deepgram TTS engine to generate emotional voice variants across different pitches and speeds. The training scripts are available in `backend/ml_model/`.

*UI Design inspired by premium, minimalist AI tools with glassmorphism and smooth dynamic real-time transcripts.*
