---
title: ThymoTalk Backend
emoji: 🎙️
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
---

# Voice Emotion Detection App

An aesthetic, real-time voice emotion intelligent conversational confident built with Next.js and FastAPI. 

## Design Philosophy

This project adopts a highly professional, 3-color minimalist aesthetic (Deep Navy, Cream/Off-white, and Muted Gold). Emojis and visual clutter are completely eliminated to ensure a frictionless, high-quality user experience.

## Real-Time Architecture

The Next.js frontend captures microphone audio directly and streams chunks continuously to the FastAPI backend. 
If the backend detects a "Sad" or "Dull" hue in the user's voice during these chunks, it will trigger an immediate **verbal interrupt**, reading a comforting prompt back via browser Native Speech Synthesis without waiting for the user to finish speaking.

## Prerequisites

- Node.js & npm
- Python 3.9+
- ffmpeg

## Frontend Setup

1. Open a terminal and navigate to the `frontend` folder.
2. Install dependencies (Requires `lucide-react` for minimalist icons).
   ```bash
   cd frontend
   npm install
   ```
3. Start the UI:
   ```bash
   npm run dev
   ```

## Backend Setup

1. Open a terminal and navigate to the `backend` folder.
2. Install pip items:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Run the microservice endpoint:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

Enjoy conversing with EvolveX.
