# Lightweight Python Image
FROM python:3.11-slim

# Install system dependencies required for handling audio (librosa/soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up standard HuggingFace Space user permissions
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirement list first for Docker caching
COPY --chown=user backend/requirements.txt requirements.txt

# Install ML dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the backend files
COPY --chown=user backend/ /app/backend/

# HuggingFace requires servers to bind to 0.0.0.0:7860
ENV HOST="0.0.0.0"
ENV PORT=7860

# We start the server from within the backend directory
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
