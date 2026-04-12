FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from backend folder
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY backend/ .

# Create cache directory for models and ensure permissions
RUN mkdir -p /.cache && chmod -R 777 /.cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOME=/app

# Run the application
CMD ["python", "main.py"]
