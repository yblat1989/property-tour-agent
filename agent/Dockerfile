FROM python:3.12-slim

WORKDIR /app

# Install system deps for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Silero VAD model at build time so it's cached in the image
RUN python -c "from livekit.plugins import silero; silero.VAD.load()"

COPY main.py .

CMD ["python", "main.py", "start"]
