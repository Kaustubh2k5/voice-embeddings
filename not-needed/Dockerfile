# ---------- Base image ----------
FROM python:3.11-slim

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /app

# ---------- Install Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy application + ONNX model ----------
COPY enroll_service.py .
COPY ecapa_embedding.onnx .

# ---------- Runtime ----------
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "enroll_service:app", "--host", "0.0.0.0", "--port", "8000"]