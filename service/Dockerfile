FROM python:3.11-slim

# NOTE: MLX uses Apple Metal (GPU) for acceleration on macOS/Apple Silicon.
# Inside Docker (Linux VM), Metal is unavailable — inference runs on CPU only,
# which is significantly slower. For production, run directly on macOS without Docker.
# This image is suitable for: integration testing, CI without GPU, non-Mac deployment.

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
