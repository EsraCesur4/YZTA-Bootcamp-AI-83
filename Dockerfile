FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for model downloading
RUN pip install gdown

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/Models

# Download models from your Google Drive links
RUN echo "Downloading models from Google Drive..." && \
    gdown 1-sOrJ-FiFgZ2zdl0uItI3cpiI7MKtyGg -O /app/Models/model.h5 && \
    gdown 1Gk_EdL8YZigmYsn7ogS__qjw-noZgLAl -O /app/Models/fracture_classification_model.h5 && \
    gdown 1tX2FEvVYTFSFKqJKSOmaFt-lXw4I6B4u -O /app/Models/fracture_classification_CNN.h5 && \
    gdown 1e9b2vT5y9OtKqYyNl2gb_Fl5BRx__3TA -O /app/Models/best.pt && \
    echo "Model download completed"

# List downloaded models for verification
RUN ls -la /app/Models/

# Set environment variables
ENV PORT=8080
ENV PYTHONPATH=/app

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "--worker-class", "sync", "app:app"]