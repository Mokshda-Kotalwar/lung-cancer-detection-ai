# Production-optimized Dockerfile for Lung Cancer Detection Web Dashboard
# Uses multi-stage builds to minimize image footprint and speed up deployment.

# Stage 1: Build & Package dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies & OpenCV library prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install dependencies into virtualenv to isolate environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    # Note: On resource-constrained environments like Render Free Tier, we install PyTorch CPU version to prevent OOM
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime container
FROM python:3.9-slim

WORKDIR /app

# Install OpenCV dependencies needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv and application code
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Set environment paths
ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/main.py"]
