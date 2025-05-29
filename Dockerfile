# Dockerfile for GenomicLightning
# Multi-stage build for optimized production image

# Build stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .
COPY VERSION .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Production stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS production

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy application code
COPY --from=builder /app /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash genomic
RUN chown -R genomic:genomic /app
USER genomic

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV TORCH_HOME=/app/.torch
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import genomic_lightning; print('✅ GenomicLightning is healthy')" || exit 1

# Default command
CMD ["python", "-c", "import genomic_lightning; print('🧬 GenomicLightning container is ready!')"]

# Expose common ports for Jupyter, TensorBoard, etc.
EXPOSE 8888 6006 8080

# Add labels for better container management
LABEL maintainer="GenomicLightning Team"
LABEL version="0.1.0"
LABEL description="Deep learning framework for genomic sequence analysis"
LABEL org.opencontainers.image.source="https://github.com/user/GenomicLightning"
