# 🐳 GenomicLightning Docker Deployment Guide

This guide provides comprehensive instructions for deploying GenomicLightning using Docker and Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional but recommended)

## 🚀 Quick Start

### 1. Basic Deployment

```bash
# Clone the repository
git clone <repository-url>
cd GenomicLightning

# Build and start the main service
docker-compose up genomic-lightning
```

### 2. Development Environment

```bash
# Start development environment with Jupyter
docker-compose --profile dev up genomic-dev
```

Access Jupyter Lab at: http://localhost:8889

### 3. Full Stack with Monitoring

```bash
# Start all services including TensorBoard
docker-compose --profile monitoring up
```

- Main application: http://localhost:8888
- TensorBoard: http://localhost:6008

## 🔧 Service Configurations

### Available Services

| Service | Description | Port | Profile |
|---------|-------------|------|---------|
| `genomic-lightning` | Main application | 8888, 6006 | default |
| `genomic-dev` | Development environment | 8889, 6007 | dev |
| `tensorboard` | Monitoring dashboard | 6008 | monitoring |
| `training` | Training job runner | - | training |
| `test-runner` | Test execution | - | test |

### Service Profiles

```bash
# Development profile
docker-compose --profile dev up

# Monitoring profile  
docker-compose --profile monitoring up

# Training profile
docker-compose --profile training up

# Testing profile
docker-compose --profile test up
```

## 💾 Volume Mounts

### Data Directories

```bash
# Create required directories
mkdir -p data logs checkpoints

# Set permissions
chmod 755 data logs checkpoints
```

### Volume Mappings

- `./data` → `/app/data` (read-only in production)
- `./logs` → `/app/logs` (training logs, TensorBoard)
- `./checkpoints` → `/app/checkpoints` (model weights)
- `./configs` → `/app/configs` (configuration files)

## 🏃‍♂️ Common Operations

### Running Training

```bash
# Start training service
docker-compose --profile training up -d training

# Execute training
docker-compose exec training python scripts/train_genomic_model.py \
    --config configs/example_danq.yml \
    --data-dir /app/data \
    --log-dir /app/logs
```

### Running Tests

```bash
# Run full test suite
docker-compose --profile test run --rm test-runner

# Run specific tests
docker-compose exec genomic-lightning python -m pytest tests/metrics/ -v
```

### Interactive Development

```bash
# Start development container
docker-compose --profile dev up -d genomic-dev

# Access shell
docker-compose exec genomic-dev bash

# Run Python interactively
docker-compose exec genomic-dev python
```

## 🌍 Environment Variables

### Required Variables

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
vim .env
```

### Key Variables

```bash
# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key

# CUDA device selection
CUDA_VISIBLE_DEVICES=0

# PyTorch cache directory
TORCH_HOME=/app/.torch
```

## 🔒 Security Considerations

### Production Deployment

```bash
# Use specific version tags
docker-compose -f docker-compose.prod.yml up

# Enable security scanning
docker scout quickview
```

### Best Practices

1. **Use non-root user**: Container runs as `genomic` user
2. **Read-only volumes**: Data mounted as read-only in production
3. **Resource limits**: Configure memory and GPU limits
4. **Health checks**: Built-in health monitoring
5. **Network isolation**: Services use dedicated network

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Verify Docker Compose GPU support
docker-compose config | grep -A 5 "devices"
```

#### Memory Issues

```bash
# Increase Docker memory limit
# Edit Docker Desktop settings or daemon.json

# Monitor resource usage
docker stats genomic_lightning_main
```

#### Permission Errors

```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) data logs checkpoints

# Check container user
docker-compose exec genomic-lightning id
```

### Log Analysis

```bash
# View service logs
docker-compose logs genomic-lightning

# Follow logs in real-time
docker-compose logs -f genomic-lightning

# View specific service logs
docker-compose logs tensorboard
```

## 📊 Monitoring and Metrics

### Health Checks

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect genomic_lightning_main | grep -A 10 "Health"
```

### Performance Monitoring

```bash
# Resource usage
docker stats

# GPU utilization (if available)
docker-compose exec genomic-lightning nvidia-smi
```

## 🔄 Updates and Maintenance

### Updating Images

```bash
# Rebuild images
docker-compose build --no-cache

# Pull latest base images
docker-compose pull

# Restart services
docker-compose down && docker-compose up -d
```

### Cleanup

```bash
# Remove containers and volumes
docker-compose down -v

# Clean up Docker system
docker system prune -a
```

## 📈 Scaling

### Multi-GPU Setup

```bash
# Edit docker-compose.yml for multiple GPUs
services:
  genomic-lightning:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # Use all GPUs
              capabilities: [gpu]
```

### Kubernetes Deployment

```bash
# Generate Kubernetes manifests
kompose convert

# Deploy to Kubernetes
kubectl apply -f genomic-lightning-*.yaml
```

---

## 🆘 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review Docker logs: `docker-compose logs`
3. Verify system requirements
4. Check GPU compatibility
5. Open an issue with system details

**Container Labels:**
- Maintainer: GenomicLightning Team
- Version: 0.1.0
- Source: GenomicLightning Repository
