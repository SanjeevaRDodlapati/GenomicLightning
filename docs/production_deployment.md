# 🚀 Production Deployment Guide

This guide covers production deployment strategies for GenomicLightning in various environments.

## 📋 Pre-Production Checklist

### System Requirements
- [ ] CUDA 11.8+ compatible GPU
- [ ] Docker Engine 20.10+
- [ ] 16GB+ RAM (32GB+ recommended)
- [ ] 100GB+ available storage
- [ ] Network connectivity for model downloads

### Security Requirements
- [ ] Non-root container execution
- [ ] Resource limits configured
- [ ] Secrets management setup
- [ ] Network policies defined
- [ ] Backup strategy implemented

### Performance Requirements
- [ ] GPU memory optimization
- [ ] Batch size tuning completed
- [ ] Load testing performed
- [ ] Monitoring setup validated

## 🏭 Production Environments

### 1. Single Server Deployment

```bash
# Clone repository
git clone <repository-url>
cd GenomicLightning

# Configure environment
cp .env.example .env
vim .env  # Edit production settings

# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
docker-compose logs genomic-lightning
```

#### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  genomic-lightning:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    restart: always
    environment:
      - NODE_ENV=production
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - /data/genomic:/app/data:ro
      - /var/log/genomic:/app/logs
      - /data/checkpoints:/app/checkpoints
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python", "-c", "import genomic_lightning"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 2. Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get pods -n genomic-lightning
kubectl logs -n genomic-lightning deployment/genomic-lightning
```

#### Namespace Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: genomic-lightning
  labels:
    name: genomic-lightning
```

#### Production Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genomic-lightning-ingress
  namespace: genomic-lightning
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - genomic.yourdomain.com
    secretName: genomic-lightning-tls
  rules:
  - host: genomic.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: genomic-lightning-service
            port:
              number: 8888
```

### 3. Cloud Deployment

#### AWS ECS

```json
{
  "family": "genomic-lightning",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "genomic-lightning",
      "image": "your-registry/genomic-lightning:latest",
      "memory": 16384,
      "cpu": 4096,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8888,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "CUDA_VISIBLE_DEVICES",
          "value": "0"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "data",
          "containerPath": "/app/data",
          "readOnly": true
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/genomic-lightning",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "data",
      "host": {
        "sourcePath": "/data/genomic"
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: genomic-lightning
  annotations:
    run.googleapis.com/client-name: gcloud
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 1
      timeoutSeconds: 3600
      containers:
      - image: gcr.io/PROJECT_ID/genomic-lightning:latest
        ports:
        - containerPort: 8888
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          limits:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: data
          mountPath: /app/data
          readOnly: true
      volumes:
      - name: data
        csi:
          driver: gcsfuse.csi.storage.gke.io
          readOnly: true
          volumeAttributes:
            bucketName: genomic-data-bucket
```

## 🔧 Configuration Management

### Environment Variables

```bash
# Production environment variables
export GENOMIC_ENV=production
export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=64
export NUM_WORKERS=8
export MODEL_CACHE_DIR=/app/models
export DATA_PATH=/app/data
export LOG_LEVEL=INFO
export ENABLE_METRICS=true
export METRICS_PORT=9090
```

### Config Files

```yaml
# configs/production.yml
environment: production

model:
  type: danq
  sequence_length: 1000
  num_classes: 919
  checkpoint_path: /app/checkpoints/best_model.ckpt

data:
  batch_size: 64
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true

training:
  max_epochs: 100
  precision: 16
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4

logging:
  level: INFO
  format: json
  file: /app/logs/genomic_lightning.log
  max_size: 100MB
  backup_count: 5

monitoring:
  enable_wandb: true
  enable_tensorboard: true
  metrics_interval: 100
  checkpoint_interval: 1000

performance:
  compile_model: true
  use_mixed_precision: true
  gradient_checkpointing: false
  num_sanity_val_steps: 0
```

## 📊 Monitoring and Observability

### Health Checks

```python
# health_check.py
import requests
import sys

def check_health():
    try:
        response = requests.get('http://localhost:8888/health', timeout=10)
        if response.status_code == 200:
            print("✅ Service is healthy")
            return 0
        else:
            print(f"❌ Service unhealthy: {response.status_code}")
            return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())
```

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'genomic-lightning'
    static_configs:
      - targets: ['genomic-lightning:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GenomicLightning Metrics",
    "panels": [
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_percent"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes"
          }
        ]
      },
      {
        "title": "Training Loss",
        "type": "graph",
        "targets": [
          {
            "expr": "genomic_training_loss"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Hardening

### Container Security

```dockerfile
# Security-hardened Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Create non-root user
RUN groupadd -r genomic && useradd -r -g genomic genomic

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set secure file permissions
COPY --chown=genomic:genomic . /app
WORKDIR /app

# Switch to non-root user
USER genomic

# Remove unnecessary packages
RUN pip uninstall -y setuptools pip wheel
```

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genomic-lightning-netpol
spec:
  podSelector:
    matchLabels:
      app: genomic-lightning
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8888
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

## 🚨 Incident Response

### Alerting Rules

```yaml
# alerting_rules.yml
groups:
- name: genomic_lightning
  rules:
  - alert: HighGPUMemoryUsage
    expr: nvidia_gpu_memory_used_percent > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High GPU memory usage detected"
      
  - alert: ServiceDown
    expr: up{job="genomic-lightning"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GenomicLightning service is down"
      
  - alert: HighTrainingLoss
    expr: genomic_training_loss > 10
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Training loss is unusually high"
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/genomic_lightning_${BACKUP_DATE}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Backup model checkpoints
cp -r /app/checkpoints "${BACKUP_DIR}/"

# Backup configuration files
cp -r /app/configs "${BACKUP_DIR}/"

# Backup logs (last 7 days)
find /app/logs -mtime -7 -type f -exec cp {} "${BACKUP_DIR}/" \;

# Compress backup
tar -czf "${BACKUP_DIR}.tar.gz" "${BACKUP_DIR}"
rm -rf "${BACKUP_DIR}"

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}.tar.gz" s3://genomic-backups/

# Clean old backups (keep last 30 days)
find /backups -name "genomic_lightning_*.tar.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_DIR}.tar.gz"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Production Deploy

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.REGISTRY }}/genomic-lightning:latest
          ${{ secrets.REGISTRY }}/genomic-lightning:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          ${{ secrets.REGISTRY }}/genomic-lightning:${{ github.ref_name }}
```

---

## 📞 Production Support

### Runbook Checklist
- [ ] Monitor system metrics
- [ ] Check application logs
- [ ] Verify GPU utilization
- [ ] Test health endpoints
- [ ] Validate backup integrity
- [ ] Review alert status

### Emergency Contacts
- Platform Team: platform-team@company.com
- On-call Engineer: +1-XXX-XXX-XXXX
- Escalation: engineering-manager@company.com

### Troubleshooting Commands

```bash
# Check service status
kubectl get pods -n genomic-lightning

# View logs
kubectl logs -f deployment/genomic-lightning -n genomic-lightning

# Scale deployment
kubectl scale deployment genomic-lightning --replicas=3 -n genomic-lightning

# Rolling restart
kubectl rollout restart deployment/genomic-lightning -n genomic-lightning

# Emergency rollback
kubectl rollout undo deployment/genomic-lightning -n genomic-lightning
```
