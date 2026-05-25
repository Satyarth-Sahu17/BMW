# Deployment Guide

This guide covers the deployment of the BMW application using Docker and Kubernetes.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Kubernetes Cluster (Optional for scaling)

## Local Deployment (Docker Compose)

1. **Build the images:**
   \`\`\`bash
   docker-compose build
   \`\`\`

2. **Start the services:**
   \`\`\`bash
   docker-compose up -d
   \`\`\`

3. **Access the services:**
   - API: `http://localhost:8000`
   - Dashboard: `http://localhost:8501`

## Production Deployment

### Environment Variables

Ensure the following environment variables are set in your `.env` file or deployment configuration:

- `MODEL_PATH`: Path to the trained model file
- `DATA_DIR`: Directory for storing data
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- `API_WORKERS`: Number of Gunicorn workers for the API

### Kubernetes

1. **Apply ConfigMaps and Secrets:**
   \`\`\`bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secrets.yaml
   \`\`\`

2. **Deploy API and Dashboard:**
   \`\`\`bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   \`\`\`

3. **Setup Ingress (Optional):**
   Configure your Ingress controller to route traffic to the services.

## Monitoring

The application exposes metrics that can be scraped by Prometheus.
- API metrics: `/metrics` (if enabled)
- System logs are written to stdout/stderr and can be collected by Fluentd/Logstash.
