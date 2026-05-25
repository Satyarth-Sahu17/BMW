#!/bin/bash
set -e

echo "Building Docker images..."
docker-compose build

echo "Stopping existing containers..."
docker-compose down

echo "Starting services..."
docker-compose up -d

echo "Deployment complete. Services are running."
docker-compose ps
