#!/bin/bash
set -e

echo "Running Unit Tests..."
pytest tests/test_audio_features.py tests/test_augmentation.py tests/test_dataset.py tests/test_metrics.py

echo "Running API Tests..."
pytest tests/test_api.py

echo "Running Integration Tests..."
pytest tests/test_integration.py

echo "All tests passed!"
