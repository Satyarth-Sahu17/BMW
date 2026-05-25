import pytest
from fastapi.testclient import TestClient
from api.app import app
import io
import numpy as np
import soundfile as sf

client = TestClient(app)

def create_dummy_audio():
    """Create a dummy audio file for testing."""
    sr = 32000
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    byte_io = io.BytesIO()
    sf.write(byte_io, audio, sr, format='WAV')
    byte_io.seek(0)
    return byte_io

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    audio_file = create_dummy_audio()
    files = {"file": ("test.wav", audio_file, "audio/wav")}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "predictions" in data
    assert "top_prediction" in data
    assert "confidence" in data

def test_predict_batch_endpoint():
    audio_file1 = create_dummy_audio()
    audio_file2 = create_dummy_audio()
    
    files = [
        ("files", ("test1.wav", audio_file1, "audio/wav")),
        ("files", ("test2.wav", audio_file2, "audio/wav"))
    ]
    
    response = client.post("/predict_batch", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2

def test_invalid_file_type():
    files = {"file": ("test.txt", io.BytesIO(b"test"), "text/plain")}
    response = client.post("/predict", files=files)
    # Depending on implementation, might be 400 or 422 or handled gracefully
    # Assuming the API handles it or returns an error
    assert response.status_code in [400, 422, 500]
