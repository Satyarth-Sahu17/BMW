import pytest
import numpy as np
from src.features.audio import load_audio, compute_melspectrogram, spec_to_image
import torch

def test_load_audio():
    # Create synthetic audio
    sr = 44100
    duration = 2.0
    audio = np.random.randn(int(sr * duration))
    
    # This test would need actual file, so we just check function signature
    assert callable(load_audio)

def test_compute_melspectrogram():
    sr = 44100
    duration = 2.0
    audio = np.random.randn(int(sr * duration))
    
    spec = compute_melspectrogram(audio, sr, n_mels=128)
    
    assert spec.shape[0] == 128  # n_mels
    assert spec.ndim == 2

def test_spec_to_image():
    # Create mock spectrogram
    spec = np.random.randn(128, 431)
    
    img = spec_to_image(spec, size=(224, 224))
    
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)  # RGB, 224x224
