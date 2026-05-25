import pytest
import numpy as np
from src.features.augmentation import AudioAugmentor

def test_time_shift():
    augmentor = AudioAugmentor()
    audio = np.random.randn(44100)
    
    shifted = augmentor.time_shift(audio)
    
    assert len(shifted) == len(audio)
    assert isinstance(shifted, np.ndarray)

def test_add_noise():
    augmentor = AudioAugmentor()
    audio = np.random.randn(44100)
    
    noisy = augmentor.add_noise(audio, noise_factor=0.01)
    
    assert len(noisy) == len(audio)
    assert not np.array_equal(audio, noisy)
