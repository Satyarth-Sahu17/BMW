"""
Audio data augmentation techniques
"""

import numpy as np
import librosa
from typing import Optional


def time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """
    Shift audio in time.
    
    Args:
        audio: Audio signal
        shift_max: Maximum shift as fraction of length
    
    Returns:
        Time-shifted audio
    """
    shift = int(len(audio) * shift_max * np.random.uniform(-1, 1))
    return np.roll(audio, shift)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """
    Shift pitch of audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_steps: Number of semitones to shift (positive or negative)
    
    Returns:
        Pitch-shifted audio
    """
    n_steps = np.random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add Gaussian noise to audio.
    
    Args:
        audio: Audio signal
        noise_factor: Standard deviation of noise
    
    Returns:
        Audio with added noise
    """
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise


def time_stretch(audio: np.ndarray, rate: float = 1.2) -> np.ndarray:
    """
    Stretch or compress audio in time.
    
    Args:
        audio: Audio signal
        rate: Stretch factor (> 1 speeds up, < 1 slows down)
    
    Returns:
        Time-stretched audio
    """
    rate = np.random.uniform(1 / rate, rate)
    return librosa.effects.time_stretch(audio, rate=rate)


def spec_augment(
    spectrogram: np.ndarray,
    freq_mask_param: int = 15,
    time_mask_param: int = 25,
    num_freq_masks: int = 2,
    num_time_masks: int = 2
) -> np.ndarray:
    """
    Apply SpecAugment to spectrogram.
    
    Args:
        spectrogram: Spectrogram (freq, time)
        freq_mask_param: Maximum frequency mask width
        time_mask_param: Maximum time mask width
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
    
    Returns:
        Augmented spectrogram
    """
    spec_aug = spectrogram.copy()
    n_mels, n_frames = spec_aug.shape
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        spec_aug[f0:f0 + f, :] = 0
    
    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, n_frames - t)
        spec_aug[:, t0:t0 + t] = 0
    
    return spec_aug


def mixup(
    audio1: np.ndarray,
    audio2: np.ndarray,
    alpha: float = 0.2
) -> tuple:
    """
    Apply mixup augmentation.
    
    Args:
        audio1: First audio signal
        audio2: Second audio signal
        alpha: Mixup parameter
    
    Returns:
        (mixed_audio, lambda_value)
    """
    lam = np.random.beta(alpha, alpha)
    
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    mixed = lam * audio1 + (1 - lam) * audio2
    
    return mixed, lam
