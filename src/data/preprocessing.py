"""
Audio preprocessing utilities: loading, normalization, silence removal, feature extraction
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict
from pathlib import Path


def load_audio(
    filepath: str,
    sr: int = 44100,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Args:
        filepath: Path to audio file
        sr: Target sample rate
        mono: Convert to mono
        offset: Start reading after this time (seconds)
        duration: Only load this duration (seconds)
    
    Returns:
        (audio, sample_rate)
    """
    audio, sample_rate = librosa.load(
        filepath,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration
    )
    return audio, sample_rate


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal.
    
    Args:
        audio: Audio signal
        method: 'peak' or 'rms'
    
    Returns:
        Normalized audio
    """
    if method == 'peak':
        return audio / (np.max(np.abs(audio)) + 1e-8)
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio ** 2))
        return audio / (rms + 1e-8) * 0.1  # Target RMS of 0.1
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 40,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Remove silence from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold in dB below reference
        frame_length: Frame length
        hop_length: Hop length
    
    Returns:
        Audio with silence removed
    """
    non_silent_intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    if len(non_silent_intervals) == 0:
        return audio
    
    non_silent_audio = np.concatenate([
        audio[start:end] for start, end in non_silent_intervals
    ])
    
    return non_silent_audio


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    delta: bool = True,
    delta_delta: bool = True
) -> np.ndarray:
    """
    Extract MFCC features.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCCs
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of Mel bands
        delta: Include delta features
        delta_delta: Include delta-delta features
    
    Returns:
        MFCC features (n_features, n_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    features = [mfcc]
    
    if delta:
        mfcc_delta = librosa.feature.delta(mfcc)
        features.append(mfcc_delta)
    
    if delta_delta:
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(mfcc_delta2)
    
    return np.vstack(features)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    power: float = 2.0,
    log: bool = True
) -> np.ndarray:
    """
    Extract Mel spectrogram.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of Mel bands
        n_fft: FFT window size
        hop_length: Hop length
        power: Exponent for magnitude spectrogram
        log: Convert to log scale
    
    Returns:
        Mel spectrogram (n_mels, n_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power
    )
    
    if log:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec


def extract_features(
    audio: np.ndarray,
    sr: int,
    feature_type: str = 'mel_spectrogram',
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Extract features based on type.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        feature_type: 'mfcc', 'mel_spectrogram', or 'log_mel'
        config: Feature extraction configuration
    
    Returns:
        Extracted features
    """
    if config is None:
        config = {}
    
    if feature_type == 'mfcc':
        return extract_mfcc(audio, sr, **config)
    elif feature_type in ['mel_spectrogram', 'log_mel']:
        return extract_mel_spectrogram(audio, sr, log=True, **config)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def create_spectrogram_image(
    spectrogram: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convert spectrogram to image for transfer learning.
    
    Args:
        spectrogram: Spectrogram (freq, time)
        target_size: Target image size (height, width)
    
    Returns:
        Image array (height, width, 3)
    """
    from PIL import Image
    
    # Normalize to 0-255
    spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
    spec_uint8 = (spec_norm * 255).astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(spec_norm)
    img = img.resize(target_size, Image.BILINEAR)
    
    # Convert to RGB
    img_array = np.array(img)
    img_rgb = np.stack([img_array] * 3, axis=-1)
    
    return img_rgb


def sliding_window(
    audio: np.ndarray,
    sr: int,
    window_size: float = 2.0,
    hop_size: float = 1.0
) -> list:
    """
    Create sliding windows from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        window_size: Window size in seconds
        hop_size: Hop size in seconds
    
    Returns:
        List of audio windows
    """
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    windows = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        end = start + window_samples
        windows.append(audio[start:end])
    
    # Add last window if audio doesn't fit exactly
    if len(audio) > window_samples and (len(audio) - window_samples) % hop_samples != 0:
        windows.append(audio[-window_samples:])
    
    return windows
