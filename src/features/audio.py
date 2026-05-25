import librosa
import numpy as np
import torch
import torchaudio

def load_audio(filepath, target_sr=44100, duration=None):
    """
    Load FULL audio if duration=None; otherwise trims/pads to fixed duration.
    """
    try:
        audio, sr = librosa.load(filepath, sr=target_sr)

        # If duration is forced (for training), apply trim
        if duration:
            desired_len = int(target_sr * duration)
            if len(audio) > desired_len:
                audio = audio[:desired_len]
            else:
                pad = desired_len - len(audio)
                audio = np.pad(audio, (0, pad))

        return audio, sr

    except Exception as e:
        print(f"Error loading {filepath}: {e}")

        if duration:
            return np.zeros(int(target_sr * duration)), target_sr
        
        return np.zeros(1), target_sr


def compute_melspectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Computes Log-Mel Spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def spec_to_image(spec, size):
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
    spec_tensor = torch.tensor(spec_norm).unsqueeze(0).unsqueeze(0)
    spec_resized = torch.nn.functional.interpolate(spec_tensor, size=size, mode='bilinear')
    spec_rgb = spec_resized.repeat(1, 3, 1, 1).squeeze(0)
    return spec_rgb

