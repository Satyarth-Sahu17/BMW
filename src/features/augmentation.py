import numpy as np
import librosa
import torch
import torchaudio.transforms as T

class AudioAugmentor:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def time_shift(self, audio, shift_limit=0.4):
        sig_len = len(audio)
        shift_amt = int(np.random.uniform(-shift_limit, shift_limit) * sig_len)
        return np.roll(audio, shift_amt)

    def pitch_shift(self, audio, n_steps=2):
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=np.random.uniform(-n_steps, n_steps))

    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    def spec_augment(self, spec_tensor, freq_mask_param=15, time_mask_param=35):
        # Expects torch tensor of shape (1, n_mels, time)
        masking = T.Compose([
            T.FrequencyMasking(freq_mask_param=freq_mask_param),
            T.TimeMasking(time_mask_param=time_mask_param)
        ])
        return masking(spec_tensor)

    def mixup(self, data, targets, alpha=1.0):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)
        new_data = data * lam + shuffled_data * (1 - lam)
        
        # Return mixed data and both targets with lambda
        return new_data, targets, shuffled_targets, lam
