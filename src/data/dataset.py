from networkx import config
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.features.audio import load_audio, compute_melspectrogram, spec_to_image

class BMWDataset(Dataset):
    def __init__(self, manifest_path, config, transform=None):
        # Load CSV manifest
        self.df = pd.read_csv(manifest_path)
        self.config = config
        self.transform = transform

        # Create ordered class list
        self.labels = sorted(self.df['label'].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['filepath']
        label_str = row['label']

        # Load audio using duration + sample rate from YAML
        audio, sr = load_audio(
            audio_path,
            target_sr=self.config['data']['sample_rate'],
            duration=self.config['data']['duration']  # using YAML duration
        )

        # Apply augmentation (optional)
        if self.transform:
            audio = self.transform(audio)

        # Compute mel-spectrogram using YAML 'features' block
        spec = compute_melspectrogram(
            audio,
            sr,
            n_mels=self.config['features']['n_mels'],
            n_fft=self.config['features']['n_fft'],
            hop_length=self.config['features']['hop_length']
        )

        # Convert spectrogram to image
        img = spec_to_image(spec, self.config['features']['spectrogram_size'])
        # Convert label string → integer
        label = self.label_to_idx[label_str]

        return img, torch.tensor(label, dtype=torch.long)
    def get_class_weights(self):
        counts = self.df['label'].value_counts().sort_index()
        weights = 1.0 / counts
        return torch.tensor(weights.values, dtype=torch.float)