"""
Dataset classes for TensorFlow and PyTorch
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Dict
from pathlib import Path

from src.data.preprocessing import (
    load_audio, normalize_audio, remove_silence,
    extract_features, sliding_window
)
from src.data.augmentation import (
    time_shift, pitch_shift, add_noise, spec_augment
)


class AudioDataset(Dataset):
    """PyTorch Dataset for audio classification"""
    
    def __init__(
        self,
        manifest_path: str,
        config: Dict,
        augment: bool = False,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            manifest_path: Path to manifest CSV
            config: Configuration dictionary
            augment: Apply augmentation
            transform: Additional transforms
        """
        self.manifest = pd.read_csv(manifest_path)
        self.config = config
        self.augment = augment
        self.transform = transform
        
        # Class encoding
        self.classes = config['data']['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Feature config
        self.sr = config['data']['sample_rate']
        self.feature_config = {
            'n_mels': config['features']['n_mels'],
            'n_fft': config['features']['n_fft'],
            'hop_length': config['features']['hop_length']
        }
        self.feature_type = config['features']['type']
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Load audio
        audio, sr = load_audio(
            row['filepath'],
            sr=self.sr,
            offset=row['start_sec'],
            duration=row['end_sec'] - row['start_sec']
        )
        
        # Preprocessing
        if self.config['preprocessing']['normalize']:
            audio = normalize_audio(audio)
        
        if self.config['preprocessing']['remove_silence']:
            audio = remove_silence(
                audio, sr,
                top_db=self.config['preprocessing']['silence_threshold']
            )
        
        # Augmentation
        if self.augment and self.config['augmentation']['enabled']:
            if np.random.random() < 0.5:
                audio = time_shift(
                    audio,
                    self.config['augmentation']['time_shift']
                )
            if np.random.random() < 0.5:
                audio = pitch_shift(
                    audio, sr,
                    self.config['augmentation']['pitch_shift']
                )
            if np.random.random() < 0.5:
                audio = add_noise(
                    audio,
                    self.config['augmentation']['noise_injection']
                )
        
        # Extract features
        features = extract_features(
            audio, sr,
            feature_type=self.feature_type,
            config=self.feature_config
        )
        
        # SpecAugment (on features)
        if self.augment and self.config['augmentation']['spec_augment']:
            if np.random.random() < 0.5:
                features = spec_augment(
                    features,
                    self.config['augmentation']['spec_augment_freq_mask'],
                    self.config['augmentation']['spec_augment_time_mask']
                )
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0)  # Add channel dim
        
        # Label
        label = self.class_to_idx[row['label']]
        label = torch.LongTensor([label])
        
        if self.transform:
            features = self.transform(features)
        
        return features, label


def create_tf_dataset(
    manifest_path: str,
    config: Dict,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset.
    
    Args:
        manifest_path: Path to manifest CSV
        config: Configuration dictionary
        batch_size: Batch size
        augment: Apply augmentation
        shuffle: Shuffle dataset
    
    Returns:
        tf.data.Dataset
    """
    manifest = pd.read_csv(manifest_path)
    classes = config['data']['classes']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    def generator():
        for idx in range(len(manifest)):
            row = manifest.iloc[idx]
            
            # Load and preprocess audio
            audio, sr = load_audio(
                row['filepath'],
                sr=config['data']['sample_rate'],
                offset=row['start_sec'],
                duration=row['end_sec'] - row['start_sec']
            )
            
            if config['preprocessing']['normalize']:
                audio = normalize_audio(audio)
            
            # Extract features
            features = extract_features(
                audio, sr,
                feature_type=config['features']['type'],
                config={
                    'n_mels': config['features']['n_mels'],
                    'n_fft': config['features']['n_fft'],
                    'hop_length': config['features']['hop_length']
                }
            )
            
            # Add channel dimension
            features = np.expand_dims(features, axis=-1)
            
            # Label
            label = class_to_idx[row['label']]
            
            yield features, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
