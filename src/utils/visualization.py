"""
Visualization utilities for spectrograms, waveforms, and results
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Tuple
from pathlib import Path


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """Plot audio waveform"""
    plt.figure(figsize=figsize)
    librosa.display.waveshow(audio, sr=sr, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "Spectrogram",
    y_axis: str = "mel",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """Plot spectrogram"""
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis=y_axis,
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_multiple_spectrograms(
    spectrograms: list,
    labels: list,
    sr: int,
    hop_length: int,
    rows: int = 2,
    cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Plot multiple spectrograms in a grid"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (spec, label) in enumerate(zip(spectrograms, labels)):
        if idx >= len(axes):
            break
        librosa.display.specshow(
            spec,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis',
            ax=axes[idx]
        )
        axes[idx].set_title(label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
