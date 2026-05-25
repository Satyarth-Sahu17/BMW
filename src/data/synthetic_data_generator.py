"""
Synthetic audio data generator for BMW project.
Creates realistic animal vocalizations and threat sounds with background noise.
"""

import numpy as np
import soundfile as sf
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
"""
Synthetic audio data generator for BMW project.
Creates realistic animal vocalizations and threat sounds with background noise.
"""

import numpy as np
import soundfile as sf
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def generate_harmonic_tone(
    duration: float,
    sr: int,
    fundamental_freq: float,
    num_harmonics: int = 5,
    harmonic_decay: float = 0.6
) -> np.ndarray:
    """
    Generate harmonic tone with multiple overtones.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        fundamental_freq: Fundamental frequency in Hz
        num_harmonics: Number of harmonic overtones
        harmonic_decay: Decay factor for harmonics
    
    Returns:
        Audio signal
    """
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    for n in range(1, num_harmonics + 1):
        amplitude = harmonic_decay ** (n - 1)
        signal += amplitude * np.sin(2 * np.pi * n * fundamental_freq * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal


def generate_wolf_howl(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate synthetic wolf howl"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Frequency modulation (rises then falls)
    f_start, f_peak, f_end = 400, 800, 500
    freq = np.concatenate([
        np.linspace(f_start, f_peak, len(t) // 2),
        np.linspace(f_peak, f_end, len(t) - len(t) // 2)
    ])
    
    # Generate signal with harmonics
    signal = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    
    # Add harmonics
    signal += 0.3 * np.sin(4 * np.pi * np.cumsum(freq) / sr)
    signal += 0.15 * np.sin(6 * np.pi * np.cumsum(freq) / sr)
    
    # Amplitude envelope (attack-sustain-release)
    n_samples = len(t)
    n_attack = int(0.1 * n_samples)
    n_sustain = int(0.7 * n_samples)
    n_release = n_samples - n_attack - n_sustain

    envelope = np.concatenate([
        np.linspace(0, 1, n_attack),
        np.ones(n_sustain),
        np.linspace(1, 0, n_release)
    ])
    
    signal = signal * envelope
    
    # Add vibrato
    vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
    signal = signal * vibrato
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal


def generate_snow_leopard_call(duration: float = 1.5, sr: int = 44100) -> np.ndarray:
    """Generate synthetic snow leopard call (chuff/roar-like)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Lower frequency range
    f_base = 200
    freq_mod = f_base + 150 * np.sin(2 * np.pi * 3 * t)
    
    # Generate signal
    signal = np.sin(2 * np.pi * np.cumsum(freq_mod) / sr)
    
    # Add noise component (breath-like)
    noise = np.random.normal(0, 0.15, len(t))
    signal = 0.7 * signal + 0.3 * noise
    
    # Pulsed envelope (chuffing)
    num_pulses = 3
    pulse_envelope = np.zeros_like(t)
    pulse_duration = len(t) // (num_pulses * 2)
    
    for i in range(num_pulses):
        start = i * 2 * pulse_duration
        end = start + pulse_duration
        if end <= len(pulse_envelope):
            pulse_envelope[start:end] = np.hanning(pulse_duration)
    
    signal = signal * pulse_envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.7
    
    return signal


def generate_tiger_roar(duration: float = 3.0, sr: int = 44100) -> np.ndarray:
    """Generate synthetic tiger roar"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Very low frequency
    f_base = 150
    freq_mod = f_base + 100 * np.sin(2 * np.pi * 2 * t)
    
    # Generate signal with rich harmonics
    signal = generate_harmonic_tone(duration, sr, f_base, num_harmonics=8)
    
    # Add growl texture (low-frequency noise)
    growl_noise = np.random.normal(0, 0.2, len(t))
    growl_filter = np.exp(-0.001 * np.arange(len(t)))  # Low-pass effect
    growl = np.convolve(growl_noise, growl_filter, mode='same')[:len(t)]
    
    signal = 0.6 * signal + 0.4 * growl
    
    # Amplitude envelope
    n_samples = len(t)
    n_attack = int(0.15 * n_samples)
    n_sustain = int(0.6 * n_samples)
    n_release = n_samples - n_attack - n_sustain

    envelope = np.concatenate([
        np.linspace(0, 1, n_attack),
        np.ones(n_sustain),
        np.linspace(1, 0, n_release)
    ])
    
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal


def generate_gunshot(duration: float = 0.5, sr: int = 44100) -> np.ndarray:
    """Generate synthetic gunshot sound"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Sharp impulse with decay
    impulse_width = int(0.01 * sr)
    impulse = np.zeros(len(t))
    impulse[:impulse_width] = 1.0
    
    # High-frequency noise
    noise = np.random.normal(0, 1, len(t))
    
    # Decay envelope
    decay = np.exp(-15 * t)
    
    signal = (impulse + 0.8 * noise) * decay
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.95
    
    return signal


def generate_chainsaw(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate synthetic chainsaw sound"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Periodic buzz with fundamental around 80-100 Hz
    fundamental = 90
    
    signal = np.zeros_like(t)
    for harmonic in range(1, 20):
        freq = fundamental * harmonic
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    noise = np.random.normal(0, 0.3, len(t))
    signal = 0.7 * signal + 0.3 * noise
    
    # Slight amplitude modulation (engine rhythm)
    modulation = 1 + 0.2 * np.sin(2 * np.pi * 25 * t)
    signal = signal * modulation
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.85
    
    return signal


def generate_background_noise(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate background environmental noise"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # White noise base
    noise = np.random.normal(0, 0.1, len(t))
    
    # Add wind-like low-frequency component
    wind_freq = 0.5
    wind = 0.05 * np.sin(2 * np.pi * wind_freq * t) * np.random.normal(1, 0.3, len(t))
    
    # Add occasional bird chirps (high frequency)
    if np.random.random() > 0.5:
        num_chirps = np.random.randint(1, 4)
        for _ in range(num_chirps):
            chirp_start = np.random.randint(0, len(t) - int(0.1 * sr))
            chirp_duration = int(0.1 * sr)
            chirp_freq = np.random.uniform(2000, 4000)
            chirp = 0.05 * np.sin(2 * np.pi * chirp_freq * np.arange(chirp_duration) / sr)
            chirp *= np.hanning(chirp_duration)
            noise[chirp_start:chirp_start + chirp_duration] += chirp
    
    signal = noise + wind
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.3
    
    return signal


def add_background_noise(signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add background noise to signal"""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise


def generate_dataset(
    output_dir: str,
    num_samples_per_class: int = 100,
    classes: List[str] = None,
    sr: int = 44100
) -> pd.DataFrame:
    """
    Generate synthetic dataset with manifest CSV.
    
    Args:
        output_dir: Output directory for audio files
        num_samples_per_class: Number of samples per class
        classes: List of class names
        sr: Sample rate
    
    Returns:
        Manifest DataFrame
    """
    if classes is None:
        classes = ['wolf', 'snow_leopard', 'tiger', 'gunshot', 'chainsaw', 'background']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generators = {
        'wolf': generate_wolf_howl,
        'snow_leopard': generate_snow_leopard_call,
        'tiger': generate_tiger_roar,
        'gunshot': generate_gunshot,
        'chainsaw': generate_chainsaw,
        'background': generate_background_noise
    }
    
    manifest_data = []
    
    print("Generating synthetic audio dataset...")
    
    for class_name in tqdm(classes, desc="Classes"):
        if class_name not in generators:
            print(f"Warning: No generator for class '{class_name}', skipping...")
            continue
        
        generator = generators[class_name]
        
        for i in tqdm(range(num_samples_per_class), desc=f"  {class_name}", leave=False):
            # Generate audio with random duration variation
            duration = np.random.uniform(1.5, 3.0)
            audio = generator(duration=duration, sr=sr)
            
            # Add variable background noise
            noise_level = np.random.uniform(0.01, 0.08)
            audio = add_background_noise(audio, noise_level)
            
            # Random amplitude scaling
            audio = audio * np.random.uniform(0.7, 1.0)
            
            # Ensure proper range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Save file
            filename = f"{class_name}_{i:04d}.wav"
            filepath = output_path / filename
            sf.write(filepath, audio, sr)
            
            # Add to manifest
            manifest_data.append({
                'id': f"{class_name}_{i:04d}",
                'filepath': str(filepath),
                'start_sec': 0.0,
                'end_sec': duration,
                'label': class_name,
                'gps_lat': np.random.uniform(27.0, 28.5),  # Himalayan region
                'gps_lon': np.random.uniform(86.0, 88.0),
                'recorder_id': f"recorder_{np.random.randint(1, 11):03d}",
                'sample_rate': sr
            })
    
    manifest_df = pd.DataFrame(manifest_data)
    
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset for BMW")
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory for audio files')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per class')
    parser.add_argument('--classes', type=str,
                        default='wolf,snow_leopard,tiger,gunshot,chainsaw,background',
                        help='Comma-separated list of classes')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation set ratio')
    
    args = parser.parse_args()
    
    classes = [c.strip() for c in args.classes.split(',')]
    
    # Generate dataset
    manifest_df = generate_dataset(
        output_dir=args.output_dir,
        num_samples_per_class=args.num_samples,
        classes=classes,
        sr=args.sample_rate
    )
    
    # Shuffle
    manifest_df = manifest_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    n_total = len(manifest_df)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    
    train_df = manifest_df[:n_train]
    val_df = manifest_df[n_train:n_train + n_val]
    test_df = manifest_df[n_train + n_val:]
    
    # Save manifests
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(data_dir / 'manifest_train.csv', index=False)
    val_df.to_csv(data_dir / 'manifest_val.csv', index=False)
    test_df.to_csv(data_dir / 'manifest_test.csv', index=False)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Total samples: {n_total}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    print(f"\nManifests saved to:")
    print(f"  - data/manifest_train.csv")
    print(f"  - data/manifest_val.csv")
    print(f"  - data/manifest_test.csv")


if __name__ == '__main__':
    main()


def generate_harmonic_tone(
    duration: float,
    sr: int,
    fundamental_freq: float,
    num_harmonics: int = 5,
    harmonic_decay: float = 0.6
) -> np.ndarray:
    """
    Generate harmonic tone with multiple overtones.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        fundamental_freq: Fundamental frequency in Hz
        num_harmonics: Number of harmonic overtones
        harmonic_decay: Decay factor for harmonics
    
    Returns:
        Audio signal
    """
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    for n in range(1, num_harmonics + 1):
        amplitude = harmonic_decay ** (n - 1)
        signal += amplitude * np.sin(2 * np.pi * n * fundamental_freq * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal


def generate_wolf_howl(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate synthetic wolf howl"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Frequency modulation (rises then falls)
    f_start, f_peak, f_end = 400, 800, 500
    freq = np.concatenate([
        np.linspace(f_start, f_peak, len(t) // 2),
        np.linspace(f_peak, f_end, len(t) - len(t) // 2)
    ])
    
    # Generate signal with harmonics
    signal = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    
    # Add harmonics
    signal += 0.3 * np.sin(4 * np.pi * np.cumsum(freq) / sr)
    signal += 0.15 * np.sin(6 * np.pi * np.cumsum(freq) / sr)
    
    # Amplitude envelope (attack-sustain-release)
    envelope = np.concatenate([
        np.linspace(0, 1, int(0.1 * len(t))),
        np.ones(int(0.7 * len(t))),
        np.linspace(1, 0, len(t) - int(0.8 * len(t)))
    ])
    
    signal = signal * envelope
    
    # Add vibrato
    vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
    signal = signal * vibrato
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal


def generate_snow_leopard_call(duration: float = 1.5, sr: int = 44100) -> np.ndarray:
    """Generate synthetic snow leopard call (chuff/roar-like)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Lower frequency range
    f_base = 200
    freq_mod = f_base + 150 * np.sin(2 * np.pi * 3 * t)
    
    # Generate signal
    signal = np.sin(2 * np.pi * np.cumsum(freq_mod) / sr)
    
    # Add noise component (breath-like)
    noise = np.random.normal(0, 0.15, len(t))
    signal = 0.7 * signal + 0.3 * noise
    
    # Pulsed envelope (chuffing)
    num_pulses = 3
    pulse_envelope = np.zeros_like(t)
    pulse_duration = len(t) // (num_pulses * 2)
    
    for i in range(num_pulses):
        start = i * 2 * pulse_duration
        end = start + pulse_duration
        if end <= len(pulse_envelope):
            pulse_envelope[start:end] = np.hanning(pulse_duration)
    
    signal = signal * pulse_envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.7
    
    return signal


def generate_tiger_roar(duration: float = 3.0, sr: int = 44100) -> np.ndarray:
    """Generate synthetic tiger roar"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Very low frequency
    f_base = 150
    freq_mod = f_base + 100 * np.sin(2 * np.pi * 2 * t)
    
    # Generate signal with rich harmonics
    signal = generate_harmonic_tone(duration, sr, f_base, num_harmonics=8)
    
    # Add growl texture (low-frequency noise)
    growl_noise = np.random.normal(0, 0.2, len(t))
    growl_filter = np.exp(-0.001 * np.arange(len(t)))  # Low-pass effect
    growl = np.convolve(growl_noise, growl_filter, mode='same')[:len(t)]
    
    signal = 0.6 * signal + 0.4 * growl
    
    # Amplitude envelope
    envelope = np.concatenate([
        np.linspace(0, 1, int(0.15 * len(t))),
        np.ones(int(0.6 * len(t))),
        np.linspace(1, 0, len(t) - int(0.75 * len(t)))
    ])
    
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal


def generate_gunshot(duration: float = 0.5, sr: int = 44100) -> np.ndarray:
    """Generate synthetic gunshot sound"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Sharp impulse with decay
    impulse_width = int(0.01 * sr)
    impulse = np.zeros(len(t))
    impulse[:impulse_width] = 1.0
    
    # High-frequency noise
    noise = np.random.normal(0, 1, len(t))
    
    # Decay envelope
    decay = np.exp(-15 * t)
    
    signal = (impulse + 0.8 * noise) * decay
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.95
    
    return signal


def generate_chainsaw(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate synthetic chainsaw sound"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Periodic buzz with fundamental around 80-100 Hz
    fundamental = 90
    
    signal = np.zeros_like(t)
    for harmonic in range(1, 20):
        freq = fundamental * harmonic
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    noise = np.random.normal(0, 0.3, len(t))
    signal = 0.7 * signal + 0.3 * noise
    
    # Slight amplitude modulation (engine rhythm)
    modulation = 1 + 0.2 * np.sin(2 * np.pi * 25 * t)
    signal = signal * modulation
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.85
    
    return signal


def generate_background_noise(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate background environmental noise"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # White noise base
    noise = np.random.normal(0, 0.1, len(t))
    
    # Add wind-like low-frequency component
    wind_freq = 0.5
    wind = 0.05 * np.sin(2 * np.pi * wind_freq * t) * np.random.normal(1, 0.3, len(t))
    
    # Add occasional bird chirps (high frequency)
    if np.random.random() > 0.5:
        num_chirps = np.random.randint(1, 4)
        for _ in range(num_chirps):
            chirp_start = np.random.randint(0, len(t) - int(0.1 * sr))
            chirp_duration = int(0.1 * sr)
            chirp_freq = np.random.uniform(2000, 4000)
            chirp = 0.05 * np.sin(2 * np.pi * chirp_freq * np.arange(chirp_duration) / sr)
            chirp *= np.hanning(chirp_duration)
            noise[chirp_start:chirp_start + chirp_duration] += chirp
    
    signal = noise + wind
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.3
    
    return signal


def add_background_noise(signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add background noise to signal"""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise


def generate_dataset(
    output_dir: str,
    num_samples_per_class: int = 100,
    classes: List[str] = None,
    sr: int = 44100
) -> pd.DataFrame:
    """
    Generate synthetic dataset with manifest CSV.
    
    Args:
        output_dir: Output directory for audio files
        num_samples_per_class: Number of samples per class
        classes: List of class names
        sr: Sample rate
    
    Returns:
        Manifest DataFrame
    """
    if classes is None:
        classes = ['wolf', 'snow_leopard', 'tiger', 'gunshot', 'chainsaw', 'background']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generators = {
        'wolf': generate_wolf_howl,
        'snow_leopard': generate_snow_leopard_call,
        'tiger': generate_tiger_roar,
        'gunshot': generate_gunshot,
        'chainsaw': generate_chainsaw,
        'background': generate_background_noise
    }
    
    manifest_data = []
    
    print("Generating synthetic audio dataset...")
    
    for class_name in tqdm(classes, desc="Classes"):
        if class_name not in generators:
            print(f"Warning: No generator for class '{class_name}', skipping...")
            continue
        
        generator = generators[class_name]
        
        for i in tqdm(range(num_samples_per_class), desc=f"  {class_name}", leave=False):
            # Generate audio with random duration variation
            duration = np.random.uniform(1.5, 3.0)
            audio = generator(duration=duration, sr=sr)
            
            # Add variable background noise
            noise_level = np.random.uniform(0.01, 0.08)
            audio = add_background_noise(audio, noise_level)
            
            # Random amplitude scaling
            audio = audio * np.random.uniform(0.7, 1.0)
            
            # Ensure proper range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Save file
            filename = f"{class_name}_{i:04d}.wav"
            filepath = output_path / filename
            sf.write(filepath, audio, sr)
            
            # Add to manifest
            manifest_data.append({
                'id': f"{class_name}_{i:04d}",
                'filepath': str(filepath),
                'start_sec': 0.0,
                'end_sec': duration,
                'label': class_name,
                'gps_lat': np.random.uniform(27.0, 28.5),  # Himalayan region
                'gps_lon': np.random.uniform(86.0, 88.0),
                'recorder_id': f"recorder_{np.random.randint(1, 11):03d}",
                'sample_rate': sr
            })
    
    manifest_df = pd.DataFrame(manifest_data)
    
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset for BMW")
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory for audio files')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per class')
    parser.add_argument('--classes', type=str,
                        default='wolf,snow_leopard,tiger,gunshot,chainsaw,background',
                        help='Comma-separated list of classes')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation set ratio')
    
    args = parser.parse_args()
    
    classes = [c.strip() for c in args.classes.split(',')]
    
    # Generate dataset
    manifest_df = generate_dataset(
        output_dir=args.output_dir,
        num_samples_per_class=args.num_samples,
        classes=classes,
        sr=args.sample_rate
    )
    
    # Shuffle
    manifest_df = manifest_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    n_total = len(manifest_df)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    
    train_df = manifest_df[:n_train]
    val_df = manifest_df[n_train:n_train + n_val]
    test_df = manifest_df[n_train + n_val:]
    
    # Save manifests
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(data_dir / 'manifest_train.csv', index=False)
    val_df.to_csv(data_dir / 'manifest_val.csv', index=False)
    test_df.to_csv(data_dir / 'manifest_test.csv', index=False)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Total samples: {n_total}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    print(f"\nManifests saved to:")
    print(f"  - data/manifest_train.csv")
    print(f"  - data/manifest_val.csv")
    print(f"  - data/manifest_test.csv")


if __name__ == '__main__':
    main()
