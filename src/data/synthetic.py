import os
import numpy as np
import pandas as pd
import soundfile as sf
import argparse
from pathlib import Path
from tqdm import tqdm

def generate_tone(freq, duration, sr=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_noise(duration, sr=44100, amplitude=0.1):
    return amplitude * np.random.normal(0, 1, int(sr * duration))

def generate_synthetic_dataset(output_dir, manifest_path, n_samples=100):
    """
    Generates a synthetic dataset with simple tones representing different classes.
    Classes: wolf (low freq), tiger (mid freq), gunshot (impulse), chainsaw (high freq), background (noise)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = {
        "wolf": {"freq": 400, "type": "tone"},
        "tiger": {"freq": 150, "type": "tone"},
        "gunshot": {"freq": 0, "type": "impulse"},
        "chainsaw": {"freq": 2000, "type": "sawtooth"},
        "background": {"freq": 0, "type": "noise"}
    }
    
    data = []
    sr = 44100
    duration = 5.0
    
    print(f"Generating {n_samples} synthetic samples...")
    
    for i in tqdm(range(n_samples)):
        label = np.random.choice(list(classes.keys()))
        filename = f"sample_{i:04d}_{label}.wav"
        filepath = output_dir / filename
        
        # Base background noise
        audio = generate_noise(duration, sr, amplitude=0.05)
        
        # Add signal
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        if classes[label]["type"] == "tone":
            signal = 0.5 * np.sin(2 * np.pi * classes[label]["freq"] * t)
            # Add some modulation
            signal *= (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        elif classes[label]["type"] == "sawtooth":
            signal = 0.3 * (2 * (t * classes[label]["freq"] - np.floor(0.5 + t * classes[label]["freq"])))
        elif classes[label]["type"] == "impulse":
            signal = np.zeros_like(t)
            idx = np.random.randint(0, len(t)-1000)
            signal[idx:idx+1000] = np.random.normal(0, 1, 1000) * np.exp(-np.linspace(0, 10, 1000))
        else: # background
            signal = np.zeros_like(t)
            
        audio += signal
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        sf.write(filepath, audio, sr)
        
        # Metadata
        start_sec = 0.0
        end_sec = duration
        gps_lat = 27.98 + np.random.normal(0, 0.1)
        gps_lon = 86.92 + np.random.normal(0, 0.1)
        
        data.append({
            "id": f"sample_{i:04d}",
            "filepath": str(filepath),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "recorder_id": f"rec_{np.random.randint(1, 5)}",
            "sample_rate": sr
        })
        
    df = pd.DataFrame(data)
    df.to_csv(manifest_path, index=False)
    print(f"Manifest saved to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--manifest_path", type=str, default="data/manifest.csv")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.output_dir, args.manifest_path, args.n_samples)
