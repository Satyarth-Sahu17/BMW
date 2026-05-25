import os
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import numpy as np

class DataIngestor:
    def __init__(self, raw_dir, processed_dir, sample_rate=44100, window_size=5.0, overlap=0.5):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_process(self, manifest_path=None):
        """Scans raw directory, segments audio, and creates a manifest."""
        audio_files = list(self.raw_dir.glob("**/*.wav")) + list(self.raw_dir.glob("**/*.mp3"))
        data = []
        
        print(f"Found {len(audio_files)} audio files.")
        
        for filepath in tqdm(audio_files, desc="Processing Audio"):
            try:
                # Load audio
                y, sr = librosa.load(filepath, sr=self.sample_rate)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Sliding window segmentation
                step = int(self.window_size * (1 - self.overlap) * sr)
                window_samples = int(self.window_size * sr)
                
                for i, start_sample in enumerate(range(0, len(y) - window_samples + 1, step)):
                    end_sample = start_sample + window_samples
                    segment = y[start_sample:end_sample]
                    
                    # Save segment
                    seg_filename = f"{filepath.stem}_seg_{i:04d}.wav"
                    seg_path = self.processed_dir / seg_filename
                    sf.write(seg_path, segment, sr)
                    
                    # Metadata (In real scenario, label comes from annotation file)
                    # Here we assume filename contains label for simplicity or use a lookup
                    label = self._extract_label(filepath.name)
                    
                    data.append({
                        "id": seg_filename.replace(".wav", ""),
                        "filepath": str(seg_path),
                        "original_file": filepath.name,
                        "start_sec": start_sample / sr,
                        "end_sec": end_sample / sr,
                        "label": label,
                        "sample_rate": sr
                    })
                    
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                
        df = pd.DataFrame(data)
        if manifest_path:
            df.to_csv(manifest_path, index=False)
            print(f"Manifest saved to {manifest_path}")
        return df

    def _extract_label(self, filename):
        # Simple heuristic: filename contains label
        # Customize this based on actual data naming convention
        valid_labels = ["wolf", "tiger", "gunshot", "chainsaw", "background"]
        for label in valid_labels:
            if label in filename.lower():
                return label
        return "unknown"

if __name__ == "__main__":
    ingestor = DataIngestor(raw_dir="data/raw", processed_dir="data/processed")
    ingestor.scan_and_process(manifest_path="data/manifest_processed.csv")
