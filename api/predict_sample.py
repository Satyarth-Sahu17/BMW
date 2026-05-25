import argparse
import requests
import json
import sys
from pathlib import Path

def predict_sample(audio_path, api_url="http://localhost:8000/predict"):
    path = Path(audio_path)
    if not path.exists():
        print(f"Error: File {audio_path} not found.")
        return

    print(f"Sending {audio_path} to {api_url}...")
    
    try:
        with open(path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Result:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: API returned {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is it running?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BMW API with a sample file")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="API URL")
    args = parser.parse_args()
    
    predict_sample(args.audio, args.url)
