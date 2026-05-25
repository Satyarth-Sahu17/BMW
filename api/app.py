from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pytorch_models import TransferLearningModel
from src.features.audio import load_audio, compute_melspectrogram, spec_to_image

app = FastAPI(
    title="BMW API",
    description="BioAcoustic Monitoring of Endangered Wildlife API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
CLASSES = ["wolf", "snow_leopard", "tiger", "gunshot", "chainsaw", "background"]

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded = False

@app.on_event("startup")
async def load_model():
    global model, model_loaded
    try:
        if Path(MODEL_PATH).exists():
            model = TransferLearningModel(
                num_classes=len(CLASSES),
                base_model='mobilenet_v2',
                pretrained=True    
            )
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            model_loaded = True
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}. Running in demo mode.")
    except Exception as e:
        print(f"Error loading model: {e}. Running in demo mode.")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "BMW API - BioAcoustic Monitoring of Endangered Wildlife",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/batch_predict", "/docs"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "classes": CLASSES
    }

@app.get("/metrics")
def metrics():
    """Prometheus-style metrics endpoint (placeholder)"""
    return {
        "predictions_total": 0, 
        "prediction_latency_seconds": 0.0,
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict species/threat from uploaded audio file.
    """
    if not model_loaded:
        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "predictions": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "label": "wolf",
                        "score": 0.87,
                        "note": "Demo mode - model not loaded"
                    }
                ],
                "metadata": {
                    "sample_rate": 44100,
                    "duration": 5.0,
                    "model": "resnet50"
                }
            }
        )

    try:
        start_time = time.time()

        contents = await file.read()

        # Save temp file
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        audio, sr = load_audio(temp_path, target_sr=44100, duration=None)
        duration = len(audio) / sr  

        spec = compute_melspectrogram(audio, sr)
        img = spec_to_image(spec, size=(224, 224))
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            score, pred_idx = torch.max(probs, 1)

        label = CLASSES[pred_idx.item()]
        confidence = float(score.item())

        # class-wise probabilities
        class_probs = {
            cls: float(prob) for cls, prob in zip(CLASSES, probs[0].cpu().numpy().tolist())
        }

        # cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        total_time = time.time() - start_time

        # -----------------------------
        # Final JSON response
        # -----------------------------
        return {
            "filename": file.filename,
            "predictions": [
                {
                    "start": 0.0,
                    "end": duration,
                    "label": label,
                    "score": confidence
                }
            ],
            "class_probabilities": class_probs,
            "metadata": {
                "sample_rate": sr,
                "duration": round(duration, 3),
                "model": "resnet50",
                "processing_time_seconds": round(total_time, 4)
            }
        }

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple audio files.
    
    Returns:
        JSON array with predictions for each file
    """
    results = []
    
    for file in files:
        try:
            # Reuse single predict logic
            result = await predict(file)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total_files": len(files)}

@app.get("/classes")
def get_classes():
    """Get list of supported classes"""
    return {"classes": CLASSES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
