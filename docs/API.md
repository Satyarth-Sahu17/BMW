# BMW API Documentation

The BioAcoustic Monitoring API provides endpoints for analyzing audio files and detecting endangered wildlife species.

## Base URL

`http://localhost:8000`

## Endpoints

### Health Check
GET `/health`

Returns the status of the API service.

**Response:**
\`\`\`json
{
  "status": "healthy",
  "timestamp": "2023-10-27T10:00:00"
}
\`\`\`

### Predict Single File
POST `/predict`

Upload a single audio file for classification.

**Parameters:**
- `file`: Audio file (WAV, MP3, FLAC)

**Response:**
\`\`\`json
{
  "filename": "recording_01.wav",
  "predictions": {
    "Species_A": 0.85,
    "Species_B": 0.10,
    "Background": 0.05
  },
  "top_prediction": "Species_A",
  "confidence": 0.85,
  "processing_time": 0.12
}
\`\`\`

### Predict Batch
POST `/predict_batch`

Upload multiple audio files for batch processing.

**Parameters:**
- `files`: List of audio files

**Response:**
\`\`\`json
{
  "results": [
    {
      "filename": "file1.wav",
      "top_prediction": "Species_A",
      "confidence": 0.92
    },
    {
      "filename": "file2.wav",
      "top_prediction": "Background",
      "confidence": 0.78
    }
  ],
  "total_processed": 2
}
\`\`\`

## Error Handling

The API returns standard HTTP error codes:
- `400`: Bad Request (invalid file format)
- `422`: Validation Error
- `500`: Internal Server Error
