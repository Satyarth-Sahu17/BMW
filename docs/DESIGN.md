# BMW System Design Document

## Architecture Overview

The BioAcoustic Monitoring of Endangered Wildlife (BMW) system follows a modular pipeline architecture:

\`\`\`
[Audio Recorders] → [Data Ingestion] → [Preprocessing] → [Feature Extraction]
                                                              ↓
[Dashboard] ← [API Service] ← [Model Inference] ← [Trained Models]
     ↓
[Alerts & Reports]
\`\`\`

## Key Design Decisions

### 1. Dual Framework Support (PyTorch & TensorFlow)

**Decision**: Implement models in both PyTorch and TensorFlow/Keras.

**Rationale**:
- PyTorch: Preferred for research, flexible experimentation
- TensorFlow: Better deployment options (TFLite for edge devices)
- Allows users to choose based on their infrastructure

**Trade-offs**:
- Increased maintenance burden
- Duplicated code for model architectures
- Benefit: Maximum compatibility and deployment flexibility

### 2. Transfer Learning as Primary Approach

**Decision**: Use pre-trained image models (ResNet, VGG) on spectrogram images.

**Rationale**:
- Limited labeled audio data for endangered species
- Image models have learned general feature hierarchies
- Spectrograms provide visual representation of audio
- Proven effectiveness in audio classification tasks

**Alternatives Considered**:
- Audio-specific models (WaveNet, SoundNet): Require more data
- Raw waveform models: Higher computational cost, less interpretable

### 3. Spectrogram Image Size: 224x224

**Decision**: Standardize on 224x224 pixel spectrograms.

**Rationale**:
- Native input size for many pre-trained models
- Balances resolution vs. computational cost
- 5-second audio clips provide sufficient temporal context

**Parameters**:
- 44.1 kHz sampling rate
- 128 Mel bins
- 2048 FFT window, 512 hop length

### 4. Handling Class Imbalance

**Decision**: Multi-strategy approach.

**Techniques**:
1. **Focal Loss**: Automatically down-weights easy examples
2. **Class Weighting**: Inverse frequency weighting
3. **Data Augmentation**: Increases minority class samples
4. **SMOTE**: For traditional ML features

**Rationale**:
- Endangered species are inherently rare
- Human threats (gunshots) are infrequent but critical
- No single technique solves all imbalance issues

### 5. Event Detection vs. Classification

**Decision**: Start with fixed-window classification, extend to event detection.

**Current**: 5-second windows with sliding overlap
**Future**: Variable-length event detection with temporal localization

**Rationale**:
- Classification is simpler to implement and train
- Provides baseline performance metrics
- Event detection added incrementally

### 6. API Design: RESTful

**Decision**: FastAPI with synchronous endpoints.

**Endpoints**:
- `POST /predict`: Single file upload
- `POST /batch_predict`: Multiple files
- `GET /health`: Service status
- `GET /metrics`: Prometheus metrics (future)

**Trade-offs**:
- Simple HTTP requests, widely compatible
- Synchronous: may be slow for large files
- Future: Add async processing with job queue (Celery + Redis)

### 7. Dashboard: Streamlit vs. Custom React

**Decision**: Use Streamlit for initial dashboard.

**Rationale**:
- Rapid prototyping in pure Python
- Built-in components for file upload, plots
- Easy for researchers without web dev experience

**Trade-offs**:
- Less customization than React
- Slower for complex interactions
- Future: Migrate to React if needed

### 8. Data Storage

**Decision**: CSV manifests + file system for audio.

**Current**:
- Manifest CSV: metadata, labels, file paths
- Audio files: stored as WAV on disk

**Future Considerations**:
- Database (PostgreSQL) for large-scale deployments
- Object storage (S3) for cloud deployments
- Time-series DB (InfluxDB) for metrics

### 9. Evaluation Metrics Priority

**Decision**: Emphasize per-class metrics over global accuracy.

**Key Metrics**:
1. Per-class Precision, Recall, F1
2. Confusion Matrix (absolute & normalized)
3. ROC-AUC (one-vs-rest)
4. Matthews Correlation Coefficient (MCC)

**Rationale**:
- Class imbalance makes accuracy misleading
- Conservation focus: false negatives (missed threats) are costly
- MCC provides single-value summary for imbalanced data

### 10. Edge Deployment Strategy

**Decision**: Support ONNX and TFLite export.

**Target Devices**:
- Raspberry Pi 4: TFLite, low power
- NVIDIA Jetson Nano: ONNX with TensorRT, GPU acceleration

**Trade-offs**:
- Model quantization reduces accuracy slightly
- Edge processing reduces bandwidth and latency
- Requires device management infrastructure

## Security Considerations

1. **API Authentication**: Implement API key-based auth
2. **Rate Limiting**: Prevent abuse of prediction endpoints
3. **Input Validation**: Sanitize uploaded files
4. **Location Masking**: Protect endangered species (see DATA_POLICY.md)

## Scalability

### Current Limits
- Single-server deployment
- Synchronous inference
- File-based storage

### Scale-Up Path
1. **Horizontal Scaling**: Load balancer + multiple API instances
2. **Async Processing**: Celery worker pool
3. **Database**: Migrate to PostgreSQL
4. **Caching**: Redis for frequent predictions
5. **CDN**: Static assets and reports

## Monitoring & Observability

**Current**: Basic logging to stdout/file

**Future**:
- Prometheus metrics (request rate, latency, errors)
- Grafana dashboards
- Model performance monitoring (accuracy drift)
- Alert thresholds (e.g., >5 gunshots in 1 hour)

## Testing Strategy

1. **Unit Tests**: Individual functions (pytest)
2. **Integration Tests**: End-to-end API calls
3. **Model Tests**: Smoke tests on tiny datasets
4. **CI/CD**: GitHub Actions on every commit

## Maintenance & Retraining

**Retraining Triggers**:
- New labeled data available
- Model performance degradation detected
- New species or threat types added

**Process**:
1. Version datasets (DVC or manual)
2. Run cross-validation
3. Compare to baseline
4. Blue-green deployment for zero-downtime updates

## Known Limitations

1. **False Positives**: High in noisy environments (rain, wind, rivers)
2. **New Sounds**: Model doesn't generalize to unseen species/threats
3. **Temporal Context**: Fixed 5s windows miss longer events
4. **Computational Cost**: Transfer learning requires GPU for real-time
5. **Data Scarcity**: Limited real-world recordings of endangered species

## Future Enhancements

- [ ] Multi-label classification (overlapping sounds)
- [ ] Speaker identification (individual animals)
- [ ] Acoustic environment classification
- [ ] Active learning for annotation efficiency
- [ ] Federated learning across conservation sites

---

**Document Version**: 1.0  
**Last Updated**: January 2025
