# BioAcoustic Monitoring of Endangered Wildlife (BMW)

## 🎯 Project Overview

Non-invasive automated detection and classification of animal vocalizations and human threats (poaching/deforestation) using audio recorded in the field.

**Target Species**: Wolves, Snow Leopards, Siberian Tigers (easily adaptable to new species)

**Threat Detection**: Gunshots, Chainsaws, Human voices

## 🚀 Quick Start

### Using Docker (Recommended)

\`\`\`bash
# Build and run the full stack (API + Dashboard + Services)
docker-compose up --build

# Access:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
\`\`\`

### Local Installation

\`\`\`bash
# Create conda environment
conda env create -f environment.yml
conda activate bmw

# Or use pip
pip install -r requirements.txt

# Generate synthetic training data
python src/data/synthetic_data_generator.py

# Train a model
python src/train.py --config configs/experiment1.yaml

# Evaluate the model
python src/evaluate.py --pred experiments/latest/predictions.json --gt data/manifest_test.json --out_dir experiments/latest/eval/

# Start API server
python api/app.py

# Start dashboard (in another terminal)
streamlit run dashboard/app.py
\`\`\`

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/BMW_Training_Colab.ipynb)

Open `notebooks/BMW_Training_Colab.ipynb` in Google Colab for GPU-accelerated training.

## 📁 Project Structure

\`\`\`
bmw/
├── api/                          # FastAPI service
│   ├── app.py                   # Main API server
│   ├── predict_sample.py        # CLI prediction tool
│   └── requirements.txt
├── configs/                      # Experiment configurations
│   └── experiment1.yaml
├── dashboard/                    # Streamlit dashboard
│   └── app.py
├── data/                        # Dataset folder
│   ├── raw/                     # Raw audio files
│   ├── processed/               # Preprocessed features
│   ├── manifest_train.csv       # Training manifest
│   └── manifest_test.csv        # Test manifest
├── deploy/                      # Deployment artifacts
│   ├── docker/
│   ├── kubernetes/
│   └── edge/                    # Edge device deployment
├── experiments/                 # Training runs and results
├── models/                      # Trained models
├── notebooks/                   # Jupyter notebooks
│   ├── BMW_Training_Colab.ipynb
│   └── EDA_and_Preprocessing.ipynb
├── samples/                     # Sample audio files
├── src/
│   ├── data/                    # Data utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Dataset classes
│   │   ├── augmentation.py      # Audio augmentation
│   │   ├── preprocessing.py     # Audio preprocessing
│   │   └── synthetic_data_generator.py
│   ├── eval/                    # Evaluation tools
│   │   ├── __init__.py
│   │   ├── confusion_and_metrics.py  # Main evaluation suite
│   │   └── interpretability.py       # Grad-CAM, SHAP
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   ├── cnn_keras.py        # TensorFlow/Keras models
│   │   ├── cnn_pytorch.py      # PyTorch models
│   │   ├── transfer_learning.py
│   │   └── crnn.py             # CRNN architecture
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── visualization.py
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── tests/                       # Unit and integration tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── environment.yml
├── LICENSE
├── DATA_POLICY.md
├── DESIGN.md
└── MODEL_CARD.md
\`\`\`

## 🎓 Usage Guide

### 1. Generate Synthetic Data

\`\`\`bash
python src/data/synthetic_data_generator.py \
    --output_dir data/raw \
    --num_samples 1000 \
    --classes wolf,snow_leopard,tiger,gunshot,chainsaw,background
\`\`\`

### 2. Train a Model

\`\`\`bash
# Using config file
python src/train.py --config configs/experiment1.yaml

# With command-line arguments
python src/train.py \
    --framework keras \
    --model_type cnn \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
\`\`\`

### 3. Evaluate

\`\`\`bash
python src/evaluate.py \
    --pred experiments/2025-01-20/predictions.json \
    --gt data/manifest_test.json \
    --out_dir experiments/2025-01-20/eval/
\`\`\`

This generates:
- `confusion_matrix.png` (absolute and normalized)
- `metrics_summary.json` (all metrics)
- `roc_curves.png`
- `pr_curves.png`
- `report.html` (comprehensive evaluation report)

### 4. Make Predictions

\`\`\`bash
# Single file
python api/predict_sample.py --audio samples/wolf_howl_1.wav

# Using API
curl -X POST "http://localhost:8000/predict" \
  -F "file=@samples/wolf_howl_1.wav"
\`\`\`

### 5. Run Dashboard

\`\`\`bash
streamlit run dashboard/app.py
\`\`\`

## 🧪 Testing

\`\`\`bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_evaluation.py -v
\`\`\`

## 📊 Evaluation Metrics

The evaluation suite computes comprehensive metrics:

- **Confusion Matrix** (absolute & normalized)
- **Classification Metrics**: Precision, Recall, F1-Score (micro/macro/weighted)
- **ROC-AUC** (per-class and macro-average)
- **PR-AUC** (Precision-Recall AUC)
- **Matthews Correlation Coefficient (MCC)**
- **Cohen's Kappa**
- **Balanced Accuracy**
- **Log-Loss**
- **Calibration Metrics** (Brier score, reliability diagram)
- **Detection Metrics**: mAP, IoU-based event matching
- **Per-class Error Analysis** with misclassification examples

## 🔧 Configuration

Edit `configs/experiment1.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Feature extraction parameters
- Class weights for imbalanced data

## 🚢 Deployment

### Docker

\`\`\`bash
docker build -t bmw-api .
docker run -p 8000:8000 bmw-api
\`\`\`

### Edge Deployment (Raspberry Pi / Jetson)

\`\`\`bash
# Convert model to ONNX
python deploy/edge/convert_to_onnx.py --model models/best_model.h5

# Deploy on edge device
cd deploy/edge
docker build -f Dockerfile.edge -t bmw-edge .
\`\`\`

### Cloud Deployment

See `deploy/cloud/` for deployment guides:
- AWS Elastic Beanstalk
- GCP Cloud Run
- Heroku

## 📈 Adding New Species

1. Collect audio samples for the new species
2. Add entries to `data/manifest_train.csv` with the new label
3. Retrain the model with updated class list
4. Evaluate on test set including new species

## 🔐 Security & Privacy

- API includes rate limiting and API key authentication
- GPS coordinates can be masked in public dashboards
- See `DATA_POLICY.md` for data handling guidelines

## 📚 Documentation

- **DESIGN.md**: Architecture decisions and trade-offs
- **MODEL_CARD.md**: Model limitations, biases, and performance
- **DATA_POLICY.md**: Privacy and ethical considerations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Audio samples for endangered species: Macaulay Library, Xeno-canto
- Pre-trained models: ImageNet, AudioSet
- Conservation partners: WWF, WCS

## ⚠️ Limitations

- False positives possible in noisy environments
- Performance depends on dataset quality and balance
- Requires human-in-the-loop review for critical alarms
- Location data requires careful privacy handling

## 📞 Support

For issues and questions, please open a GitHub issue or contact the development team.
