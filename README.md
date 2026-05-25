# BioAcoustic Monitoring of Endangered Wildlife (BMW)

## Project Overview

Non-invasive automated detection and classification of animal vocalizations and human threats (poaching/deforestation) using audio recorded in the field.

**Target Species**: Wolves, Snow Leopards, Siberian Tigers (easily adaptable to new species)

**Threat Detection**: Gunshots, Chainsaws, Human voices

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run the full stack (API + Dashboard + Services)
docker-compose up --build

# Access:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Local Installation

```bash
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
```

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Satyarth-Sahu17/BMW/blob/main/notebooks/BMW_Training_Colab.ipynb)

Open `notebooks/BMW_Training_Colab.ipynb` in Google Colab for GPU-accelerated training.

## Project Structure

```
bmw/
├── backend/                      # Backend services
│   ├── api/                      # FastAPI service
│   │   ├── app.py               # Main API server
│   │   ├── predict_sample.py    # CLI prediction tool
│   │   └── requirements.txt
│   ├── src/                     # Source code
│   │   ├── models/              # Model architectures
│   │   │   ├── __init__.py
│   │   │   ├── cnn_keras.py     # TensorFlow/Keras models
│   │   │   ├── cnn_pytorch.py   # PyTorch models
│   │   │   ├── transfer_learning.py
│   │   │   └── crnn.py          # CRNN architecture
│   │   ├── data/                # Data utilities
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py   # Dataset classes
│   │   │   ├── augmentation.py  # Audio augmentation
│   │   │   ├── preprocessing.py # Audio preprocessing
│   │   │   └── synthetic_data_generator.py
│   │   ├── eval/                # Evaluation tools
│   │   │   ├── __init__.py
│   │   │   ├── confusion_and_metrics.py # Main evaluation suite
│   │   │   └── interpretability.py      # Grad-CAM, SHAP analysis
│   │   ├── utils/               # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── config.py        # Configuration loader
│   │   │   ├── logger.py        # Logging utilities
│   │   │   └── visualization.py # Visualization helpers
│   │   ├── train.py             # Training script
│   │   └── evaluate.py          # Evaluation script
│   ├── configs/                 # Experiment configurations
│   │   └── experiment1.yaml     # Sample experiment config
│   ├── models/                  # Trained model artifacts
│   └── requirements.txt         # Python dependencies
│
├── frontend/                     # Frontend dashboard
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── LandingPage.jsx           # Welcome page with features
│   │   │   ├── UploadPage.jsx            # File upload interface
│   │   │   ├── ResultsDashboard.jsx      # Main analysis results
│   │   │   ├── ComparisonPage.jsx        # Contract comparison view
│   │   │   ├── LoadingScreen.jsx         # Analysis progress indicator
│   │   │   ├── RiskGauge.jsx             # Circular risk score display
│   │   │   └── IssueCard.jsx             # Expandable issue details
│   │   ├── data/                # Mock data
│   │   │   └── mockContractData.js       # Sample contract analysis data
│   │   ├── App.jsx              # Main application component
│   │   └── index.css            # Global styles and Tailwind imports
│   └── package.json             # Frontend dependencies
│
├── data/                        # Dataset folder
│   ├── raw/                     # Raw audio files
│   ├── processed/               # Preprocessed features
│   ├── manifest_train.csv       # Training manifest
│   └── manifest_test.csv        # Test manifest
│
├── deploy/                      # Deployment artifacts
│   ├── docker/                  # Docker configurations
│   ├── kubernetes/              # Kubernetes manifests
│   ├── cloud/                   # Cloud deployment guides
│   │   ├── aws_eb.md
│   │   ├── gcp_cloudrun.md
│   │   └── heroku.md
│   └── edge/                    # Edge device deployment
│       ├── Dockerfile.edge
│       └── convert_to_onnx.py
│
├── notebooks/                   # Jupyter notebooks
│   ├── BMW_Training_Colab.ipynb # Google Colab training notebook
│   ├── EDA_and_Preprocessing.ipynb # Exploratory data analysis
│   └── Model_Evaluation.ipynb   # Model evaluation analysis
│
├── samples/                     # Sample audio files for testing
│   ├── wolf_howl_1.wav
│   ├── gunshot_1.wav
│   └── chainsaw_1.wav
│
├── experiments/                 # Training runs and results
│   └── latest/
│       ├── predictions.json
│       ├── metrics_summary.json
│       └── eval/
│           ├── confusion_matrix.png
│           ├── roc_curves.png
│           ├── pr_curves.png
│           └── report.html
│
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_preprocessing.py    # Data preprocessing tests
│   ├── test_models.py           # Model architecture tests
│   ├── test_data_loader.py      # Data loader tests
│   └── test_evaluation.py       # Evaluation metrics tests
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # GitHub Actions CI/CD
│       └── deploy.yml           # Deployment workflow
│
├── docker-compose.yml           # Multi-container Docker setup
├── Dockerfile                   # Backend Docker image
├── .dockerignore
├── .gitignore
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment file
├── LICENSE                      # MIT License
├── DATA_POLICY.md              # Privacy and data handling
├── DESIGN.md                   # Architecture and design decisions
├── MODEL_CARD.md               # Model documentation and limitations
└── README.md                   # This file
```

## Usage Guide

### Generate Synthetic Data

```bash
python backend/src/data/synthetic_data_generator.py \
    --output_dir data/raw \
    --num_samples 1000 \
    --classes wolf,snow_leopard,tiger,gunshot,chainsaw,background
```

### Train a Model

```bash
# Using config file
python backend/src/train.py --config backend/configs/experiment1.yaml

# With command-line arguments
python backend/src/train.py \
    --framework keras \
    --model_type cnn \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Evaluate

```bash
python backend/src/evaluate.py \
    --pred experiments/2025-01-20/predictions.json \
    --gt data/manifest_test.json \
    --out_dir experiments/2025-01-20/eval/
```

This generates:
- `confusion_matrix.png` (absolute and normalized)
- `metrics_summary.json` (all metrics)
- `roc_curves.png`
- `pr_curves.png`
- `report.html` (comprehensive evaluation report)

### Make Predictions

```bash
# Single file
python backend/api/predict_sample.py --audio samples/wolf_howl_1.wav

# Using API
curl -X POST "http://localhost:8000/predict" \
  -F "file=@samples/wolf_howl_1.wav"
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend/src --cov-report=html

# Run specific test file
pytest tests/test_evaluation.py -v
```

## Evaluation Metrics

The evaluation suite computes comprehensive metrics:

- Confusion Matrix (absolute & normalized)
- Classification Metrics: Precision, Recall, F1-Score (micro/macro/weighted)
- ROC-AUC (per-class and macro-average)
- PR-AUC (Precision-Recall AUC)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Balanced Accuracy
- Log-Loss
- Calibration Metrics (Brier score, reliability diagram)
- Detection Metrics: mAP, IoU-based event matching
- Per-class Error Analysis with misclassification examples

## Configuration

Edit `backend/configs/experiment1.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Feature extraction parameters
- Class weights for imbalanced data

## Deployment

### Docker

```bash
docker build -t bmw-api .
docker run -p 8000:8000 bmw-api
```

### Docker Compose (Full Stack)

```bash
docker-compose up --build
```

This starts:
- Backend API on `http://localhost:8000`
- Frontend on `http://localhost:3000`
- Streamlit Dashboard on `http://localhost:8501`

### Edge Deployment (Raspberry Pi / Jetson)

```bash
# Convert model to ONNX
python deploy/edge/convert_to_onnx.py --model backend/models/best_model.h5

# Deploy on edge device
cd deploy/edge
docker build -f Dockerfile.edge -t bmw-edge .
docker run -p 8000:8000 bmw-edge
```

### Cloud Deployment

See deployment guides in `deploy/cloud/`:
- **AWS Elastic Beanstalk**: `deploy/cloud/aws_eb.md`
- **GCP Cloud Run**: `deploy/cloud/gcp_cloudrun.md`
- **Heroku**: `deploy/cloud/heroku.md`

## Adding New Species

1. Collect audio samples for the new species
2. Add entries to `data/manifest_train.csv` with the new label
3. Update the config file with the new class
4. Retrain the model: `python backend/src/train.py --config backend/configs/experiment1.yaml`
5. Evaluate on test set including new species

## Security and Privacy

- API includes rate limiting and API key authentication
- GPS coordinates can be masked in public dashboards
- All audio data encrypted in transit and at rest
- See `DATA_POLICY.md` for comprehensive data handling guidelines

## Documentation

- **DESIGN.md**: Architecture decisions, trade-offs, and system design
- **MODEL_CARD.md**: Model limitations, biases, performance benchmarks, and training details
- **DATA_POLICY.md**: Privacy considerations, ethical guidelines, and data handling best practices

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and write tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

- Audio samples for endangered species: [Macaulay Library](https://www.macaulaylibrary.org/), [Xeno-canto](https://www.xeno-canto.org/)
- Pre-trained models: ImageNet, AudioSet
- Conservation partners: [WWF](https://www.worldwildlife.org/), [WCS](https://www.wcs.org/)

## Limitations

- False positives possible in noisy environments (requires field testing and validation)
- Performance depends on dataset quality and class balance
- Requires human-in-the-loop review for critical alarms in production
- Location data requires careful privacy handling to protect endangered species habitats
- Model performance may vary across different geographical regions

## Citation

If you use BMW in your research, please cite:

```bibtex
@software{bmw2025,
  title={BioAcoustic Monitoring of Endangered Wildlife},
  author={Sahu, Satyarth},
  year={2025},
  url={https://github.com/Satyarth-Sahu17/BMW}
}
```

## Support

For issues and questions:
- 📝 Open a [GitHub issue](https://github.com/Satyarth-Sahu17/BMW/issues)
- 📧 Contact the development team
- 🐛 Report bugs with detailed reproduction steps
- 💡 Suggest features and improvements
