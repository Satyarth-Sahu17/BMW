# BioAcoustic Monitoring of Endangered Wildlife (BMW)

## Project Overview

Non-invasive automated detection and classification of animal vocalizations and human threats (poaching/deforestation) using audio recorded in the field.

**Target Species**: Wolves, Snow Leopards, Siberian Tigers (easily adaptable to new species)

**Threat Detection**: Gunshots, Chainsaws, Human voices

## Quick Start

```bash
# Install dependencies
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

## Project Structure

```
BMW/
├── src/                              # Backend source code
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── cnn_keras.py              # TensorFlow/Keras models
│   │   ├── cnn_pytorch.py            # PyTorch models
│   │   ├── transfer_learning.py
│   │   └── crnn.py                   # CRNN architecture
│   │
│   ├── data/                         # Data utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Dataset classes
│   │   ├── augmentation.py           # Audio augmentation
│   │   ├── preprocessing.py          # Audio preprocessing
│   │   └── synthetic_data_generator.py
│   │
│   ├── eval/                         # Evaluation tools
│   │   ├── __init__.py
│   │   ├── confusion_and_metrics.py  # Main evaluation suite
│   │   └── interpretability.py       # Grad-CAM, SHAP analysis
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration loader
│   │   ├── logger.py                 # Logging utilities
│   │   └── visualization.py          # Visualization helpers
│   │
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Evaluation script
│
├── api/                              # FastAPI service
│   ├── app.py                        # Main API server
│   ├── predict_sample.py             # CLI prediction tool
│   └── requirements.txt
│
├── dashboard/                        # Streamlit dashboard
│   ├── app.py                        # Main dashboard application
│   └── requirements.txt
│
├── frontend/                         # React frontend (optional)
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── LandingPage.jsx       # Welcome page with features
│   │   │   ├── UploadPage.jsx        # File upload interface
│   │   │   ├── ResultsDashboard.jsx  # Main analysis results
│   │   │   ├── ComparisonPage.jsx    # Contract comparison view
│   │   │   ├── LoadingScreen.jsx     # Analysis progress indicator
│   │   │   ├── RiskGauge.jsx         # Circular risk score display
│   │   │   └── IssueCard.jsx         # Expandable issue details
│   │   ├── data/                     # Mock data
│   │   │   └── mockContractData.js   # Sample analysis data
│   │   ├── App.jsx                   # Main application component
│   │   └── index.css                 # Global styles and Tailwind imports
│   └── package.json                  # Frontend dependencies
│
├── configs/                          # Experiment configurations
│   └── experiment1.yaml              # Sample experiment config
│
├── data/                             # Dataset folder
│   ├── raw/                          # Raw audio files
│   ├── processed/                    # Preprocessed features
│   ├── manifest_train.csv            # Training manifest
│   └── manifest_test.csv             # Test manifest
│
├── models/                           # Trained model artifacts
│
├── deploy/                           # Deployment artifacts
│   ├── docker/                       # Docker configurations
│   │   ├── Dockerfile.backend        # Backend container
│   │   ├── Dockerfile.dashboard      # Dashboard container
│   │   └── Dockerfile.api            # API container
│   ├── kubernetes/                   # Kubernetes manifests
│   ├── cloud/                        # Cloud deployment guides
│   │   ├── aws_eb.md
│   │   ├── gcp_cloudrun.md
│   │   └── heroku.md
│   └── edge/                         # Edge device deployment
│       ├── Dockerfile.edge
│       └── convert_to_onnx.py
│
├── notebooks/                        # Jupyter notebooks
│   ├── BMW_Training_Colab.ipynb      # Google Colab training notebook
│   ├── EDA_and_Preprocessing.ipynb   # Exploratory data analysis
│   └── Model_Evaluation.ipynb        # Model evaluation analysis
│
├── samples/                          # Sample audio files for testing
│   ├── wolf_howl_1.wav
│   ├── gunshot_1.wav
│   └── chainsaw_1.wav
│
├── experiments/                      # Training runs and results
│   └── latest/
│       ├── predictions.json
│       ├── metrics_summary.json
│       └── eval/
│           ├── confusion_matrix.png
│           ├── roc_curves.png
│           ├── pr_curves.png
│           └── report.html
│
├── tests/                            # Unit and integration tests
│   ├── __init__.py
│   ├── test_preprocessing.py         # Data preprocessing tests
│   ├── test_models.py                # Model architecture tests
│   ├── test_data_loader.py           # Data loader tests
│   └── test_evaluation.py            # Evaluation metrics tests
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # GitHub Actions CI/CD
│       └── deploy.yml                # Deployment workflow
│
├── docker-compose.yml                # Multi-container Docker setup
├── Dockerfile                        # Backend Docker image
├── .dockerignore
├── .gitignore
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment file
├── LICENSE                           # MIT License
├── DATA_POLICY.md                    # Privacy and data handling
├── DESIGN.md                         # Architecture and design decisions
├── MODEL_CARD.md                     # Model documentation and limitations
└── README.md                         # This file
```

## Running Components

### Backend ML Training
```bash
python src/train.py --config configs/experiment1.yaml
```

### API Server
```bash
python api/app.py
```

The API will be available at `http://localhost:8000`

### Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

### React Frontend (Optional)
```bash
cd frontend
npm install
npm start
```

## Testing

```bash
# Run all tests
pytest tests/ -v

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

Edit `configs/experiment1.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Feature extraction parameters
- Class weights for imbalanced data

## Adding New Species

1. Collect audio samples for the new species
2. Add entries to `data/manifest_train.csv` with the new label
3. Update the config file with the new class
4. Retrain the model: `python src/train.py --config configs/experiment1.yaml`
5. Evaluate on test set including new species

## Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Or run individual services
docker build -t bmw-backend .
docker run -p 8000:8000 bmw-backend python api/app.py
```

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
