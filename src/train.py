import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import BMWDataset
from src.models.pytorch_models import TransferLearningModel
from src.eval.confusion_and_metrics import generate_evaluation_report
from src.utils.logger import setup_logger


def train(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = setup_logger("BMW_Train")
    logger.info(f"Starting training with config: {config}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------
    # Load Train & Validation sets
    # ------------------------------
    train_dataset = BMWDataset(config['data']['manifest_train'], config)
    val_dataset = BMWDataset(config['data']['manifest_val'], config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # ------------------------------
    # Model
    # ------------------------------

    # Automatically determine number of classes
    num_classes = len(train_dataset.labels)

    model = TransferLearningModel(
        num_classes=num_classes,
        base_model=config['model']['pretrained_model'],  # matches your YAML
        pretrained=True
    ).to(device)

    # ------------------------------
    # Loss & Optimizer
    # ------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # ------------------------------
    # Training Loop
    # ------------------------------
    best_val_acc = 0.0
    checkpoint_dir = Path(config['output']['model_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        logger.info(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

        # ------------------------------
        # Validation
        # ------------------------------
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_acc = val_correct / val_total
        logger.info(f"Validation Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            logger.info("Saved best model")

    # ------------------------------
    # Final evaluation
    # ------------------------------
    logger.info("Running final evaluation on validation set...")

    eval_dir = Path(config['output']['experiment_dir']) / f"eval_{int(time.time())}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    generate_evaluation_report(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_pred_proba=np.array(all_probs),
        labels=train_dataset.labels,
        output_dir=str(eval_dir)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment1.yaml")
    args = parser.parse_args()

    train(args.config)
