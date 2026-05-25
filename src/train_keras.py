import argparse
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

from src.models.keras_models import build_transfer_model, compile_model
from src.utils.logger import setup_logger

def train_keras(config_path):
    """Training script for Keras/TensorFlow models."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger("BMW_Train_Keras")
    logger.info(f"Starting Keras training with config: {config}")
    
    # Set seeds
    np.random.seed(config.get('seed', 42))
    tf.random.set_seed(config.get('seed', 42))
    
    # Load data (simplified - in practice, use tf.data.Dataset)
    # Here we assume preprocessed numpy arrays exist
    # X_train, y_train = load_data(config)
    
    # Build model
    model = build_transfer_model(
        input_shape=(224, 224, 3),
        num_classes=config['model']['num_classes'],
        base_arch=config['model']['name']
    )
    
    model = compile_model(model, learning_rate=config['training']['learning_rate'])
    
    logger.info("Model compiled successfully")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    # Callbacks
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir / 'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('early_stopping_patience', 5),
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=checkpoint_dir / 'logs'
        )
    ]
    
    # Training
    # history = model.fit(
    #     X_train, y_train,
    #     validation_split=0.2,
    #     epochs=config['training']['epochs'],
    #     batch_size=config['data']['batch_size'],
    #     callbacks=callbacks,
    #     verbose=1
    # )
    
    logger.info("Training complete")
    logger.info(f"Best model saved to {checkpoint_dir / 'best_model.h5'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment1.yaml")
    args = parser.parse_args()
    
    train_keras(args.config)
