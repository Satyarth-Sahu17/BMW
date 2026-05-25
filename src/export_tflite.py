import argparse
import tensorflow as tf
from pathlib import Path
from src.utils.logger import setup_logger

def export_to_tflite(model_path, output_path):
    logger = setup_logger("BMW_TFLite_Export")
    logger.info(f"Loading Keras model from {model_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    logger.info(f"TFLite model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/best_model.tflite")
    args = parser.parse_args()
    
    export_to_tflite(args.model_path, args.output)
