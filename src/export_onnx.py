import argparse
import torch
from pathlib import Path
from src.models.pytorch_models import TransferLearningModel
from src.utils.logger import setup_logger

def export_to_onnx(model_path, output_path, num_classes=5):
    logger = setup_logger("BMW_ONNX_Export")
    logger.info(f"Loading model from {model_path}")
    
    device = torch.device("cpu")
    model = TransferLearningModel(num_classes=num_classes, base_model='resnet18')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    logger.info(f"Exporting to {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info("ONNX export complete!")
    logger.info(f"Model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/best_model.onnx")
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()
    
    export_to_onnx(args.model_path, args.output, args.num_classes)
