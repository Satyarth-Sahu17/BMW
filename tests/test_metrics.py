import pytest
import numpy as np
from src.eval.confusion_and_metrics import Evaluator

def test_evaluator_metrics():
    y_true = [0, 1, 0, 1, 2]
    # Mock probs: 3 classes
    y_pred_probs = np.array([
        [0.9, 0.1, 0.0], # 0
        [0.2, 0.7, 0.1], # 1
        [0.8, 0.1, 0.1], # 0 (Correct)
        [0.3, 0.6, 0.1], # 1 (Correct)
        [0.1, 0.2, 0.7]  # 2 (Correct)
    ])
    labels = ["A", "B", "C"]
    
    evaluator = Evaluator(y_true, y_pred_probs, labels, out_dir="tests/output")
    metrics = evaluator.compute_metrics()
    
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_macro'] == 1.0
    assert 'per_class' in metrics
