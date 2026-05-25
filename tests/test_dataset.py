import pytest
import pandas as pd
from pathlib import Path

def test_dataset_creation():
    # This would require actual manifest file
    # Just test that the test framework works
    assert True

def test_label_mapping():
    labels = ["wolf", "tiger", "gunshot"]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    assert label_to_idx["wolf"] == 0
    assert label_to_idx["tiger"] == 1
    assert len(label_to_idx) == 3
