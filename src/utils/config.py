"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
