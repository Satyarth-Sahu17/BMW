"""
Logging configuration for BMW application
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Remove existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
