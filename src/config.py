# Configuration module
import os
from datetime import datetime
from pathlib import Path


def get_default_data_root() -> str:
    """Get default data root directory."""
    return "./datasets/MVTecAD"


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def now_run_id() -> str:
    """Generate a run ID based on current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_device(device: str) -> str:
    """
    Resolve device string to actual device.
    
    Args:
        device: "auto", "cpu", or "cuda"
        
    Returns:
        "cpu" or "cuda"
    """
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    elif device == "cuda":
        return "cuda"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cpu', or 'cuda'")
