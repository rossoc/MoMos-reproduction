from typing import Optional, Any
import torch
import os


def load_model_from_checkpoint(checkpoint_path: str) -> Optional[dict[str, Any]]:
    """
    Load model from local checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with state_dict and config
    """
    try:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        return {"state_dict": state_dict, "config": config}
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
