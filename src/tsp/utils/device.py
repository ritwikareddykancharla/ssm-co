# --------------------------------------------------
# Unified device utilities for CPU / CUDA / TPU
# - Safe on Kaggle (no torch_xla import unless TPU)
# - Correct synchronization for timing
# --------------------------------------------------

import torch


def get_device(device: str):
    """
    Returns the appropriate torch device handle.

    Args:
        device (str): 'cpu', 'cuda', or 'tpu'

    Returns:
        torch.device or xm.xla_device()
    """
    device = device.lower()

    if device == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError as e:
            raise RuntimeError(
                "TPU selected but torch_xla is not installed."
            ) from e

    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device("cpu")


def optimizer_step(optimizer, device: str):
    """
    Performs the correct optimizer step depending on device.
    """
    device = device.lower()

    if device == "tpu":
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()


def sync(device: str):
    """
    Synchronizes device for correct timing / logging.
    """
    device = device.lower()

    if device == "tpu":
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    elif device == "cuda":
        torch.cuda.synchronize()
