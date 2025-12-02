import torch

# ---------- Basic range conversions ----------

def to_01_from_255(x: torch.Tensor):
    """[0, 255] → [0, 1]"""
    return x / 255.0

def to_neg1_1_from_255(x: torch.Tensor):
    """[0, 255] → [-1, 1]"""
    return (x / 127.5) - 1.0

def to_01_from_neg1_1(x: torch.Tensor):
    """[-1, 1] → [0, 1]"""
    return (x + 1.0) / 2.0

def to_255_from_neg1_1(x: torch.Tensor):
    """[-1, 1] → [0, 255]"""
    return ((x + 1.0) * 127.5).clamp(0, 255)

def to_neg1_1_from_01(x: torch.Tensor):
    """[0, 1] → [-1, 1]"""
    return (x * 2.0) - 1.0

def to_255_from_01(x: torch.Tensor):
    """[0, 1] → [0, 255]"""
    return (x * 255.0).clamp(0, 255)