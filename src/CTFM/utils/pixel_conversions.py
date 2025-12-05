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

def window_ct_hu_to_png(
    hu: torch.Tensor,
    center: float = -600.0,
    width: float = 1500.0,
    bit_depth: int = 8,
) -> torch.Tensor:
    """
    Apply DICOM-compliant windowing to a CT HU tensor and map to [0, 2^bit_depth - 1].

    Args:
        hu: torch.Tensor
            Tensor of HU values (any shape: 2D slice, 3D volume, batch, etc.).
        center: float
            Window center (a.k.a. level), in HU.
        width: float
            Window width, in HU.
        bit_depth: int
            Output bit depth (default 8 → range [0, 255] for PNG).

    Returns:
        torch.Tensor (dtype=torch.uint8)
            Windowed image(s) scaled to [0, 2^bit_depth - 1], suitable for PNG.
            Shape matches `hu`.
    """
    # Ensure float for math
    hu = hu.to(torch.float32)

    # Output range (e.g. 0–255 for 8-bit)
    y_min = 0.0
    y_max = float(2**bit_depth - 1)
    y_range = y_max - y_min

    # DICOM conventions
    c = center - 0.5
    w = width - 1.0

    # Avoid division by zero if width is pathological
    if w <= 0:
        raise ValueError(f"Window width must be > 1, got {width}")

    # Masks for regions
    below = hu <= (c - w / 2.0)
    above = hu > (c + w / 2.0)
    between = (~below) & (~above)

    # Initialize output
    out = torch.empty_like(hu, dtype=torch.float32)

    # Below window → black
    out[below] = y_min
    # Above window → white
    out[above] = y_max

    # Linear mapping for in-window values
    if between.any():
        out[between] = ((hu[between] - c) / w + 0.5) * y_range + y_min

    # Clamp just in case of numeric fuzz and cast to uint8
    out = out.clamp(y_min, y_max).round().to(torch.uint8)
    return out
