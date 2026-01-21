import torch
import numpy as np

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

def reverse_normalize(tensor: torch.FloatTensor, clip_window = [-2000, 500]) -> torch.FloatTensor:
    """
    Reverse the normalization of a tensor from [-1, 1] back to original HU values.

    Args:
        tensor: torch.FloatTensor with values in [-1, 1]
        clip_window: Tuple[int, int] defining the original clipping window (min, max)
    
    Returns:
        torch.FloatTensor with original HU values
    """
    hu_tensor = 0.5 * (tensor + 1) * (clip_window[1] - clip_window[0]) + clip_window[0]
    return hu_tensor

def window_ct_hu_to_png(
    hu: torch.Tensor,
    center: float = -600.0,
    width: float = 1500.0,
    bit_depth: int = 8,
    return_float: bool = False,
) -> torch.Tensor:
    """
    Apply DICOM-compliant windowing to a CT HU tensor and map to [0, 2^bit_depth - 1].

    Args:
        hu: torch.Tensor or numpy.ndarray
            Tensor of HU values (any shape: 2D slice, 3D volume, batch, etc.).
        center: float
            Window center (a.k.a. level), in HU.
        width: float
            Window width, in HU.
        bit_depth: int
            Output bit depth (default 8 → range [0, 255] for PNG).
        return_float: bool
            If True, return a float tensor instead of uint8. It will never be converted to uint8

    Returns:
        torch.Tensor (dtype=torch.uint8)
            Windowed image(s) scaled to [0, 2^bit_depth - 1], suitable for PNG.
            Shape matches `hu`.
    """
    if type(hu) is np.ndarray:
        hu = torch.from_numpy(hu)
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
    if between.any().item(): #Added .item() later
        out[between] = ((hu[between] - c) / w + 0.5) * y_range + y_min

    if not return_float:
        # Clamp just in case of numeric fuzz and cast to uint8
        out = out.clamp(y_min, y_max).round().to(torch.uint8)
    else:
        out = out.clamp(y_min, y_max).float()
    return out

def prepare_for_wandb_hu(slice_2d, window = [-2000, 500], center: float = -600.0, width: float = 1500.0):
    """
    slice_2d: torch.Tensor of shape (H, W) or (1, H, W) with values in [-1, 1]
    window: list of two ints, the HU clipping window used during preprocessing
    center: float, window center for DICOM windowing
    width: float, window width for DICOM windowing
    returns: np.ndarray uint8, shape (H, W)
    """
    slice_hu = reverse_normalize(slice_2d, window)
    slice_png = window_ct_hu_to_png(slice_hu, center=center, width=width, bit_depth=8)
    slice_png = slice_png.cpu().numpy()
    return slice_png

def prepare_for_wandb(slice_2d):
    """
    slice_2d: torch.Tensor or np.ndarray, shape (H, W) or (1, H, W)
              values expected in [-1, 1]
    returns: np.ndarray uint8, shape (H, W)
    """
    if hasattr(slice_2d, "detach"):  # torch.Tensor
        slice_2d = slice_2d.detach().to(torch.float32).cpu().numpy()

    slice_2d = np.squeeze(slice_2d)          # drop channel if (1, H, W)
    slice_2d = np.clip(slice_2d, -1.0, 1.0)  # enforce range
    slice_2d = (slice_2d + 1.0) / 2.0        # [-1,1] -> [0,1]
    slice_2d = (slice_2d * 255.0).round().astype("uint8")
    return slice_2d

def volume_to_gif_frames(volume_3d, every_n: int = 1, is_hu: bool = True):
    """
    volume_3d: torch.Tensor or np.ndarray, shape (D, H, W)
               values in [-1, 1]
    every_n: use every n-th slice to keep GIF small
    returns: list of uint8 (H, W) frames
    """
    if hasattr(volume_3d, "detach"):
        volume_3d = volume_3d.detach().to(torch.float32).cpu().numpy()
    D = volume_3d.shape[0]
    frames = []
    for d in range(0, D, every_n):
        if is_hu:
            frame = prepare_for_wandb_hu(volume_3d[d])
        else:
            frame = prepare_for_wandb(volume_3d[d])
        frames.append(frame)
    return frames

