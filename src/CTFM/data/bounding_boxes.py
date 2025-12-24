import os
from typing import Dict, Sequence, Optional, List, Any, Iterable, Union
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
import torch


def get_geometry_from_nifti(path: str) -> Dict[str, Any]:
    """
    Load NIfTI geometry and convert it to ITK/ANTs-style LPS world coordinates.

    Returns a dict with:
      - origin:    (3,) np.array, LPS world coord of voxel (0,0,0)
      - spacing:   (3,) np.array, voxel size along each axis
      - direction: (3,3) np.array, direction cosines in LPS
      - shape:     tuple, voxel grid shape
      - affine:    (4,4) np.array, *LPS* voxel->world affine (for sanity)
    """
    img = nib.load(path)
    affine_ras = img.affine           # nibabel affine is effectively RAS

    # Flip from RAS -> LPS: X and Y axes change sign
    F = np.eye(4, dtype=float)
    F[0, 0] = -1.0
    F[1, 1] = -1.0

    affine_lps = F @ affine_ras       # now in LPS world

    RZS = affine_lps[:3, :3]
    origin = affine_lps[:3, 3]

    spacing = np.linalg.norm(RZS, axis=0)
    spacing[spacing == 0] = 1.0

    direction = RZS / spacing

    shape = img.shape

    return {
        "origin": origin,
        "spacing": spacing,
        "direction": direction,
        "shape": shape,
        "affine": affine_lps,
    }

def get_geometry_from_dicoms(paths: List[str]) -> Dict[str, Any]:
    """
    Infer 3D geometry from a list of DICOM slices (one series).

    Assumes `paths` are ALREADY sorted in the correct slice order
    (e.g., by InstanceNumber or ImagePositionPatient).

    paths: list of file paths for all slices in a single CT volume.

    Returns a dict with:
      - origin:     (3,) np.array, world coord of voxel (0, 0, 0)
      - spacing:    (3,) np.array, [s0, s1, s2]
                     s0 = along rows, s1 = along columns, s2 = slice direction
      - direction:  (3, 3) np.array, direction cosines matrix
                    columns = [row_dir, col_dir, slice_dir]
      - shape:      (rows, cols, num_slices)
    """
    if len(paths) == 0:
        raise ValueError("get_geometry_from_dicoms: paths list is empty.")

    # Read headers only (no pixel data)
    datasets = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]

    # Reference slice (first one with geometry tags)
    ref_ds = None
    for ds in datasets:
        if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
            ref_ds = ds
            break
    if ref_ds is None:
        raise ValueError("No DICOM in the list has ImagePositionPatient / ImageOrientationPatient.")

    # ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z]
    iop = np.array(ref_ds.ImageOrientationPatient, dtype=float)
    row_cos = iop[:3]
    col_cos = iop[3:]
    slice_cos = np.cross(row_cos, col_cos)

    # Pixel spacing [row_spacing, col_spacing] per DICOM standard
    pixel_spacing = np.array(ref_ds.PixelSpacing, dtype=float)
    row_spacing, col_spacing = pixel_spacing

    # Project each slice's IPP onto slice normal, in the GIVEN order
    dists = []
    ipps = []
    for ds in datasets:
        if not hasattr(ds, "ImagePositionPatient"):
            raise ValueError("All slices must have ImagePositionPatient when paths are pre-sorted.")
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        dist = float(np.dot(ipp, slice_cos))
        dists.append(dist)
        ipps.append(ipp)

    dists = np.array(dists)
    ipps = np.stack(ipps, axis=0)  # (num_slices, 3)

    num_slices = len(paths)

    # Origin = position of voxel (0,0,0): first slice's IPP in provided order
    origin = ipps[0]

    # Slice spacing: median distance between consecutive slice positions (in given order)
    if num_slices > 1:
        dzs = np.diff(dists)
        spacing_z = float(np.median(np.abs(dzs)))

        # Optional: enforce slice_cos to point "with" the slice order
        if dzs.mean() < 0:
            slice_cos = -slice_cos
    else:
        spacing_z = float(getattr(ref_ds, "SliceThickness", 1.0))

    # Spacing aligned with axes: axis 0 = rows, axis 1 = columns, axis 2 = slices
    spacing = np.array([row_spacing, col_spacing, spacing_z], dtype=float)

    # Direction cosines:
    # columns correspond to index directions (i, j, k):
    #   i (axis 0 / rows)    -> row_cos
    #   j (axis 1 / columns) -> col_cos
    #   k (axis 2 / slices)  -> slice_cos
    direction = np.column_stack([row_cos, col_cos, slice_cos])

    # Shape: (rows, cols, num_slices)
    rows = int(ref_ds.Rows)
    cols = int(ref_ds.Columns)
    shape = (rows, cols, num_slices)

    return {
        "origin": origin,
        "spacing": spacing,
        "direction": direction,
        "shape": shape,
    }

def compute_centered_crop_pad_lower(current_size, target_size):
    """
    Compute how many voxels are cropped/padded on the *lower* side of each axis
    for a centered CropOrPad operation (matching ants_crop_or_pad_like_torchio).

    Args:
        current_size: iterable of ints, e.g. (X, Y, Z)
        target_size: iterable of ints, e.g. (Tx, Ty, Tz)

    Returns:
        crop_lower: tuple[int], voxels cropped from the low side per dim.
        pad_lower:  tuple[int], voxels padded on the low side per dim.
    """
    current_size = np.array(current_size, dtype=int)
    target_size = np.array(target_size, dtype=int)

    if current_size.shape != target_size.shape:
        raise ValueError(
            f"current_size and target_size must have same ndim, "
            f"got {current_size.shape[0]} vs {target_size.shape[0]}"
        )

    ndim = current_size.shape[0]


    crop_lower = np.zeros(ndim, dtype=int)
    pad_lower = np.zeros(ndim, dtype=int)

    for d in range(ndim):
        diff = target_size[d] - current_size[d]

        if diff < 0:
            # cropping |diff| voxels total, centered -> floor goes to lower side
            total_crop = -diff
            crop_lower[d] = total_crop // 2
        elif diff > 0:
            # padding diff voxels total, centered -> floor goes to lower side
            total_pad = diff
            pad_lower[d] = total_pad // 2
        # diff == 0 -> no crop/pad

    return tuple(crop_lower.tolist()), tuple(pad_lower.tolist())


################ Now saving the bounding boxes ##################


def save_bbox_slices(
    volume: Union[np.ndarray, torch.Tensor],
    bbox: Sequence[int],
    slice_indices: Optional[Iterable[int]] = None,
    output_dir: str = "./test_bbox",
    prefix: str = "transformed",
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Save 2D slices from a 3D volume with a 3D bounding box drawn on each slice.

    Assumptions:
        - volume shape: (Z, H, W)
        - bbox order: [i_min, i_max, j_min, j_max, k_min, k_max]
          where:
            i = x (cols), j = y (rows), k = z (slice index)

    Args
    ----
    volume:
        3D array (Z, H, W) or torch.Tensor with that shape.
    bbox:
        [i_min, i_max, j_min, j_max, k_min, k_max] in volume index space.
    slice_indices:
        Iterable of k-indices to visualize.
        If None, defaults to range(k_min, k_max + 1) from the bbox.
    output_dir:
        Directory where PNGs will be saved.
    prefix:
        Prefix for output filenames, e.g. "orig" or "transformed".
    cmap:
        Matplotlib colormap for the slices.
    vmin, vmax:
        Optional intensity limits for imshow. If None, matplotlib auto-scales.

    Output
    ------
    Saves PNG files like:
        {output_dir}/{prefix}_z{kk:03d}.png
    """
    # Convert torch.Tensor -> numpy if needed
    if torch is not None and isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    volume = np.asarray(volume)
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]  # squeeze channel dim

    if volume.ndim != 3:
        raise ValueError(f"Expected volume of shape (Z, H, W), got {volume.shape}")

    Z, H, W = volume.shape

    # Unpack bbox
    bbox = np.array(bbox, dtype=int)
    if bbox.shape[0] != 6:
        raise ValueError(f"bbox must have 6 elements, got {bbox}")
    i_min, i_max, j_min, j_max, k_min, k_max = bbox

    # Clamp bbox to valid range (just in case)
    assert i_min >= 0 and i_max < W and j_min >= 0 and j_max < H and k_min >= 0 and k_max < Z, \
        "Warning: bbox is out of volume bounds, needs clamping."

    # i_min = max(i_min, 0)
    # j_min = max(j_min, 0)
    # k_min = max(k_min, 0)
    # i_max = min(i_max, W - 1)
    # j_max = min(j_max, H - 1)
    # k_max = min(k_max, Z - 1)

    # Default slice range = bbox z-span
    if slice_indices is None:
        slice_indices = range(k_min, k_max + 1)
    else:
        slice_indices = list(slice_indices)

    os.makedirs(output_dir, exist_ok=True)

    for k in slice_indices:
        if k < 0 or k >= Z:
            continue

        slice_img = volume[k]  # shape (H, W)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(slice_img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Slice k={k}")
        ax.axis("off")

        # Draw bbox on this slice if it intersects this k
        if k_min <= k <= k_max:
            # Rectangle: top-left (i_min, j_min), width, height
            from matplotlib.patches import Rectangle

            width = j_max - j_min + 1
            height = i_max - i_min + 1


            rect = Rectangle(
                (j_min, i_min),
                width,
                height,
                linewidth=1.5,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        # Optional colorbar if you want
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        out_path = os.path.join(output_dir, f"{prefix}_z{k:03d}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    
