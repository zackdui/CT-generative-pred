import os
from typing import Dict, Sequence, Callable, Optional, Tuple, List, Any, Iterable, Union
import numpy as np
import nibabel as nib
import pydicom
import ants
import pandas as pd
import json
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

# def moving_to_fixed_world(points_xyz: np.ndarray, fwd_mat_path: str) -> np.ndarray:
#     """
#     points_xyz: (N, 3) world coords in MOVING space.
#     returns:    (N, 3) world coords in FIXED space.
#     """
#     df = pd.DataFrame(points_xyz, columns=["x", "y", "z"])
#     df_t = ants.apply_transforms_to_points(
#         dim=3,
#         points=df,
#         transformlist=[fwd_mat_path],
#         whichtoinvert=[True],   # <-- invert the affine for points
#     )
#     return df_t[["x", "y", "z"]].to_numpy()

# def parse_ants_rigid_transform(mat_path: str):
#     """
#     Parse an ANTs/ITK rigid transform (.mat) produced by ants.registration
#     with type_of_transform='Rigid'.

#     Returns:
#         R: (3,3) rotation matrix
#         T: (3,) translation vector
#         c: (3,) center of rotation
#     """
#     with open(mat_path, "r") as f:
#         lines = f.readlines()

#     params_line = None
#     fixed_line  = None
#     for line in lines:
#         if line.startswith("Parameters:"):
#             params_line = line
#         elif line.startswith("FixedParameters:"):
#             fixed_line = line

#     if params_line is None or fixed_line is None:
#         raise ValueError(f"Could not find Parameters / FixedParameters in {mat_path}")

#     # Parse Parameters: R (9 values) + T (3 values)
#     params = [float(x) for x in params_line.replace("Parameters:", "").split()]
#     if len(params) != 12:
#         raise ValueError(f"Expected 12 parameters, got {len(params)} in {mat_path}")
#     R_flat = params[:9]
#     T = np.array(params[9:], dtype=float)  # (3,)

#     R = np.array(R_flat, dtype=float).reshape(3, 3)  # row-major

#     # Parse FixedParameters: center c (3 values)
#     fixed = [float(x) for x in fixed_line.replace("FixedParameters:", "").split()]
#     if len(fixed) != 3:
#         raise ValueError(f"Expected 3 fixed parameters, got {len(fixed)} in {mat_path}")
#     c = np.array(fixed, dtype=float)

#     return R, T, c
# def moving_to_fixed_world_rigid(points_xyz: np.ndarray, mat_path: str) -> np.ndarray:
#     """
#     points_xyz: (N, 3) world coords in MOVING space (LPS).
#     returns:    (N, 3) world coords in FIXED space (LPS).

#     Using the rigid transform encoded in mat_path.
#     """
#     R, T, c = parse_ants_rigid_transform(mat_path)
#     pts = np.asarray(points_xyz, dtype=float)  # (N,3)

#     # y = R @ (x - c) + c + T
#     # shape-safe broadcast
#     return (pts - c) @ R.T + c + T

# def map_bbox_via_ants_points(
#     bbox,
#     moving_img: ants.ANTsImage,
#     fixed_img: ants.ANTsImage,
#     fwd_transform_path: str,
# ) -> np.ndarray:
#     """
#     bbox: [i_min, i_max, j_min, j_max, k_min, k_max] in MOVING image index convention:
#           i = x (cols), j = y (rows), k = z (slices)

#     Returns bbox in FIXED image index convention (same format).
#     """
#     bbox = np.asarray(bbox, dtype=float)
#     i_min, i_max, j_min, j_max, k_min, k_max = bbox

#     # 1. Build corners in MOVING **index** space.
#     # ANTs index is (row, col, slice) = (y, x, z) = (j, i, k).
#     corners_idx_moving = [
#         (int(j_min), int(i_min), int(k_min)),
#         (int(j_min), int(i_min), int(k_max)),
#         (int(j_min), int(i_max), int(k_min)),
#         (int(j_min), int(i_max), int(k_max)),
#         (int(j_max), int(i_min), int(k_min)),
#         (int(j_max), int(i_min), int(k_max)),
#         (int(j_max), int(i_max), int(k_min)),
#         (int(j_max), int(i_max), int(k_max)),
#     ]

#     # 2. MOVING indices -> MOVING physical points (world coords) via ANTs
#     pts_world_moving = [
#         ants.transform_index_to_physical_point(moving_img, idx)
#         for idx in corners_idx_moving
#     ]  # list of (x,y,z) in LPS
#     pts_world_moving = np.array(pts_world_moving)

#     # 3. Apply the registration transform in WORLD space
#     #    fwd_transform is moving -> fixed, so we DO NOT invert.
#     df = pd.DataFrame(pts_world_moving, columns=["x", "y", "z"])
#     df_t = ants.apply_transforms_to_points(
#         dim=3,
#         points=df,
#         transformlist=[fwd_transform_path],
#         whichtoinvert=[True],
#     )
#     pts_world_fixed = df_t[["x", "y", "z"]].to_numpy()

#     # 4. FIXED physical points -> FIXED indices (row, col, slice)
#     idx_fixed = [
#         ants.transform_physical_point_to_index(fixed_img, tuple(pt))
#         for pt in pts_world_fixed
#     ]  # list of (row, col, slice)
#     idx_fixed = np.array(idx_fixed, dtype=float)

#     # 5. Take min/max over corners to get bbox in FIXED index space
#     row_min, col_min, slice_min = np.floor(idx_fixed.min(axis=0)).astype(int)
#     row_max, col_max, slice_max = np.ceil(idx_fixed.max(axis=0)).astype(int)

#     # 6. Convert back to your bbox convention: [i_min, i_max, j_min, j_max, k_min, k_max]
#     # row = j, col = i, slice = k
#     i_min_new, i_max_new = col_min, col_max
#     j_min_new, j_max_new = row_min, row_max
#     k_min_new, k_max_new = slice_min, slice_max

#     return np.array(
#         [i_min_new, i_max_new, j_min_new, j_max_new, k_min_new, k_max_new],
#         dtype=int,
#     )

# def map_bbox_between_geometries_try(
#     bbox: Sequence[int],
#     geom_src: Dict,
#     geom_tgt: Dict,
#     fwd_mat_path: Optional[str] = None,
#     crop_lower: Optional[Tuple[int, int, int]] = None,
#     pad_lower: Optional[Tuple[int, int, int]] = None,
# ) -> np.ndarray:
#     """
#     Map a 3D bounding box from a source image space to a target image space.

#     bbox:
#         [x_min, x_max, y_min, y_max, z_min, z_max] in SOURCE index space
#         where:
#             x = columns (axis 1 of slice)
#             y = rows    (axis 0 of slice)
#             z = slices  (axis 2 / through-plane)

#     geom_src, geom_tgt:
#         Dicts with:
#         - "origin":    (3,) np.array
#         - "spacing":   (3,) np.array
#         - "direction": (3, 3) np.array
#         - "shape":     tuple (ny, nx, nz) or similar

#     Returns:
#         np.array([x_min_new, x_max_new,
#                   y_min_new, y_max_new,
#                   z_min_new, z_max_new])
#     """
#     bbox = np.asarray(bbox, dtype=float)
#     x_min, x_max, y_min, y_max, z_min, z_max = bbox

#     # 1. Build the 8 bbox corner indices in SOURCE index space
#     #    Array indexing is [y, x, z] = [row, col, slice].
#     corners_idx_src = np.array([
#         [y_min, x_min, z_min],
#         [y_min, x_min, z_max],
#         [y_min, x_max, z_min],
#         [y_min, x_max, z_max],
#         [y_max, x_min, z_min],
#         [y_max, x_min, z_max],
#         [y_max, x_max, z_min],
#         [y_max, x_max, z_max],
#     ], dtype=float)  # shape (8, 3) = [row, col, slice]

#     # 2. SOURCE indices -> SOURCE world coordinates
#     spacing_src   = np.asarray(geom_src["spacing"], dtype=float)    # (3,)
#     direction_src = np.asarray(geom_src["direction"], dtype=float)  # (3,3)
#     origin_src    = np.asarray(geom_src["origin"], dtype=float)     # (3,)

#     # scaled_idx_src: (8,3) * (3,) -> (8,3)
#     scaled_idx_src = corners_idx_src * spacing_src
#     # world = origin + direction @ (idx * spacing)
#     world_src = origin_src + (direction_src @ scaled_idx_src.T).T   # (8,3)

#     # 3. Apply world-space transform (rigid, etc.) if provided
#     if fwd_mat_path is not None:
#         world_tgt = moving_to_fixed_world(world_src, fwd_mat_path)  # (8,3)
#     else:
#         world_tgt = world_src

#     idx_min = np.floor(world_tgt.min(axis=0)).astype(int)
#     idx_max = np.ceil(world_tgt.max(axis=0)).astype(int)
#     print("world_tgt corners:", world_tgt)
#     print("idx_min:", idx_min)
#     print("idx_max:", idx_max)
#     # 4. TARGET world -> TARGET indices
#     spacing_tgt   = np.asarray(geom_tgt["spacing"], dtype=float)
#     direction_tgt = np.asarray(geom_tgt["direction"], dtype=float)
#     origin_tgt    = np.asarray(geom_tgt["origin"], dtype=float)

#     rel_tgt = (world_tgt - origin_tgt)  # (8,3)
#     D_inv = np.linalg.inv(direction_tgt)
#     unscaled_idx_tgt = (D_inv @ rel_tgt.T).T       # (8,3) in [row, col, slice]
#     idx_tgt_float = unscaled_idx_tgt / spacing_tgt # (8,3)

#     # 5. Integer bbox in TARGET index space
#     idx_min = np.floor(idx_tgt_float.min(axis=0)).astype(int)
#     idx_max = np.ceil(idx_tgt_float.max(axis=0)).astype(int)

#     # idx_* are [row_min, col_min, slice_min] = [y_min, x_min, z_min]
#     row_min, col_min, slice_min = idx_min.tolist()
#     row_max, col_max, slice_max = idx_max.tolist()

#     # 6. Adjust for crop / pad in TARGET index space (still [y,x,z])
#     if crop_lower is not None:
#         crop_lower = np.asarray(crop_lower, dtype=int)  # [dy, dx, dz]
#         row_min -= crop_lower[0]
#         row_max -= crop_lower[0]
#         col_min -= crop_lower[1]
#         col_max -= crop_lower[1]
#         slice_min -= crop_lower[2]
#         slice_max -= crop_lower[2]

#     if pad_lower is not None:
#         pad_lower = np.asarray(pad_lower, dtype=int)  # [py, px, pz]
#         row_min += pad_lower[0]
#         row_max += pad_lower[0]
#         col_min += pad_lower[1]
#         col_max += pad_lower[1]
#         slice_min += pad_lower[2]
#         slice_max += pad_lower[2]

#     # 7. Convert back to bbox format [x_min, x_max, y_min, y_max, z_min, z_max]
#     x_min_new, x_max_new = col_min, col_max
#     y_min_new, y_max_new = row_min, row_max
#     z_min_new, z_max_new = slice_min, slice_max

#     return np.array(
#         [x_min_new, x_max_new,
#          y_min_new, y_max_new,
#          z_min_new, z_max_new],
#         dtype=int,
#     )

# def map_bbox_between_geometries(
#     bbox: Sequence[int],
#     geom_src: Dict,
#     geom_tgt: Dict,
#     fwd_mat_path: Optional[str] = None,
#     crop_lower: Optional[Tuple[int, int, int]] = None,
#     pad_lower: Optional[Tuple[int, int, int]] = None,
# ) -> np.ndarray:
#     """
#     Map a 3D bounding box from a source image space to a target image space.

#     bbox:
#         [i_min, i_max, j_min, j_max, k_min, k_max] in SOURCE index space
#         (0-based indices).

#     geom_src, geom_tgt:
#         Dicts from get_geometry_from_nifti / get_geometry_from_dicoms:
#         - "origin":    (3,) np.array
#         - "spacing":   (3,) np.array
#         - "direction": (3, 3) np.array
#         - "shape":     tuple

#     crop_lower:
#         Optional (cx, cy, cz) 3-tuple of voxels cropped from the *low* side
#         of each axis AFTER the target image is defined (i.e., post-registration).

#     pad_lower:
#         Optional (px, py, pz) 3-tuple of voxels padded on the *low* side
#         of each axis AFTER cropping.

#     Returns:
#         np.array([i_min_new, i_max_new, j_min_new, j_max_new, k_min_new, k_max_new])
#     """
#     bbox = np.asarray(bbox, dtype=float)
#     i_min, i_max, j_min, j_max, k_min, k_max = bbox

#     # 1. Build the 8 bbox corner indices in SOURCE index space
#     corners_idx_src = np.array([
#         [i_min, j_min, k_min],
#         [i_min, j_min, k_max],
#         [i_min, j_max, k_min],
#         [i_min, j_max, k_max],
#         [i_max, j_min, k_min],
#         [i_max, j_min, k_max],
#         [i_max, j_max, k_min],
#         [i_max, j_max, k_max],
#     ], dtype=float)  # shape (8, 3)

#     world_tgt = moving_to_fixed_world(corners_idx_src, fwd_mat_path)
#     idx_min = np.floor(world_tgt.min(axis=0)).astype(int)
#     idx_max = np.ceil(world_tgt.max(axis=0)).astype(int)
#     print("world_tgt corners:", world_tgt)
#     print("idx_min:", idx_min)
#     print("idx_max:", idx_max)
#     # 2. Convert SOURCE indices -> SOURCE world coordinates
#     # x = origin + direction @ (idx * spacing)
#     spacing_src = np.asarray(geom_src["spacing"], dtype=float)
#     direction_src = np.asarray(geom_src["direction"], dtype=float)
#     origin_src = np.asarray(geom_src["origin"], dtype=float)

#     scaled_idx_src = corners_idx_src * spacing_src  # (8, 3) * (3,) -> (8, 3)
#     world_src = origin_src + (direction_src @ scaled_idx_src.T).T  # (8, 3)

#     # 3. Apply world-space transform (rigid, etc.) if provided
#     if fwd_mat_path is not None:
#         world_tgt = moving_to_fixed_world_rigid(world_src, fwd_mat_path)  # (8, 3)
#     else:
#         world_tgt = world_src

#     idx_min = np.floor(world_tgt.min(axis=0)).astype(int)
#     idx_max = np.ceil(world_tgt.max(axis=0)).astype(int)
#     print("world_tgt corners:", world_tgt)
#     print("idx_min:", idx_min)
#     print("idx_max:", idx_max)
#     # 4. Convert TARGET world coordinates -> TARGET indices (float)
#     spacing_tgt = np.asarray(geom_tgt["spacing"], dtype=float)
#     direction_tgt = np.asarray(geom_tgt["direction"], dtype=float)
#     origin_tgt = np.asarray(geom_tgt["origin"], dtype=float)

#     # Solve direction_tgt @ (idx * spacing_tgt) = world_tgt - origin_tgt
#     rel_tgt = (world_tgt - origin_tgt)  # (8, 3)
#     # For each point: sol = direction_tgt^{-1} @ rel_tgt
#     D_inv = np.linalg.inv(direction_tgt)
#     unscaled_idx_tgt = (D_inv @ rel_tgt.T).T  # (8, 3)
#     idx_tgt_float = unscaled_idx_tgt / spacing_tgt  # (8, 3)

#     # 5. Get integer bbox in TARGET space by min/max over corners
#     idx_min = np.floor(idx_tgt_float.min(axis=0)).astype(int)
#     idx_max = np.ceil(idx_tgt_float.max(axis=0)).astype(int)

#     # 6. Adjust for crop / pad if they were applied AFTER target image
#     if crop_lower is not None:
#         crop_lower = np.asarray(crop_lower, dtype=int)
#         idx_min -= crop_lower
#         idx_max -= crop_lower

#     if pad_lower is not None:
#         pad_lower = np.asarray(pad_lower, dtype=int)
#         idx_min += pad_lower
#         idx_max += pad_lower

#     # 7. Optionally clamp into target shape
#     # if clamp_to_target and "shape" in geom_tgt:
#     #     shape = np.asarray(geom_tgt["shape"], dtype=int)
#     #     shape = np.asarray([512, 512, 208], dtype=int)  # only first 3 dims
#     #     print("Target shape:", shape)
#     #     # assuming same axis order between spacing/direction and shape
#     #     idx_min = np.maximum(idx_min, 0)
#     #     idx_max = np.minimum(idx_max, shape - 1)

#     # Pack back into [i_min, i_max, j_min, j_max, k_min, k_max]
#     i_min_new, j_min_new, k_min_new = idx_min.tolist()
#     i_max_new, j_max_new, k_max_new = idx_max.tolist()

#     return np.array([
#         i_min_new, i_max_new,
#         j_min_new, j_max_new,
#         k_min_new, k_max_new,
#     ], dtype=int)


# def map_bbox_from_rows(full_row, coords, crop_pad_size=None, fixed_geometry=None, fwd_mat_path=None):
#     """
#     full_row: a row from the registrations DataFrame
#     coords:   [i_min, i_max, j_min, j_max, k_min, k_max] in MOVING index space
#     crop_pad_size: optional (tx, ty, tz) target size after crop/pad
#     """
#     if full_row["has_nifti"] == True:
#         moving_geometry = get_geometry_from_nifti(full_row["nifti_path"])
#     else:
#         moving_geometry = get_geometry_from_dicoms(json.loads(full_row["sorted_paths"]))
#     print("Moving geometry:", moving_geometry)
#     # fixed_geometry["spacing"] = np.array([0.703125, 0.703125, 2.5], dtype=float)
#     # fixed_geometry = {"origin": full_row["fixed_origin"],
#     #                   "spacing": full_row["fixed_spacing"],
#     #                   "direction": full_row["fixed_direction"].reshape((3, 3)),
#     #                   "shape": full_row["fixed_shape"]}
#     print("Fixed geometry:", fixed_geometry)
#     # fwd_mat_path = full_row["registration_file"]
#     if crop_pad_size is not None:
#         crop_lower, pad_lower = compute_centered_crop_pad_lower(fixed_geometry["shape"], crop_pad_size)
#     else:
#         crop_lower, pad_lower = None, None
#     print("pad_lower:", pad_lower)
#     transformed_box = map_bbox_between_geometries(coords, 
#                                                   moving_geometry, 
#                                                   fixed_geometry, 
#                                                   fwd_mat_path, 
#                                                   crop_lower, 
#                                                   pad_lower)
#     return transformed_box

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

    
