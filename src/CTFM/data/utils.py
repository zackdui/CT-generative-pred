# Portions of this file are adapted from:
# pgmikhael/SybilX (MIT License)
# Copyright (c) 2021 Peter Mikhael & Jeremy Wohlwend
#
# Modifications (c) 2025 Zack Duitz

import os
import pydicom
from pydicom.pixels import apply_modality_lut
import numpy as np
import nibabel as nib
import torchio as tio
import torch
import pickle
import ants
import json
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple


# Error Messages
METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"
LOADING_ERROR = "LOADING ERROR! {}"

DEVICE_ID = {
    "GE MEDICAL SYSTEMS": 0,
    "Philips": 1,
    "PHIs": 1,
    "SIEMENS": 2,
    "Siemens Healthcare": 2,  # note: same id as SIEMENS
    "TOSHIBA": 3,
    "Vital Images, Inc.": 4,
    "Hitachi Medical Corporation": 5,
    "LightSpeed16": 6,
    -1: 7,
}

def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine

def _slice_spacing(sorted_datasets):
    if len(sorted_datasets) > 1:
        slice_positions = _slice_positions(sorted_datasets)
        slice_positions_diffs = np.diff(slice_positions)
        return np.median(slice_positions_diffs)

    return getattr(sorted_datasets[0], "SpacingBetweenSlices", 0)


def _slice_positions(sorted_datasets):
    image_orientation = sorted_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in sorted_datasets]

def _ijk_to_patient_xyz_transform_matrix(sorted_datasets):
    first_dataset = sorted_datasets[0]
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(sorted_datasets)

    transform = np.identity(4, dtype=np.float32)
    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing
    transform[:3, 3] = first_dataset.ImagePositionPatient
    return transform

def pydicom_to_nifti(paths, output_path, return_nifti=False, save_nifti=True):
    slices = [pydicom.dcmread(p) for p in paths]
    image = np.stack(
        [s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in slices], -1
    )  # y, x, z
    if return_nifti or save_nifti:
        affine = _ijk_to_patient_xyz_transform_matrix(slices)
        nifti_img = nib.Nifti1Image(np.transpose(image, (1, 0, 2)), affine)  # (x, y, z)
    if save_nifti:
        nib.save(nifti_img, output_path)
    if return_nifti:
        return image, nifti_img
    return image

def inspect_nifti_basic(path: str) -> None:
    """
    Inspect a NIfTI file and print out basic information.
    Parameters:
    -----------
    path : str
        Path to the NIfTI file.
    -----------
    """
    img = nib.load(path)
    hdr = img.header

    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 ** 2)

    shape = img.shape
    zooms = hdr.get_zooms()  # voxel spacing
    dtype = hdr.get_data_dtype()
    # scaling_applied = img.get_fdata() 
    raw = np.asanyarray(img.dataobj)
    scaled = img.get_fdata()

    print(raw.min(), raw.max())
    print(scaled.min(), scaled.max())
    print("scl_slope:", hdr["scl_slope"])
    print("scl_inter:", hdr["scl_inter"]) 

    print(f"{os.path.basename(path)}")
    print(f"  shape: {shape}")
    print(f"  voxel size (mm): {zooms}")
    print(f"  dtype: {dtype}")
    print(f"  file size: {size_mb:.2f} MB")
    print()

def inspect_with_torchio(path: str) -> None:
    """
    Inspect a CT scan stored in NIfTI format using TorchIO.
    Prints out shape, spacing, data type, and basic HU statistics.
    
    Parameters:
    -----------
    path : str
        Path to the NIfTI file.
    -----------
    """
    # Load with TorchIO directly from file
    subject = tio.Subject(ct=tio.ScalarImage(path))
    img = subject.ct

    data = img.data  # (1, X, Y, Z)
    print(f"Original shape (C, X, Y, Z): {tuple(data.shape)}")
    print(f"Original spacing (sx, sy, sz) [mm]: {img.spacing}")
    print("original data type:", data.dtype)

    # Ensure float32
    data = data.float()
    img.set_data(data)

    # HU stats
    hu_np = data.numpy().astype(np.float32)
    hu_min = float(hu_np.min())
    hu_max = float(hu_np.max())
    hu_mean = float(hu_np.mean())
    p1, p50, p99 = np.percentile(hu_np, [1, 50, 99])

    print("\n=== HU Statistics (float32) ===")
    print(f"min:  {hu_min:.1f}")
    print(f"p1:   {p1:.1f}")
    print(f"p50:  {p50:.1f}")
    print(f"p99:  {p99:.1f}")
    print(f"max:  {hu_max:.1f}")
    print(f"mean: {hu_mean:.1f}")

    if hu_min < -2000 or hu_max > 4000:
        print("\n[WARN] HU range looks unusual for CT. "
              "Check that RescaleSlope/Intercept were applied correctly.")
    else:
        print("\n[OK] HU range looks reasonable for CT.")


# Dicom loader with only ToTensor augmentation
# Additionally all helpers needed to load the dicoms properly in this format

def apply_windowing(image, center, width, bit_size=16):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460
    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    y_min = 0
    y_max = 2**bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    below = image <= (c - w / 2)  # pixels to be set as black
    above = image > (c + w / 2)  # pixels to be set as white
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image

def fix_repeated_shared(path: str) -> str:
    """
    If there are two 'shared' segments in the path, remove everything up to (and including)
    the second 'shared', and prepend '/data/rbg/' back to make a clean path.

    Example:
      '/data/rbg/shared/...batch1shared/datasets/...'
      -> '/data/rbg/shared/datasets/...'
    """
    parts = path.split("shared")
    if len(parts) < 3:
        return path  # no duplicate 'shared'

    # everything after the second 'shared'
    suffix = "shared".join(parts[2:])
    # prepend '/data/rbg/' and 'shared/' back
    return "/data/rbg/shared" + suffix

def transform_to_hu(dcm):
    """Transform DICOM pixel array to Hounsfield units
    Args:
        dcm (pydicom Dataset): dcm object read with pydicom

    Returns:
        np.array: numpy array of the DICOM image in Hounsfield
    """
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    hu_image = dcm.pixel_array * slope + intercept
    return hu_image

class DicomLoader:
    def __init__(self, args):
        """
        Minimal DICOM loader: no caching, only augmentation is ToTensor.
        """
        self.args = args
        self.window_center = -600
        self.window_width = 1500

    def get_image(self, path, sample):
        """
        Parameters
        ----------
        path : str
            Path to a single DICOM file
        sample : dict
            Metadata for this sample (e.g., annotations). Currently unused.

        Returns
        -------
        dict
            {"input": torch.FloatTensor, "mask": None}
        """
       
        # Normal DICOM load
        try:
            path = fix_repeated_shared(path)  # if you still need this helper
            dcm = pydicom.dcmread(path)
            # image_position = dcm.ImagePositionPatient      # [x, y, z]
            # image_orientation = dcm.ImageOrientationPatient  # [row_x, row_y, row_z, col_x, col_y, col_z]

            # Go to HU-like values
            hu = apply_modality_lut(dcm.pixel_array, dcm)

            # Windowing (your function should take the HU array)
            arr = apply_windowing(hu, self.window_center, self.window_width)

            # Optional: keep your "8-bit parity" behavior
            arr = arr // 256

            # Convert to torch
            arr = torch.from_numpy(arr).float()

        except Exception:
            # You can be more specific here if you like
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))

        # 3. No masks for now → simplest
        return {"input": arr, "mask": None}

def get_sample_loader(split_group, args):
    return DicomLoader(args)

def get_exam_id(exam_dict):
    """
    Given an exam dictionary, compute and return the exam ID.
    Must contain keys: 'pid', 'screen_timepoint', 'series'.
    """
    pid = exam_dict["pid"]
    screen_timepoint = exam_dict["screen_timepoint"]
    series_id = exam_dict["series"]
    exam_id = int(
        "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )
    )
    return exam_id

def ants_crop_or_pad_like_torchio(img: ants.ANTsImage,
                                  target_size,
                                  pad_value=0,
                                  only_xy=False):
    """
    Mimic torchio.transforms.CropOrPad (centered) for an ANTsImage.

    Args:
        img: ants.ANTsImage, with shape (X, Y, Z) or (X, Y)
        target_size: iterable of ints, e.g. (x, y, z)
        pad_value: constant value used for padding
        only_xy: if True, only crop/pad in x and y dimensions, leave z unchanged

    Returns:
        new_img: ants.ANTsImage cropped/padded to target_size, with updated origin
    """
    data = img.numpy()
    ndim = data.ndim
    target_size = np.array(target_size, dtype=int)
    current_size = np.array(data.shape, dtype=int)

    if only_xy and ndim >= 3:
        target_size[-1] = current_size[-1]

    if target_size.shape[0] != ndim:
        raise ValueError(f"target_size must have {ndim} dims, got {target_size.shape[0]}")

    # Compute how much to crop/pad on each side, per dim
    crop_lower = np.zeros(ndim, dtype=int)
    crop_upper = np.zeros(ndim, dtype=int)
    pad_lower = np.zeros(ndim, dtype=int)
    pad_upper = np.zeros(ndim, dtype=int)

    for d in range(ndim):
        if only_xy and d == ndim - 1:
            continue  # skip z-dim if only_xy is True

        diff = target_size[d] - current_size[d]
        if diff < 0:
            # need to crop |diff| voxels in this dim, centered
            total_crop = -diff
            crop_lower[d] = total_crop // 2
            crop_upper[d] = total_crop - crop_lower[d]
        elif diff > 0:
            # need to pad diff voxels in this dim, centered
            total_pad = diff
            pad_lower[d] = total_pad // 2
            pad_upper[d] = total_pad - pad_lower[d]
        # diff == 0 -> no crop/pad

    # ---- 1) CROP ----
    slices = []
    for d in range(ndim):
        start = crop_lower[d]
        stop = current_size[d] - crop_upper[d]
        slices.append(slice(start, stop))
    cropped = data[tuple(slices)]

    # ---- 2) PAD ----
    pad_width = [(int(pad_lower[d]), int(pad_upper[d])) for d in range(ndim)]
    padded = np.pad(
        cropped,
        pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    # ---- 3) UPDATE ORIGIN (to mimic affine update in TorchIO) ----
    origin = np.array(img.origin)
    spacing = np.array(img.spacing)

    # direction is a matrix of size (2x2 or 3x3)
    direction = np.array(img.direction)

    # Cropping removes voxels from the "lower" side -> origin moves forward
    physical_shift_crop = spacing * crop_lower
    # Padding adds voxels on the lower side -> origin moves backward
    physical_shift_pad = spacing * pad_lower

    # Apply in world coordinates
    # new_origin = old_origin + R * (shift_crop - shift_pad)
    new_origin = origin + direction @ (physical_shift_crop - physical_shift_pad)

    # ---- 4) Wrap back into ANTsImage ----
    new_img = ants.from_numpy(
        padded,
        origin=new_origin.tolist(),
        spacing=img.spacing,
        direction=img.direction,
    )
    return new_img

def nib_to_ants(nifti_img: nib.Nifti1Image) -> ants.ANTsImage:
    data = nifti_img.get_fdata().astype(np.float32)  # HU as float32
    affine = nifti_img.affine                        # 4x4

    # Decompose affine into spacing / direction / origin
    RZS = affine[:3, :3]
    spacing = np.sqrt((RZS ** 2).sum(axis=0))
    spacing[spacing == 0] = 1.0

    direction = (RZS / spacing)  # 3x3
    origin = affine[:3, 3]

    ants_img = ants.from_numpy(
        data,
        origin=tuple(origin.tolist()),
        spacing=tuple(spacing.tolist()),
        direction=tuple(direction.tolist()),
    )
    return ants_img

def get_ants_image_from_row(row: dict) -> ants.ANTsImage:
    """
    Given a data row (pandas Series or dict-like) with either:
      - 'has_nifti' == True and 'nifti_path' to load from, or
      - 'has_nifti' == False and 'sorted_paths' (JSON list of DICOM paths)
    Load and return the corresponding ANTsImage.
    """
    if row['has_nifti']:
        # Load from NIfTI
        ants_img = ants.image_read(row["nifti_path"])
    else:
        paths = json.loads(row["sorted_paths"])
        out_path = os.path.join("/data/rbg/scratch/lung_ct/nlst_nifti", f"sample_{row['exam_id']}.nii.gz")
        _, nifti_image = pydicom_to_nifti(
                                                        paths,
                                                        output_path=out_path,
                                                        save_nifti=False,
                                                        return_nifti=True,
                                                    )
        ants_img = nib_to_ants(nifti_image)
    return ants_img

def build_dummy_fixed(row):
    shape     = tuple(row["fixed_shape"])
    spacing   = tuple(row["fixed_spacing"])
    origin    = tuple(row["fixed_origin"])
    direction = np.array(row["fixed_direction"]).reshape(3, 3)

    dummy = ants.from_numpy(
        np.zeros(shape, dtype=np.float32),
        spacing=spacing,
        origin=origin,
        direction=direction,
    )
    return dummy

def apply_transforms(image: ants.ANTsImage, 
                     forward_transform: str | None, 
                     dummy_fixed: ants.ANTsImage | None = None, 
                     row: dict | None = None,
                     resampling: bool = True, 
                     resampling_params: tuple = (0.703125 ,0.703125, 2.5), 
                     crop_pad: bool = True, 
                     target_size: tuple = (512, 512, 208), 
                     pad_hu: int = -1350,
                     only_xy: bool = False) -> ants.ANTsImage:
    """
    This function applies transforms to a single ants image.
    It can also resample the image before applying the transform.
    The function returns the transformed image.

    Parameters:
    - image: ants.core.image.ANTsImage
        The moving image to be transformed.
    - dummy_fixed: ants.core.image.ANTsImage
        The fixed image to which the moving image will be registered. We need the header info only here.
        That is why we use a dummy image so we don't have to load the full fixed image.
        Build this like this # row = df.loc[idx_moving_image]  and then dummy_fixed = build_dummy_fixed(row)
    - row: dict
        The data row corresponding to the moving image. Used only if dummy_fixed is None.
    - forward_transform: str | None
        The path to the forward transform file or None if no transform is to be applied.
    - resampling: bool
        Whether to resample the image before applying the transform.
    - resampling_params: tuple
        The resampling parameters (spacing in mm).
    - crop_pad: bool
        Whether to crop or pad the image to the target size after resampling.
    - target_size: tuple
        The target size for cropping or padding (nx, ny, nz).
    - pad_hu: int
        The HU value to use for padding.
    - only_xy: bool
        If True, only crop/pad in x and y dimensions, leave z unchanged.
    """
    assert dummy_fixed is not None or row is not None, "Either dummy_fixed or row must be provided."
    if dummy_fixed is None:
        dummy_fixed = build_dummy_fixed(row)

    if resampling:
        image = ants.resample_image(
            image,
            resample_params=resampling_params,
            use_voxels=False,
            interp_type=1
        )
    
    # If it is already aligned because it is the first image, skip transform application
    if forward_transform is not None:
        transformed_img = ants.apply_transforms(
            fixed=dummy_fixed,
            moving=image,
            transformlist=[forward_transform],
            interpolator="linear"
        )
    else: # If transform is None
        transformed_img = image

    if crop_pad:
        transformed_img = ants_crop_or_pad_like_torchio(
            transformed_img,
            target_size=target_size,
            pad_value=pad_hu,
            only_xy=only_xy,
        )
        
    return transformed_img

def ants_to_normalized_tensor(ants_img: ants.ANTsImage, clip_window: Tuple[int, int]) -> torch.FloatTensor:
    """
    Convert an ANTsImage to a normalized PyTorch tensor.
    The output tensor has shape (1, Z, Y, X) and values normalized to [-1, 1].

    Args:
        ants_img: ants.ANTsImage
    """
    # Shape is (X, Y, Z) -> (Z, Y, X)
    image_arr = ants_img.numpy().transpose(2, 1, 0)
    # Shape is (Z, Y, X) -> (1, Z, Y, X)
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).float()
    image_tensor = torch.clamp(image_tensor, clip_window[0], clip_window[1])
    image_tensor = 2 * (image_tensor - clip_window[0]) / (clip_window[1] - clip_window[0]) - 1
    return image_tensor

def reverse_normalize(tensor: torch.FloatTensor, clip_window: Tuple[int, int]) -> torch.FloatTensor:
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

def collate_image_meta(batch):
    """
    batch: list of (image, meta_dict)
    returns:
      images: stacked tensor
      metas:  list of meta_dicts
    """
    images, metas = zip(*batch)      # images: tuple of tensors, metas: tuple of dicts
    images = torch.stack(images, dim=0)
    metas = list(metas)
    return images, metas