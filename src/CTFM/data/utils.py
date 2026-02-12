# Some functions in this file are adapted from:
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
import torch.nn.functional as F
from pathlib import Path
import tempfile
import imageio.v2 as imageio
import ants
import json
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable
import itk
import math
import matplotlib.pyplot as plt


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


def pydicom_to_nifti(paths, output_path, return_nifti=False, save_nifti=True):
    """
    Reads DICOMs using pydicom/numpy, but saves using ITK to ensure
    perfect compatibility with ANTsPy.
    """

    # --- 1. Load Data into Numpy ---
    # We assume 'paths' is already sorted by Z-position
    slices = [pydicom.dcmread(p) for p in paths]

    # Create volume (Z, Y, X)
    # Note: ITK python wraps numpy as (Z, Y, X), so this matches perfectly.
    volume = np.stack(
        [
            s.pixel_array.astype(np.float32) * s.RescaleSlope + s.RescaleIntercept
            for s in slices
        ],
        axis=0,
    )

    # --- 2. Convert to ITK Image ---
    # This creates a wrapper, avoiding memory duplication
    itk_image = itk.image_view_from_array(volume)

    # --- 3. Extract Geometry (LPS) ---
    first_ds = slices[0]
    last_ds = slices[-1]

    # Spacing
    # DICOM PixelSpacing is [RowSpacing (Y), ColSpacing (X)]
    spacing_y, spacing_x = first_ds.PixelSpacing

    # Calculate Z-spacing/direction using the full stack extent (handles Gantry Tilt)
    # We do NOT use 'SliceThickness' or cross-products here.
    pos_first = np.array(first_ds.ImagePositionPatient, dtype=float)
    pos_last = np.array(last_ds.ImagePositionPatient, dtype=float)
    n_slices = len(slices)

    # Total vector from first to last slice
    stack_vector = pos_last - pos_first

    # The average step vector between slices
    step_vector = stack_vector / (n_slices - 1)

    # The magnitude of the step is the Z-spacing
    spacing_z = np.linalg.norm(step_vector)

    # Set Spacing (X, Y, Z)
    itk_image.SetSpacing([float(spacing_x), float(spacing_y), float(spacing_z)])

    # Origin (X, Y, Z) - The center of the first voxel
    itk_image.SetOrigin([float(x) for x in pos_first])

    # --- 4. Build Direction Matrix ---
    # ITK Direction is a 3x3 Matrix. Columns are the axis vectors.
    # Columns must be normalized (unit length).

    iop = np.array(first_ds.ImageOrientationPatient, dtype=float)
    row_cosines = iop[:3]  # X-axis orientation
    col_cosines = iop[3:]  # Y-axis orientation

    # Z-axis orientation (Normalized step vector)
    slice_cosines = step_vector / spacing_z

    # Construct the 3x3 matrix (flattened list or numpy array)
    # Matrix = [ X_vec, Y_vec, Z_vec ] (columns)
    # But ITK setDirection expects a flat list or matrix in row-major order?
    # PyITK expects: [[xx, yx, zx], [xy, yy, zy], [xz, yz, zz]]

    direction_matrix = np.eye(3)
    direction_matrix[0, 0] = row_cosines[0]
    direction_matrix[1, 0] = row_cosines[1]
    direction_matrix[2, 0] = row_cosines[2]

    direction_matrix[0, 1] = col_cosines[0]
    direction_matrix[1, 1] = col_cosines[1]
    direction_matrix[2, 1] = col_cosines[2]

    direction_matrix[0, 2] = slice_cosines[0]
    direction_matrix[1, 2] = slice_cosines[1]
    direction_matrix[2, 2] = slice_cosines[2]

    # ITK python usually accepts the numpy matrix directly
    itk_image.SetDirection(direction_matrix)

    # --- 5. Save ---
    # This handles compression (.nii.gz) and RAS conversion automatically
    if save_nifti:
        itk.imwrite(itk_image, output_path)
    if return_nifti:
        return volume, itk_image
    return volume

def detect_affine_source(nifti_path, first_dicom_path):
    """
    Determines if a NIfTI file has a 'Correct' (ITK/RAS) or 'Incorrect' (Raw/LPS) affine
    by comparing it to the source DICOM.
    
    Returns:
        'ITK' (Correct, RAS)
        'NIBABEL_RAW' (Incorrect, LPS)
        'UNKNOWN' (Ambiguous)
    """
    try:
        # 1. Load DICOM Header (Stop before pixels for speed)
        dcm = pydicom.dcmread(first_dicom_path, stop_before_pixels=True)
        dcm_origin = np.array(dcm.ImagePositionPatient, dtype=float)
        
        # 2. Load NIfTI Header (Header only)
        # nibabel loads header lazily, it won't read the big image array here
        nii = nib.load(nifti_path)
        nii_affine = nii.affine
        
        # Extract NIfTI origin (4th column, first 3 rows)
        nii_origin = nii_affine[:3, 3]
        
        # --- The Check ---
        # We focus on the X-axis (index 0) and Y-axis (index 1).
        # In a proper conversion (LPS -> RAS), these signs must flip.
        
        # We use a tolerance because float conversion might introduce tiny errors
        is_x_flipped = np.isclose(nii_origin[0], -dcm_origin[0], atol=1e-3)
        is_y_flipped = np.isclose(nii_origin[1], -dcm_origin[1], atol=1e-3)
        
        is_x_same = np.isclose(nii_origin[0], dcm_origin[0], atol=1e-3)
        is_y_same = np.isclose(nii_origin[1], dcm_origin[1], atol=1e-3)

        # Logic
        if is_x_flipped and is_y_flipped:
            return "ITK" # Correctly converted to RAS
        elif is_x_same and is_y_same:
            return "NIBABEL_RAW" # Incorrect: Contains raw LPS coordinates
        else:
            # Fallback: Check Description field
            # SimpleITK often writes "Insight Toolkit" in the description
            descrip = str(nii.header.get('descrip', b''))
            if 'Insight Toolkit' in descrip:
                return "ITK"
            return "UNKNOWN"
    except Exception as e:
        print(f"Error processing {nifti_path} or {first_dicom_path}: {e}")
        return "UNKNOWN"


FLIPPER = np.eye(3)
FLIPPER[0, 0] = -1
FLIPPER[1, 1] = -1


def correct_affine(ants_image):
    origin = list(ants_image.origin)
    origin[0] = -origin[0]
    origin[1] = -origin[1]
    ants_image.set_origin(tuple(origin))
    ants_image.set_direction(ants_image.direction @ FLIPPER)
    return ants_image

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

def find_nifti_files(root: str | Path) -> List[Path]:
    """
    Recursively find all .nii and .nii.gz files under a root directory.
    """
    root = Path(root)
    nifti_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                nifti_paths.append(Path(dirpath) / fname)
    return nifti_paths

def compute_dataset_nifti_quantiles(
    root: str | Path,
    q_low: float = 0.005,
    q_high: float = 0.995,
    max_samples: int = 100,
    print_every: int = 10,
) -> Tuple[float, float]:
    """
    Scan a directory of NIfTI files and compute dataset-wide
    low/high HU quantiles by averaging per-file quantiles.

    Parameters
    ----------
    root : str or Path
        Root directory to recursively search for NIfTI files.
    q_low : float
        Lower quantile (e.g. 0.005 = 0.5%).
    q_high : float
        Upper quantile (e.g. 0.995 = 99.5%).
    max_samples : int
        Maximum number of NIfTI files to sample across the dataset.
    print_every : int
        Print progress every N files.

    Returns
    -------
    lo_global : float
        Suggested global lower HU clip (average of per-file q_low).
    hi_global : float
        Suggested global upper HU clip (average of per-file q_high).
    """
    nifti_paths = find_nifti_files(root)
    if not nifti_paths:
        raise FileNotFoundError(f"No NIfTI files found under {root}")

    # Optionally subsample if there are many files
    nifti_paths = nifti_paths[:max_samples]

    q_lows = []
    q_highs = []
    global_min = np.inf
    global_max = -np.inf

    print(f"Found {len(nifti_paths)} NIfTI files to inspect (max_samples={max_samples}).")
    print(f"Using quantiles: low={q_low}, high={q_high}\n")

    for i, path in enumerate(nifti_paths, start=1):
        img = nib.load(str(path))
        # Use scaled HU values (float32 to save memory)
        data = img.get_fdata(dtype=np.float32)

        # Track global min/max just for info
        global_min = min(global_min, float(np.min(data)))
        global_max = max(global_max, float(np.max(data)))

        # Flatten for quantile computation
        flat = data.reshape(-1)
        lo = float(np.quantile(flat, q_low))
        hi = float(np.quantile(flat, q_high))

        q_lows.append(lo)
        q_highs.append(hi)

        if i % print_every == 0 or i == len(nifti_paths):
            print(f"[{i}/{len(nifti_paths)}] {path.name}: "
                  f"q_low={lo:.2f}, q_high={hi:.2f}")

    # Aggregate per-file quantiles
    lo_global = float(np.mean(q_lows))
    hi_global = float(np.mean(q_highs))

    print("\n===== DATASET-WIDE SUMMARY =====")
    print(f"Number of files used   : {len(nifti_paths)}")
    print(f"Global HU min (raw)    : {global_min:.2f}")
    print(f"Global HU max (raw)    : {global_max:.2f}")
    print()
    print(f"Per-file {q_low*100:.2f}%-quantile:")
    print(f"  mean = {np.mean(q_lows):.2f}, "
          f"min = {np.min(q_lows):.2f}, "
          f"max = {np.max(q_lows):.2f}")
    print(f"Per-file {q_high*100:.2f}%-quantile:")
    print(f"  mean = {np.mean(q_highs):.2f}, "
          f"min = {np.min(q_highs):.2f}, "
          f"max = {np.max(q_highs):.2f}")
    print()
    print("===== SUGGESTED GLOBAL CLIP RANGE =====")
    print(f"Clip HU to approximately: [{lo_global:.2f}, {hi_global:.2f}]")
    print()

    return lo_global, hi_global

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
    Given an exam dictionary, compute and return the exam ID as a str
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
    return str(exam_id)

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
    # Normalize orientation in ANTs land
    # ants_img = ants.reorient_image2(ants_img, "LPS")
    return ants_img

def itk_to_ants(itk_img):
    # ITK -> NumPy: shape is (z, y, x)

    arr_zyx = itk.GetArrayViewFromImage(itk_img, keep_axes=False)
    # Convert to (x, y, z) for ants
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

    # ANTs expects a NumPy array; it will internally transpose, so DON'T transpose here
    data = arr_xyz.astype(np.float32)

    # Convert ITK vector-like things to plain Python tuples
    spacing   = tuple(itk_img.GetSpacing())
    origin    = tuple(itk_img.GetOrigin())
    direction = np.array(itk_img.GetDirection()).reshape(3, 3)

    ants_img = ants.from_numpy(
        data,
        origin=origin,
        spacing=spacing,
        direction=direction,
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
        if row["nifti_label"] == "NIBABEL_RAW":
            ants_img = correct_affine(ants_img)
    else:
        paths = json.loads(row["sorted_paths"])
        out_path = os.path.join("/data/rbg/scratch/lung_ct/nlst_nifti", f"sample_{row['exam_id']}.nii.gz")
        _, nifti_image = pydicom_to_nifti(
                                                        paths,
                                                        output_path=out_path,
                                                        save_nifti=False,
                                                        return_nifti=True,
                                                    )
        ants_img = itk_to_ants(nifti_image)
    return ants_img

def build_dummy_fixed(row, geometry = None):
    shape     = tuple(row["fixed_shape"])
    spacing   = tuple(row["fixed_spacing"])
    origin    = tuple(row["fixed_origin"])
    direction = np.array(row["fixed_direction"]).reshape(3, 3)

    if geometry is not None:
        shape     = geometry["shape"]
        spacing   = geometry["spacing"]
        origin    = geometry["origin"]
        direction = geometry["direction"]
        # Convert spacing/origin to Python floats
        shape = tuple(int(s) for s in shape)
        spacing = tuple(float(s) for s in spacing)
        origin  = tuple(float(o) for o in origin)
    
    dummy = ants.from_numpy(
        np.zeros(shape, dtype=np.float32),
        spacing=spacing,
        origin=origin,
        direction=direction,
    )

    return dummy

def apply_transforms(image: ants.ANTsImage, 
                     forward_transform: str | None, 
                     reverse_transform: bool | None = None,
                     dummy_fixed: ants.ANTsImage | None = None, 
                     row: dict | None = None,
                     geometry = None,
                     resampling: bool = True, 
                     resampling_params: tuple = (0.703125 ,0.703125, 2.5), 
                     crop_pad: bool = True, 
                     target_size: tuple = (512, 512, 208), 
                     pad_hu: int = -2000,
                     only_xy: bool = False,
                     interp: str = "linear") -> ants.ANTsImage:
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
    - interp: str
        The interpolation method to use when applying the transform.
    """
    assert dummy_fixed is not None or row is not None, "Either dummy_fixed or row must be provided."

    if forward_transform is not None:
        assert reverse_transform is not None, \
            "reverse_transform must be provided when forward_transform is not None"
        
    # If it is already aligned because it is the first image, skip transform application
    if forward_transform is not None:
        if dummy_fixed is None:
            dummy_fixed = build_dummy_fixed(row, geometry)
        transformed_img = ants.apply_transforms(
            fixed=dummy_fixed,
            moving=image,
            transformlist=[forward_transform],
            whichtoinvert=[reverse_transform],
            interpolator=interp
        )
    else: # If transform is None
        transformed_img = image

    if resampling:
        transformed_img = ants.resample_image(
            transformed_img,
            resample_params=resampling_params,
            use_voxels=False,
            interp_type=1
        )
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

def reverse_normalize(tensor: torch.FloatTensor, clip_window: Tuple[int, int] = [-2000, 500]) -> torch.FloatTensor:
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

def bbox_padded_coords(bbox, image_shape, target_size):
    """
    Expand a bbox within image bounds and compute remaining padding needed
    to reach target size.

    Args:
        bbox: (i_min, i_max, j_min, j_max, k_min, k_max)  # inclusive
        image_shape: (J, I, K)  # mask[j, i, k]
        target_size: (target_J, target_I, target_K)

    Returns:
        new_bbox:
            (i_min_new, i_max_new, j_min_new, j_max_new, k_min_new, k_max_new)
        padding:
            {
              "i": (pad_i_before, pad_i_after),
              "j": (pad_j_before, pad_j_after),
              "k": (pad_k_before, pad_k_after),
            }
    """
    i_min, i_max, j_min, j_max, k_min, k_max = map(int, bbox)
    J, I, K = image_shape
    targ_J, targ_I, targ_K = target_size

    # Current sizes
    size_i = i_max - i_min + 1
    size_j = j_max - j_min + 1
    size_k = k_max - k_min + 1

    # Compute padding needed on each side
    pad_i = max(targ_I - size_i, 0)
    pad_j = max(targ_J - size_j, 0)
    pad_k = max(targ_K - size_k, 0)

     # desired symmetric expansion
    want_i_before = pad_i // 2
    want_i_after  = pad_i - want_i_before

    want_j_before = pad_j // 2
    want_j_after  = pad_j - want_j_before

    want_k_before = pad_k // 2
    want_k_after  = pad_k - want_k_before

    # actual expansion limited by image bounds
    # The max I can grow in the min direction is the coordinate itself
    grow_i_before = min(want_i_before, i_min)
    grow_i_after  = min(want_i_after, I - 1 - i_max)

    grow_j_before = min(want_j_before, j_min)
    grow_j_after  = min(want_j_after, J - 1 - j_max)

    grow_k_before = min(want_k_before, k_min)
    grow_k_after  = min(want_k_after, K - 1 - k_max)

    # new bbox inside image
    i_min_new = i_min - grow_i_before
    i_max_new = i_max + grow_i_after

    j_min_new = j_min - grow_j_before
    j_max_new = j_max + grow_j_after

    k_min_new = k_min - grow_k_before
    k_max_new = k_max + grow_k_after

    # remaining padding needed AFTER cropping
    pad_i = (want_i_before - grow_i_before, want_i_after - grow_i_after)
    pad_j = (want_j_before - grow_j_before, want_j_after - grow_j_after)
    pad_k = (want_k_before - grow_k_before, want_k_after - grow_k_after)

    new_bbox = (i_min_new, i_max_new, j_min_new, j_max_new, k_min_new, k_max_new)
    padding = {"i": pad_i, "j": pad_j, "k": pad_k}

    return new_bbox, padding

def pad_ZYX(vol_zyx: torch.Tensor, padding, pad_value=0):
    """
    vol_zyx: (Z, Y, X) tensor
    padding: dict with keys "i","j","k" mapping to (before, after)
             i -> X, j -> Y, k -> Z
    """
    assert vol_zyx.ndim == 3, f"Expected (Z,Y,X), got {tuple(vol_zyx.shape)}"

    (pad_i0, pad_i1) = padding["i"]  # Y
    (pad_j0, pad_j1) = padding["j"]  # X
    (pad_k0, pad_k1) = padding["k"]  # Z

    # it pads last dimension to first dimension order
    pad_tuple = (pad_j0, pad_j1, pad_i0, pad_i1, pad_k0, pad_k1)

    # F.pad for 3D works cleanly on a 5D tensor (N,C,D,H,W)
    x = vol_zyx.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

    x = F.pad(x, pad_tuple, mode="constant", value=pad_value)

    return x.squeeze(0).squeeze(0)  # back to (Z,Y,X)

def recover_small_bbox(tight_bbox, image_shape, goal_size=(128,128,32)):
    """
    Given the original bounding box coordinates and the image_shape and the goal size of the 
    patches recover the original bounding box in the patches.

    Returned in the same order as the input. (y_min, y_max, x_min, x_max, z_min, z_max)
    """
    updated_bbox, padding = bbox_padded_coords(tight_bbox, image_shape, goal_size)

    ui0, ui1, uj0, uj1, uk0, uk1 = updated_bbox
    ti0, ti1, tj0, tj1, tk0, tk1 = tight_bbox

    # shift volume coords -> cropped (unpadded) patch coords
    ti0, ti1 = ti0 - ui0, ti1 - ui0
    tj0, tj1 = tj0 - uj0, tj1 - uj0
    tk0, tk1 = tk0 - uk0, tk1 - uk0

    # shift cropped coords -> padded_patch coords (add left padding)
    pi0, _ = padding["i"]  # i is Y
    pj0, _ = padding["j"]  # j is X
    pk0, _ = padding["k"]  # k is Z

    return (
        ti0 + pi0, ti1 + pi0,
        tj0 + pj0, tj1 + pj0,
        tk0 + pk0, tk1 + pk0,
    )


def save_slices(
    volume: Union[np.ndarray, torch.Tensor],
    slice_indices: Optional[Iterable[int]] = None,
    output_dir: str = "./test_bbox",
    prefix: str = "transformed",
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    return_images: bool = True,
) -> None:
    """
    Save 2D slices from a 3D volume either with the provided slice or the middle few slices.

    Assumptions:
        - volume shape: (Z, H, W)

    Args
    ----
    volume:
        3D array (Z, H, W) or torch.Tensor with that shape.
    slice_indices:
        Iterable of k-indices to visualize.
        If None, defaults to the middle few slices.
    output_dir:
        Directory where PNGs will be saved.
    prefix:
        Prefix for output filenames, e.g. "orig" or "transformed".
    cmap:
        Matplotlib colormap for the slices.
    vmin, vmax:
        Optional intensity limits for imshow. If None, matplotlib auto-scales.
    return_images: bool = True
        Whether to return the images as a dictionary as well as saving them.

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

    # Default slice range = bbox z-span
    if slice_indices is None:
        slice_indices = range(Z // 2 - 3, Z // 2 + 3)  # middle 6 slices
    else:
        slice_indices = list(slice_indices)

    collected = {} if return_images else None

    os.makedirs(output_dir, exist_ok=True)

    for k in slice_indices:
        if k < 0 or k >= Z:
            continue

        slice_img = volume[k]  # shape (H, W)

        if return_images:
            collected[k] = slice_img

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(slice_img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Slice k={k}")
        ax.axis("off")

        out_path = os.path.join(output_dir, f"{prefix}_z{k:03d}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    if return_images:
        return collected
    
def save_side_by_side_slices(
    slicesA: dict,  # k -> HxW
    slicesB: dict,  # k -> HxW
    output_dir: str,
    prefix: str = "pair",
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gap: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)
    ks = sorted(set(slicesA.keys()) | set(slicesB.keys()))

    for k in ks:
        a = slicesA.get(k)
        b = slicesB.get(k)
        if a is None and b is None:
            continue
        if a is None:
            combo = b
        elif b is None:
            combo = a
        else:
            # simple horizontal concat with a small black gap
            gap_col = np.zeros((a.shape[0], gap), dtype=a.dtype)
            combo = np.concatenate([a, gap_col, b], axis=1)

        out_path = os.path.join(output_dir, f"{prefix}_z{k:03d}.png")
        plt.imsave(out_path, combo, cmap=cmap, vmin=vmin, vmax=vmax)

def save_mp4(frames, fps=10):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    imageio.mimsave(tmp.name, frames, fps=fps)  # imageio automatically picks mp4 writer
    return tmp.name

def safe_delete(path):
    try:
        os.remove(path)
    except OSError:
        pass

def save_montage(
    volume,                 # shape (Z, H, W)
    out_path,
    slice_indices=list(range(0, 32)),
    ncols=6,
    cmap="gray",
    vmin=None,
    vmax=None,
    title=None,
    show_slice_labels=True,
    save_fig=True,
    return_fig=False,
):
    """
    If you return the fig make sure to do plt.close(fig) after returning it
    """
    if len(volume.shape) == 4:
        volume = volume.squeeze(0)
    if torch.is_tensor(volume) and volume.is_cuda:
        volume = volume.detach().to(dtype=torch.float32).cpu()
        
    Z = volume.shape[0]
    slice_indices = [int(k) for k in slice_indices if 0 <= int(k) < Z]
    if len(slice_indices) == 0:
        raise ValueError("No valid slice indices to montage.")

    n = len(slice_indices)
    nrows = math.ceil(n / ncols)

    # Make figure size scale with grid
    fig_w = 2.2 * ncols
    fig_h = 2.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # axes can be 2D or 1D depending on nrows/ncols
    axes = np.atleast_2d(axes)

    for idx, k in enumerate(slice_indices):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        slice_ = volume[k]

        if torch.is_tensor(slice_):
            slice_ = slice_.numpy()
        ax.imshow(slice_, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        if show_slice_labels:
            ax.set_title(f"z={k}", fontsize=10)

    # Turn off any unused panels
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    # If you used suptitle, leave room for it
    if title:
        fig.subplots_adjust(top=0.92)

    if save_fig:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
    if return_fig:
        return fig
    plt.close(fig)

def save_two_figs_side_by_side(fig_left, fig_right, out_path, dpi=200, title=None, subtitle_1 = None, subtitle_2=None):
    """
    Takes two matplotlib Figures and saves them side-by-side as one image.
    """
    # Draw canvases so sizes are known
    fig_left.canvas.draw()
    fig_right.canvas.draw()

    # Get pixel sizes
    w1, h1 = fig_left.canvas.get_width_height()
    w2, h2 = fig_right.canvas.get_width_height()

    # Create combined figure
    fig, axes = plt.subplots(
        1, 2,
        figsize=((w1 + w2) / dpi, max(h1, h2) / dpi),
        dpi=dpi
    )

    axes[0].imshow(fig_left.canvas.buffer_rgba())
    axes[1].imshow(fig_right.canvas.buffer_rgba())

    for ax in axes:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=18, y=0.98)

        if subtitle_1 is not None:
            axes[0].set_title(subtitle_1, fontsize=12)
        if subtitle_2 is not None:
            axes[1].set_title(subtitle_2, fontsize=12)

        # Important so the title doesn't get clipped
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
