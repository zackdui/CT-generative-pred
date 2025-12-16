import shutil
import os, sys

import torch
from nibabel.spatialimages import HeaderDataError
from types import SimpleNamespace
import pickle
import pydicom
import numpy as np
import nibabel as nib
import torchio as tio
import pandas as pd
import ants
from pathlib import Path
from tqdm import tqdm

import json

from .utils import (pydicom_to_nifti, 
                    nib_to_ants, 
                    apply_transforms, 
                    get_ants_image_from_row,
                    correct_affine,
                    itk_to_ants,
                    )
from .bounding_boxes import get_geometry_from_nifti, get_geometry_from_dicoms

def _log_skip(jsonl_path, exam_id, pid, error=None):
    """"
    Logs a skipped exam to a JSONL file with the exam ID, patient ID, and optional error message.
    Helpful when processing large datasets and you need to track issues
    """
    entry = {
        "exam_id": exam_id,
        "pid": pid,
        "error": str(error) if error else None,
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def _load_jsonl_to_lists(jsonl_path):
    """
    Loads a JSONL file and extracts exam IDs and patient IDs into separate lists.
    """
    exam_ids = []
    pids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            exam_ids.append(str(record["exam_id"]))
            pids.append(str(record["pid"]))
    return exam_ids, pids

def write_new_nifti_files(full_data_parquet, 
                          logger, 
                          nifti_output_dir: str = "/data/rbg/scratch/lung_ct/nlst_nifti/", 
                          max_files: int = None, 
                          replace_existing: bool = False) -> str:
    """
    This function reads a parquet file containing metadata about medical exams,
    checks for exams that do not have associated NIfTI files, converts the corresponding
    DICOM files to NIfTI format, and saves the new NIfTI files to a specified output directory.
    It updates the parquet file to reflect the addition of the new NIfTI files.
    """
    df = pd.read_parquet(full_data_parquet)
    files_added = 0
    for index, row in df.iterrows():
        if row["has_nifti"]:
            continue  # already has nifti
        output_path = os.path.join(nifti_output_dir, f"sample_{row['exam_id']}.nii.gz")
        pydicom_to_nifti(json.loads(row["sorted_paths"]), output_path, save_nifti=True)
        row["has_nifti"] = True
        row["sorted_paths"] = None
        row["nifti_path"] = output_path
        row["nifti_label"] = "ITK"

        logger.log_record(
            exam_id=row["exam_id"],
            has_nifti=True,
            nifti_path=output_path,
            sorted_paths=None
        )
        files_added += 1
        if max_files is not None and files_added >= max_files:
            break
    if replace_existing:
        df.to_parquet(full_data_parquet, index=False)
    return full_data_parquet

def write_registration_matrices(single_patient_parquet,
                                all_data_parquet,
                                logger,
                                resample: bool = True,
                                resampling_params: tuple = (0.703125 ,0.703125, 2.5),
                                matrix_output_dir: str = "/data/rbg/scratch/lung_ct/registration_zack",
                                save_transforms: bool = False, 
                                transform_nifti_output_dir: str = "/data/rbg/scratch/lung_ct/nlst_nifti_transformed", 
                                save_nifti: bool = False, 
                                nifti_output_dir: str = "/data/rbg/scratch/lung_ct/nlst_nifti",
                                max_nifti_files: int = 10,
                                max_transform_nifti_files: int = 100,
                                bad_exam_json: str = "./bad_registration_exams.jsonl",
                                registration_file: str | None = "/data/rbg/users/duitz/CT-generative-pred/metadata/mapping_registered_exams.pkl"):
    """
    This function reads two parquet files: one containing single patient timelines and another
    containing all exam data. It performs image registration for patients with multiple exams,
    registering each subsequent exam to the first exam. The function saves the resulting
    transformation matrices and optionally saves the transformed NIfTI images. It updates
    the all data parquet file to reflect the registration status and paths to the registration files.
    It will also optionally create NIfTI files for exams that do not have them yet.
    It returns the path to the updated all data parquet file.
    It also writes to a specific RegistrationLogger instance to log the registrations performed and nifti files
    created. In case it breaks before finishing the main parquet file can be updated with the log file.
    The transformed nifti files saved will not be written to the main parquet file but the file format is 
    f"transformed_{exam_id}.nii.gz" and will be saved in the transform_nifti_output_dir.

    Saved updated all_data_parquet file to the same directory as the input all_data_parquet with the name
    "all_data_with_registrations.parquet".

    If registration_exists is true and then registration_file is pd.NA that means it is the first exam for that patient.

    Parameters:
    - single_patient_parquet: str
        Path to the parquet file containing single patient timelines.
    - all_data_parquet: str
        Path to the parquet file containing all exam data.
    - logger: RegistrationLogger
        An instance of RegistrationLogger to log registration events.
    - resample: bool
        Whether to resample images after registration before saving the transforms.
    - resampling_params: tuple
        Parameters for resampling (spacing in mm).
    - matrix_output_dir: str
        Directory to save the registration matrices.
    - save_transforms: bool
        Whether to save the transformed NIfTI images.
    - transform_nifti_output_dir: str
        Directory to save the transformed NIfTI images.
    - save_nifti: bool
        Whether to create NIfTI files for exams that do not have them.
    - nifti_output_dir: str
        Directory to save the created NIfTI files.
    - max_nifti_files: int or None
        Maximum number of NIfTI files to create. If None, no limit.
    - max_transform_nifti_files: int or None
        Maximum number of transformed NIfTI files to save. If None, no limit.
    - bad_exam_json: str
        Path to a JSONL file to log bad exams that cannot be processed.
        It must contain exam_id and pid fields.
    - registration_file: str
        Path to a pickle file containing the registration mapping. The Keys are tuple(sorted(fixed_exam, moving_exam))

    Returns:
    - str
        Path to the updated all data parquet file.
    """
    os.makedirs(matrix_output_dir, exist_ok=True)
    os.makedirs(transform_nifti_output_dir, exist_ok=True)
    os.makedirs(nifti_output_dir, exist_ok=True)
    bad_exam_ids, bad_pids  = _load_jsonl_to_lists(bad_exam_json)
    bad_pids_set = set(bad_pids)

    if registration_file is not None:
        with open(registration_file, "rb") as f:
            registration_mapping = pickle.load(f)
    else:
        registration_mapping = {}

    # Variables
    regular_nifti_count = 0
    transform_nifti_count = 0
    total_registrations_saved = 0
    in_pair_key = 0
    
    # Load both dataframes
    df_single_patient = pd.read_parquet(single_patient_parquet)

    df_all_data = pd.read_parquet(all_data_parquet)
    if "registration_exists" not in df_all_data.columns:
        df_all_data["registration_exists"] = False
    if "registration_file" not in df_all_data.columns:
        df_all_data["registration_file"] = pd.NA
    if "fixed_shape" not in df_all_data.columns:
        df_all_data["fixed_shape"] = pd.NA
    if "fixed_spacing" not in df_all_data.columns:
        df_all_data["fixed_spacing"] = pd.NA
    if "fixed_origin" not in df_all_data.columns:
        df_all_data["fixed_origin"] = pd.NA
    if "fixed_direction" not in df_all_data.columns:
        df_all_data["fixed_direction"] = pd.NA
    if "transformed_nifti_path" not in df_all_data.columns:
        df_all_data["transformed_nifti_path"] = pd.NA
    if "reverse_transform" not in df_all_data.columns:
        df_all_data["reverse_transform"] = pd.NA
    if "fixed_exam_id" not in df_all_data.columns:
        df_all_data["fixed_exam_id"] = pd.NA
    if "original_shape" not in df_all_data.columns:
        df_all_data["original_shape"] = pd.NA
    if "original_spacing" not in df_all_data.columns:
        df_all_data["original_spacing"] = pd.NA
    if "original_origin" not in df_all_data.columns:
        df_all_data["original_origin"] = pd.NA
    if "original_direction" not in df_all_data.columns:
        df_all_data["original_direction"] = pd.NA

    # make sure boolean columns are actually boolean
    df_all_data["registration_exists"] = df_all_data["registration_exists"].fillna(False).astype("bool")
    # Set has_nifti to boolean
    df_all_data["has_nifti"] = df_all_data["has_nifti"].fillna(False).astype("boolean")
    index_map = {eid: i for i, eid in enumerate(df_all_data["exam_id"])}


    for index, row in tqdm(
                            df_single_patient.iterrows(),
                            total=len(df_single_patient),
                            desc="Registering patients",
                            mininterval=5.0,
                        ):
        if row["pid"] in bad_pids_set:
            continue  # skip bad exams
        fixed_index = index_map[row["exam_id"][0]]
        fixed_index = int(fixed_index)
        fixed_exam_id = row["exam_id"][0]
        full_data_exam_row = df_all_data.iloc[fixed_index]
        if row["n_exams"] < 2:
            df_all_data.loc[fixed_index, "registration_exists"] = True
            logger.log_record(
                exam_id=fixed_exam_id,
                registration_exists=True
            )
            continue  # no registration needed

        # if df_all_data.iloc[fixed_index]["registration_exists"] != 1.0:
        #     import pdb; pdb.set_trace()
        if df_all_data.iloc[fixed_index]["registration_exists"]:
            continue  # already has registration; The first exam is only updated to true after the other exams are registered to it
        
        # Get the fixed image in ants format and save the nifti file if needed
        fixed_img = None
        if not full_data_exam_row["has_nifti"]:
            # Create nifti file to register
            paths = json.loads(full_data_exam_row["sorted_paths"])
            out_path = os.path.join(nifti_output_dir, f"sample_{full_data_exam_row['exam_id']}.nii.gz")
            # This nifti_image is going to be a nibabel object while everything else is ants
            # Save if save_nifti is true and we have not exceeded the max number of nifti files to create
            do_save_nifti = save_nifti and (max_nifti_files is None or (max_nifti_files is not None and regular_nifti_count < max_nifti_files))
            try:
                _, nifti_image = pydicom_to_nifti(
                                                        paths,
                                                        output_path=out_path,
                                                        save_nifti=do_save_nifti,
                                                        return_nifti=True,
                                                    )
            except HeaderDataError:
                # *** SKIP THIS EXAM, PRINT ONLY pid + exam_id, CONTINUE ***
                print(f"SKIP_BAD_NIFTI pid={full_data_exam_row['pid']} exam_id={full_data_exam_row['exam_id']}")
                _log_skip(bad_exam_json, full_data_exam_row['exam_id'], full_data_exam_row['pid'], error="HeaderDataError during NIfTI conversion")
                continue
            fixed_img = itk_to_ants(nifti_image)
            fixed_img = fixed_img.astype("float32")

            if do_save_nifti:
                logger.log_record(
                    exam_id=full_data_exam_row["exam_id"],
                    has_nifti=True,
                    nifti_path=out_path,
                    sorted_paths=None,
                    nifti_label="ITK",
                )
                regular_nifti_count += 1

            fixed_direction_flat = np.asarray(fixed_img.direction).ravel().tolist()
            logger.log_record(
                exam_id=full_data_exam_row["exam_id"],
                fixed_shape=[int(x) for x in fixed_img.shape],
                fixed_spacing=[float(x) for x in fixed_img.spacing],
                fixed_origin=[float(x) for x in fixed_img.origin],
                fixed_direction=fixed_direction_flat,
                fixed_exam_id=fixed_exam_id,
                original_shape=[int(x) for x in fixed_img.shape],
                original_spacing=[float(x) for x in fixed_img.spacing],
                original_origin=[float(x) for x in fixed_img.origin],
                original_direction=fixed_direction_flat,
            )
            df_all_data.loc[fixed_index, "has_nifti"] = True
            df_all_data.loc[fixed_index, "nifti_path"] = out_path
            df_all_data.loc[fixed_index, "sorted_paths"] = None
            df_all_data.loc[fixed_index, "nifti_label"] = "ITK"
            df_all_data.at[fixed_index, "fixed_shape"]    = [int(x) for x in fixed_img.shape]        # [nx, ny, nz]
            df_all_data.at[fixed_index, "fixed_spacing"]  = [float(x) for x in fixed_img.spacing]       # [sx, sy, sz]
            df_all_data.at[fixed_index, "fixed_origin"]   = [float(x) for x in fixed_img.origin]        # [ox, oy, oz]
            df_all_data.at[fixed_index, "fixed_direction"] = fixed_direction_flat    # length-9 flattened list
            df_all_data.at[fixed_index, "fixed_exam_id"] = fixed_exam_id
            df_all_data.at[fixed_index, "original_shape"] = [int(x) for x in fixed_img.shape]
            df_all_data.at[fixed_index, "original_spacing"] = [float(x) for x in fixed_img.spacing]
            df_all_data.at[fixed_index, "original_origin"] = [float(x) for x in fixed_img.origin]
            df_all_data.at[fixed_index, "original_direction"] = fixed_direction_flat
        else:
            fixed_img = ants.image_read(full_data_exam_row["nifti_path"])
            if full_data_exam_row["nifti_label"] == "NIBABEL_RAW":
                fixed_img = correct_affine(fixed_img)
            elif full_data_exam_row["nifti_label"] == "UNKNOWN":
                print(f"SKIP_BAD_NIFTI pid={full_data_exam_row['pid']} exam_id={full_data_exam_row['exam_id']}, Unknown Nifti label")
                _log_skip(bad_exam_json, full_data_exam_row["exam_id"], full_data_exam_row['pid'], error="Unknown Nifti label, cannot correct affine")
                continue
            fixed_img = fixed_img.astype("float32")

            if pd.isna(df_all_data.iloc[fixed_index]["fixed_shape"]):
                df_all_data.at[fixed_index, "fixed_shape"]    = [int(x) for x in fixed_img.shape]        # [nx, ny, nz]
                df_all_data.at[fixed_index, "fixed_spacing"]  = [float(x) for x in fixed_img.spacing]       # [sx, sy, sz]
                df_all_data.at[fixed_index, "fixed_origin"]   = [float(x) for x in fixed_img.origin]        # [ox, oy, oz]
                fixed_direction_flat = np.asarray(fixed_img.direction).ravel().tolist()
                df_all_data.at[fixed_index, "fixed_direction"] = fixed_direction_flat
                df_all_data.at[fixed_index, "fixed_exam_id"] = fixed_exam_id
                df_all_data.at[fixed_index, "original_shape"] = [int(x) for x in fixed_img.shape]
                df_all_data.at[fixed_index, "original_spacing"] = [float(x) for x in fixed_img.spacing]
                df_all_data.at[fixed_index, "original_origin"] = [float(x) for x in fixed_img.origin]
                df_all_data.at[fixed_index, "original_direction"] = fixed_direction_flat
                logger.log_record(
                        exam_id=full_data_exam_row["exam_id"],
                        fixed_shape=[int(x) for x in fixed_img.shape],
                        fixed_spacing=[float(x) for x in fixed_img.spacing],
                        fixed_origin=[float(x) for x in fixed_img.origin],
                        fixed_direction=fixed_direction_flat,
                        fixed_exam_id=fixed_exam_id,
                        original_shape=[int(x) for x in fixed_img.shape],
                        original_spacing=[float(x) for x in fixed_img.spacing],
                        original_origin=[float(x) for x in fixed_img.origin],
                        original_direction=fixed_direction_flat,
                    )
        
        # Register each moving image to the fixed image
        for exam_id in row["exam_id"][1:]:
            moving_img = None
            moving_index = index_map[exam_id]
            moving_index = int(moving_index)
            full_data_exam_row_moving = df_all_data.iloc[moving_index]
            if not full_data_exam_row_moving["has_nifti"]:
                # Create nifti file to register
                paths = json.loads(full_data_exam_row_moving["sorted_paths"])
                out_path = os.path.join(nifti_output_dir, f"sample_{full_data_exam_row_moving['exam_id']}.nii.gz")
                # This nifti_image is going to be a nibabel object while everything else is ants
                do_save_nifti = save_nifti and (max_nifti_files is None or (max_nifti_files is not None and regular_nifti_count < max_nifti_files))
                try:
                    image, nifti_image = pydicom_to_nifti(
                                                            paths,
                                                            output_path=out_path,
                                                            save_nifti=do_save_nifti,
                                                            return_nifti=True,
                                                        )
                except HeaderDataError:
                    print(f"SKIP_BAD_NIFTI pid={full_data_exam_row['pid']} exam_id={exam_id}")
                    _log_skip(bad_exam_json, exam_id, full_data_exam_row['pid'], error="HeaderDataError during NIfTI conversion")
                    continue
                moving_img = itk_to_ants(nifti_image)
                moving_img = moving_img.astype("float32")

                if do_save_nifti:
                    logger.log_record(
                        exam_id=full_data_exam_row_moving["exam_id"],
                        has_nifti=True,
                        nifti_path=out_path,
                        sorted_paths=None,
                        nifti_label="ITK",
                    )
                    regular_nifti_count += 1
                    df_all_data.at[moving_index, "has_nifti"] = True
                    df_all_data.at[moving_index, "nifti_path"] = out_path
                    df_all_data.at[moving_index, "sorted_paths"] = None
                    df_all_data.at[moving_index, "nifti_label"] = "ITK"

            else:
                moving_img = ants.image_read(full_data_exam_row_moving["nifti_path"])
                if full_data_exam_row_moving["nifti_label"] == "NIBABEL_RAW":
                    moving_img = correct_affine(moving_img)
                elif full_data_exam_row_moving["nifti_label"] == "UNKNOWN":
                    print(f"SKIP_BAD_NIFTI pid={full_data_exam_row_moving['pid']} exam_id={full_data_exam_row_moving['exam_id']}, Unknown Nifti label")
                    _log_skip(bad_exam_json, full_data_exam_row_moving["exam_id"], full_data_exam_row_moving['pid'], error="Unknown Nifti label, cannot correct affine")
                    continue
                moving_img = moving_img.astype("float32")
                
            pair_key = tuple(sorted((fixed_exam_id, exam_id)))
            if pair_key in registration_mapping:
                in_pair_key += 1
                registration_fixed = str(registration_mapping[pair_key]["fixed"])
                forward_path = str(registration_mapping[pair_key]["path"])
                reverse_transform = registration_fixed != fixed_exam_id
                if in_pair_key % 200 == 0:
                    print(f"Up to precomputed pair_keys count={in_pair_key}")
            else:
                rigid = ants.registration(fixed_img, moving_img, type_of_transform="Rigid")
                forward_path = os.path.join(matrix_output_dir, f"forward_f{fixed_exam_id}_m{exam_id}.mat")
                reverse_transform = False
                fwd_tmp = rigid["fwdtransforms"][0]
                
                if not os.path.exists(fwd_tmp):
                    print(
                        f"SKIP_BAD_TRANSFORM pid={full_data_exam_row_moving['pid']} "
                        f"exam_id={exam_id} reason=missing_forward_tmp"
                    )
                    _log_skip(bad_exam_json, exam_id, full_data_exam_row_moving['pid'], error="Missing temporary forward transform file")
                    continue

                try:
                    shutil.move(fwd_tmp, forward_path)
                except FileNotFoundError:
                    # Extremely defensive: if something still went wrong, skip this exam.
                    print(
                        f"SKIP_BAD_TRANSFORM pid={full_data_exam_row_moving['pid']} "
                        f"exam_id={exam_id} reason=move_failed"
                    )
                    _log_skip(bad_exam_json, exam_id, full_data_exam_row_moving['pid'], error="Failed to move temporary forward transform file")
                    continue
            
            df_all_data.at[moving_index, "registration_file"] = forward_path
            df_all_data.at[moving_index, "registration_exists"] = True
            df_all_data.at[moving_index, "reverse_transform"] = reverse_transform
            df_all_data.at[moving_index, "fixed_shape"]    = [int(x) for x in fixed_img.shape]        # [nx, ny, nz]
            df_all_data.at[moving_index, "fixed_spacing"]  = [float(x) for x in fixed_img.spacing]       # [sx, sy, sz]
            df_all_data.at[moving_index, "fixed_origin"]   = [float(x) for x in fixed_img.origin]      # [ox, oy, oz]
            fixed_direction_flat = np.asarray(fixed_img.direction).ravel().tolist()
            df_all_data.at[moving_index, "fixed_direction"] = fixed_direction_flat    # length-9 flattened list
            df_all_data.at[moving_index, "fixed_exam_id"] = fixed_exam_id
            df_all_data.at[moving_index, "original_shape"] = [int(x) for x in moving_img.shape]
            df_all_data.at[moving_index, "original_spacing"] = [float(x) for x in moving_img.spacing]
            df_all_data.at[moving_index, "original_origin"] = [float(x) for x in moving_img.origin]
            moving_direction_flat = np.asarray(moving_img.direction).ravel().tolist()
            df_all_data.at[moving_index, "original_direction"] = moving_direction_flat


            logger.log_record(
                exam_id=exam_id,
                registration_file=forward_path,
                registration_exists=True,
                fixed_shape=[int(x) for x in fixed_img.shape],
                fixed_spacing=[float(x) for x in fixed_img.spacing],
                fixed_origin=[float(x) for x in fixed_img.origin],
                fixed_direction=fixed_direction_flat,
                fixed_exam_id=fixed_exam_id,
                original_shape=[int(x) for x in moving_img.shape],
                original_spacing=[float(x) for x in moving_img.spacing],
                original_origin=[float(x) for x in moving_img.origin],
                original_direction=moving_direction_flat,
            )

            total_registrations_saved += 1
            if total_registrations_saved % 100 == 0:
                print(f"Saved {total_registrations_saved} registrations so far...")

            if save_transforms and (max_transform_nifti_files is None or (max_transform_nifti_files is not None and transform_nifti_count < max_transform_nifti_files)):
                transformed_nifti_path = os.path.join(transform_nifti_output_dir, f"transformed_{exam_id}.nii.gz")
                transformed_img = apply_transforms(moving_img,
                                                   forward_transform=forward_path,
                                                   reverse_transform=reverse_transform,
                                                   row=df_all_data.iloc[moving_index],
                                                   resampling=resample,
                                                   resampling_params=resampling_params,)
                ants.image_write(transformed_img, transformed_nifti_path)
                df_all_data.at[moving_index, "transformed_nifti_path"] = transformed_nifti_path
                logger.log_record(
                    exam_id=exam_id,
                    transformed_nifti_path=transformed_nifti_path
                )
                transform_nifti_count += 1
                

        df_all_data.loc[fixed_index, "registration_exists"] = True
        logger.log_record(
            exam_id=fixed_exam_id,
            registration_exists=True
        )
        if save_transforms and (max_transform_nifti_files is None or (max_transform_nifti_files is not None and transform_nifti_count < max_transform_nifti_files)):
            if resample:
                fixed_img = ants.resample_image(
                    fixed_img,
                    resample_params=resampling_params,
                    use_voxels=False,
                    interp_type=1
                )
            transformed_nifti_path = os.path.join(transform_nifti_output_dir, f"transformed_{fixed_exam_id}.nii.gz")
            ants.image_write(fixed_img, transformed_nifti_path)
            transform_nifti_count += 1

    df_all_data["registration_exists"] = df_all_data["registration_exists"].astype("bool")
    dirpath = os.path.dirname(all_data_parquet)
    new_path = os.path.join(dirpath, "all_data_with_registrations.parquet")
    df_all_data.to_parquet(new_path, index=False)
    return new_path

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

def save_transformed_nifti_files(exam_id_list: list[str],
                                full_data_parquet: str,
                                output_dir: str = "/data/rbg/scratch/lung_ct/nlst_nifti_transformed",
                                saved_transforms: dict[str, str] = None,
                                resampling: bool = True,
                                resampling_params: tuple = (0.703125 ,0.703125, 2.5),
                                crop_pad: bool = False,
                                pad_value: int = -2000,
                                target_size: tuple = (512, 512, 208)):
    """
    This function saves transformed NIfTI files for a list of exam IDs based on the registration
    information stored in a parquet file. It applies the saved transformations to the original
    NIfTI images and saves the resulting transformed images to a specified output directory.
    It returns a list of exam IDs for which the transformed NIfTI files were successfully saved
    """
    total_images_wrote = 0
    exam_ids_wrote = []
    df = pd.read_parquet(full_data_parquet)
    exam_to_index = {eid: i for i, eid in enumerate(df["exam_id"])}
    if saved_transforms is None:
        saved_transforms = {}
    for exam_id in exam_id_list:
        index = exam_to_index[exam_id]
        row = df.iloc[index]
        if exam_id in saved_transforms:
            continue
        # Process the row and save the transformed NIfTI file
        ants_image = get_ants_image_from_row(row)
        forward_transform = row["registration_file"]
        reverse_transform = row["reverse_transform"]
        if pd.isna(forward_transform):
            forward_transform = None
            reverse_transform = None
        ants_transformed_image = apply_transforms(ants_image,
                                                  forward_transform=forward_transform,
                                                  reverse_transform=reverse_transform,
                                                  row=row,
                                                  resampling=resampling, 
                                                  resampling_params=resampling_params, 
                                                  crop_pad=crop_pad, 
                                                  target_size=target_size,
                                                  pad_hu=pad_value,
                                                  only_xy=False)
    
        transformed_nifti_path = os.path.join(output_dir, f"transformed_{exam_id}.nii.gz")

        ants.image_write(ants_transformed_image, transformed_nifti_path)
        total_images_wrote += 1
        exam_ids_wrote.append(exam_id)

    print(f"Total transformed NIfTI images wrote: {total_images_wrote}")
    return exam_ids_wrote

def remove_bad_exams(
    full_data_parquet: str,
    bad_exam_json: str = "./bad_registration_exams.jsonl"
) -> str:
    """
    This function removes bad pids from the full data parquet file based on a JSONL file
    containing the exam IDs and patient IDs of the bad exams. It returns the path to the
    updated parquet file with the bad exams removed.
    """
    bad_exam_ids, bad_pids  = _load_jsonl_to_lists(bad_exam_json)
    bad_pids_set = set(bad_pids)

    df = pd.read_parquet(full_data_parquet)
    initial_count = len(df)
    df_cleaned = df[~df["pid"].isin(bad_pids_set)].reset_index(drop=True)
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    print(f"Removed {removed_count} bad exams from the parquet file.")
    
    dirpath = os.path.dirname(full_data_parquet)
    new_path = os.path.join(dirpath, "full_data_cleaned.parquet")
    df_cleaned.to_parquet(new_path, index=False)
    return new_path

def save_correct_bounding_boxes(full_data_parquet: str,
                                nodule_parquet: str,
                                resampling: bool = True,
                                resampling_params: tuple = (0.703125 ,0.703125, 2.5),
                                crop_pad: bool = False,
                                target_size: tuple = (512, 512, 208),
                                output_file: str | None = None) -> str:
    """
    This function corrects the bounding boxes of nodules based on the registration
    information stored in a full data parquet file. It reads the nodule data from another
    parquet file, applies the transformations to the original bounding box coordinates,
    and saves the corrected bounding boxes to a new parquet file. It returns the path to
    the output parquet file containing the nodules with corrected bounding boxes.
    
    Parameters:
    - full_data_parquet: str
        Path to the parquet file containing full exam data with registration information.
    - nodule_parquet: str
        Path to the parquet file containing nodule data with original bounding box coordinates.
    - resampling: bool
        Whether to resample images after registration before applying transformations.
    - resampling_params: tuple
        Parameters for resampling (spacing in mm).
    - crop_pad: bool
        Whether to crop or pad the images to a target size.
    - target_size: tuple
        Target size for cropping or padding (nx, ny, nz).
    - output_file: str or None
        Path to the output parquet file. If None, a default name will be used in the
        same directory as the nodule_parquet. (nodules_with_fixed_bboxes.parquet)
    Returns:
    - str
        Path to the output parquet file containing nodules with corrected bounding boxes.
    """
    full_data_df = pd.read_parquet(full_data_parquet).reset_index(drop=True)
    exam_id_to_row = {row["exam_id"]: row for _, row in full_data_df.iterrows()}
    
    nodule_df = pd.read_parquet(nodule_parquet).reset_index(drop=True)
    nodule_exam_id_to_indices = (
                                nodule_df
                                .groupby("exam").indices
                            )
    total_nodules_added = 0
    new_nodule_rows = []
    for _, exam_full_row in tqdm(full_data_df.iterrows(), total=len(full_data_df), desc="Correcting bounding boxes", mininterval=5.0):
        exam_id = exam_full_row["exam_id"]
        nodule_indices = nodule_exam_id_to_indices.get(exam_id, [])
       
        # Create mask in original space
        # Was the original shape saved
        if exam_full_row["original_shape"] is None or pd.isna(exam_full_row["original_shape"]) is True:
            if exam_full_row["has_nifti"]:
                geometry = get_geometry_from_nifti(exam_full_row["nifti_path"])
            else:
                geometry = get_geometry_from_dicoms(json.loads(exam_full_row["sorted_paths"]))
            shape = tuple(geometry["shape"])
            spacing = tuple(geometry["spacing"])
            origin = tuple(geometry["origin"])
            direction = geometry["direction"]
        else:
            shape = tuple(exam_full_row["original_shape"])
            spacing = tuple(exam_full_row["original_spacing"])
            origin = tuple(exam_full_row["original_origin"])
            direction = np.array(exam_full_row["original_direction"]).reshape(3, 3)

        for nodule_ind in nodule_indices:
            nodule_row = nodule_df.iloc[nodule_ind]
            
            coords_orig = np.array(nodule_row["coords"])
            i_min, i_max, j_min, j_max, k_min, k_max = map(int, coords_orig)
            mask = np.zeros(shape, dtype=np.uint8)

            mask[int(j_min):int(j_max)+1, int(i_min):int(i_max)+1, int(k_min):int(k_max)+1] = 1

            mask_ants = ants.from_numpy(mask, spacing=spacing,
                                origin=origin,
                                direction=direction)
            forward_transform = exam_full_row["registration_file"]
            reverse_transform = exam_full_row["reverse_transform"]
            if pd.isna(forward_transform) or forward_transform is None:
                forward_transform = None
                reverse_transform = None
            mask_transform = apply_transforms(mask_ants,
                                                forward_transform=forward_transform,
                                                reverse_transform=reverse_transform,
                                                row=exam_full_row,
                                                resampling=resampling,
                                                resampling_params=resampling_params,
                                                crop_pad=crop_pad,
                                                pad_hu=0,
                                                target_size=target_size,
                                                interp="nearestNeighbor")
            warped_np = mask_transform.numpy()
            coords_new = np.argwhere(warped_np > 0.2)
            j_min_t, i_min_t, k_min_t = coords_new.min(axis=0)
            j_max_t, i_max_t, k_max_t = coords_new.max(axis=0)
            bbox_fixed = [i_min_t, i_max_t, j_min_t, j_max_t, k_min_t, k_max_t]
            new_row = nodule_row.copy()
            new_row["coords_fixed"] = bbox_fixed
            new_nodule_rows.append(new_row)
            total_nodules_added += 1
            if total_nodules_added % 100 == 0:
                print(f"Corrected bounding boxes for {total_nodules_added} nodules so far...")
    new_nodule_df = pd.DataFrame(new_nodule_rows)
    if output_file is None:
        dirpath = os.path.dirname(full_data_parquet)
        output_file = os.path.join(dirpath, "nodules_with_fixed_bboxes.parquet")
    new_nodule_df.to_parquet(output_file, index=False)
    print(f"Saved corrected bounding boxes for {total_nodules_added} nodules to {output_file}")
    return output_file
