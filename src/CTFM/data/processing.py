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

import json

from .utils import pydicom_to_nifti, ants_crop_or_pad_like_torchio, nib_to_ants
from CTFM.utils.config import load_config
from CTFM.utils.registration_logger import RegistrationLogger, merge_log_into_parquet_sequential

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
                                save_transforms: bool = True, 
                                transform_nifti_output_dir: str = "/data/rbg/scratch/lung_ct/nlst_nifti_transformed", 
                                save_nifti: bool = True, 
                                nifti_output_dir: str = "/data/rbg/scratch/lung_ct/nlst_nifti",
                                max_nifti_files: int = 10,
                                max_transform_nifti_files: int = 100):
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
        Whether to resample images before registration.
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

    Returns:
    - str
        Path to the updated all data parquet file.
    """
    os.makedirs(matrix_output_dir, exist_ok=True)
    os.makedirs(transform_nifti_output_dir, exist_ok=True)
    os.makedirs(nifti_output_dir, exist_ok=True)

    # Variables
    regular_nifti_count = 0
    transform_nifti_count = 0
    total_registrations_saved = 0
    
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

    # make sure boolean columns are actually boolean
    df_all_data["registration_exists"] = df_all_data["registration_exists"].fillna(False)
    index_map = {eid: i for i, eid in enumerate(df_all_data["exam_id"])}

    for index, row in df_single_patient.iterrows():
        fixed_index = index_map[row["exam_id"][0]]
        fixed_index = int(fixed_index)
        fixed_exam_id = row["exam_id"][0]
        full_data_exam_row = df_all_data.iloc[fixed_index]
        if row["n_exams"] < 2:
            df_all_data.loc[fixed_index, "registration_exists"] = True
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
                                                        save_nifti=save_nifti,
                                                        return_nifti=True,
                                                    )
            except HeaderDataError:
                # *** SKIP THIS EXAM, PRINT ONLY pid + exam_id, CONTINUE ***
                print(f"SKIP_BAD_NIFTI pid={full_data_exam_row['pid']} exam_id={full_data_exam_row['exam_id']}")
                continue
            fixed_img = nib_to_ants(nifti_image)
            fixed_img = fixed_img.astype("float32")

            if do_save_nifti:
                logger.log_record(
                    exam_id=full_data_exam_row["exam_id"],
                    has_nifti=True,
                    nifti_path=out_path,
                    sorted_paths=None
                )
                regular_nifti_count += 1
            if resample:
                fixed_img = ants.resample_image(
                    fixed_img,
                    resample_params=resampling_params,
                    use_voxels=False,
                    interp_type=1
                )
            fixed_direction_flat = np.asarray(fixed_img.direction).ravel().tolist()
            logger.log_record(
                exam_id=full_data_exam_row["exam_id"],
                fixed_shape=[int(x) for x in fixed_img.shape],
                fixed_spacing=[float(x) for x in fixed_img.spacing],
                fixed_origin=[float(x) for x in fixed_img.origin],
                fixed_direction=fixed_direction_flat
            )
            df_all_data.loc[fixed_index, "has_nifti"] = True
            df_all_data.loc[fixed_index, "nifti_path"] = out_path
            df_all_data.loc[fixed_index, "sorted_paths"] = None
            df_all_data.at[fixed_index, "fixed_shape"]    = [int(x) for x in fixed_img.shape]        # [nx, ny, nz]
            df_all_data.at[fixed_index, "fixed_spacing"]  = [float(x) for x in fixed_img.spacing]       # [sx, sy, sz]
            df_all_data.at[fixed_index, "fixed_origin"]   = [float(x) for x in fixed_img.origin]        # [ox, oy, oz]
            df_all_data.at[fixed_index, "fixed_direction"] = fixed_direction_flat    # length-9 flattened list
        else:
            fixed_img = ants.image_read(full_data_exam_row["nifti_path"])
            fixed_img = fixed_img.astype("float32")
            if resample:
                fixed_img = ants.resample_image(
                    fixed_img,
                    resample_params=resampling_params,
                    use_voxels=False,
                    interp_type=1
                )
            if pd.isna(df_all_data.iloc[fixed_index]["fixed_shape"]):
                df_all_data.at[fixed_index, "fixed_shape"]    = [int(x) for x in fixed_img.shape]        # [nx, ny, nz]
                df_all_data.at[fixed_index, "fixed_spacing"]  = [float(x) for x in fixed_img.spacing]       # [sx, sy, sz]
                df_all_data.at[fixed_index, "fixed_origin"]   = [float(x) for x in fixed_img.origin]        # [ox, oy, oz]
                fixed_direction_flat = np.asarray(fixed_img.direction).ravel().tolist()
                df_all_data.at[fixed_index, "fixed_direction"] = fixed_direction_flat
                logger.log_record(
                        exam_id=full_data_exam_row["exam_id"],
                        fixed_shape=[int(x) for x in fixed_img.shape],
                        fixed_spacing=[float(x) for x in fixed_img.spacing],
                        fixed_origin=[float(x) for x in fixed_img.origin],
                        fixed_direction=fixed_direction_flat
                    )
        
        # Register each moving image to the fixed image
        for exam_id in row["exam_id"][1:]:
            moving_img = None
            moving_index = index_map[exam_id]
            moving_index = int(moving_index)
            full_data_exam_row = df_all_data.iloc[moving_index]
            if not full_data_exam_row["has_nifti"]:
                # Create nifti file to register
                paths = json.loads(full_data_exam_row["sorted_paths"])
                out_path = os.path.join(nifti_output_dir, f"sample_{full_data_exam_row['exam_id']}.nii.gz")
                # This nifti_image is going to be a nibabel object while everything else is ants
                do_save_nifti = save_nifti and (max_nifti_files is None or (max_nifti_files is not None and regular_nifti_count < max_nifti_files))
                try:
                    image, nifti_image = pydicom_to_nifti(
                                                            paths,
                                                            output_path=out_path,
                                                            save_nifti=save_nifti,
                                                            return_nifti=True,
                                                        )
                except HeaderDataError:
                    print(f"SKIP_BAD_NIFTI pid={full_data_exam_row['pid']} exam_id={exam_id}")
                    continue
                moving_img = nib_to_ants(nifti_image)
                moving_img = moving_img.astype("float32")

                if do_save_nifti:
                    logger.log_record(
                        exam_id=full_data_exam_row["exam_id"],
                        has_nifti=True,
                        nifti_path=out_path,
                        sorted_paths=None
                    )
                    regular_nifti_count += 1
                df_all_data.at[moving_index, "has_nifti"] = True
                df_all_data.at[moving_index, "nifti_path"] = out_path
                df_all_data.at[moving_index, "sorted_paths"] = None

            else:
                moving_img = ants.image_read(full_data_exam_row["nifti_path"])
                moving_img = moving_img.astype("float32")
                
            if resample:
                moving_img = ants.resample_image(
                    moving_img,
                    resample_params=resampling_params,
                    use_voxels=False,
                    interp_type=1
                )


            rigid = ants.registration(fixed_img, moving_img, type_of_transform="Rigid")
            forward_path = os.path.join(matrix_output_dir, f"forward_f{fixed_exam_id}_m{exam_id}.mat")
            shutil.move(rigid["fwdtransforms"][0], forward_path)
            
            df_all_data.at[moving_index, "registration_file"] = forward_path
            df_all_data.at[moving_index, "registration_exists"] = True
            df_all_data.at[moving_index, "fixed_shape"]    = [int(x) for x in fixed_img.shape]        # [nx, ny, nz]
            df_all_data.at[moving_index, "fixed_spacing"]  = [float(x) for x in fixed_img.spacing]       # [sx, sy, sz]
            df_all_data.at[moving_index, "fixed_origin"]   = [float(x) for x in fixed_img.origin]      # [ox, oy, oz]
            fixed_direction_flat = np.asarray(fixed_img.direction).ravel().tolist()
            df_all_data.at[moving_index, "fixed_direction"] = fixed_direction_flat    # length-9 flattened list


            logger.log_record(
                exam_id=exam_id,
                registration_file=forward_path,
                registration_exists=True,
                fixed_shape=[int(x) for x in fixed_img.shape],
                fixed_spacing=[float(x) for x in fixed_img.spacing],
                fixed_origin=[float(x) for x in fixed_img.origin],
                fixed_direction=fixed_direction_flat
            )

            total_registrations_saved += 1
            if total_registrations_saved % 100 == 0:
                print(f"Saved {total_registrations_saved} registrations so far...")

            if save_transforms and (max_transform_nifti_files is None or (max_transform_nifti_files is not None and transform_nifti_count < max_transform_nifti_files)):
                transformed_nifti_path = os.path.join(transform_nifti_output_dir, f"transformed_{exam_id}.nii.gz")
                transformed_img = ants.apply_transforms(
                    fixed=fixed_img,
                    moving=moving_img,
                    transformlist=[forward_path],
                    interpolator="linear"
                )
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


if __name__ == "__main__":
    # Variables
    nifti_output_dir = "/data/rbg/users/duitz/CT-generative-pred/test_data"
    max_files = 5  # Set to None to process all files

    # Config loading
    path_yaml = "configs/paths.yaml"
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet
    single_patient_parquet = base_paths.full_data_train_timelines_parquet

    # Logger setup
    logger = RegistrationLogger(
        "/data/rbg/users/duitz/CT-generative-pred/test_data/nifti_test_creation_log.jsonl",
        key_cols=["exam_id"],
    )
    logger_registration = RegistrationLogger(
        "/data/rbg/users/duitz/CT-generative-pred/metadata/train/registration_logv2.jsonl",
        key_cols=["exam_id"],
    )

    # # Write new NIfTI files
    # updated_parquet = write_new_nifti_files(full_data_parquet, logger, nifti_output_dir, max_files=max_files)

    # # Update the main parquet with the log
    # merge_log_into_parquet_sequential(
    #     base_parquet=full_data_parquet,
    #     log_path= "/data/rbg/users/duitz/CT-generative-pred/test_data/nifti_test_creation_log.jsonl",
    #     key_cols=["exam_id"],
    #     output_parquet="/data/rbg/users/duitz/CT-generative-pred/test_data/updated.parquet",
    # )

    print("Starting writing registration matrices...")
    update_parquet_registration = write_registration_matrices(
        single_patient_parquet=single_patient_parquet,
        all_data_parquet=full_data_parquet,
        logger=logger_registration,
        resample=True)
    print("Finished writing registration matrices.")
    print(f"Updated parquet with registrations saved to: {update_parquet_registration}")

    # # Update the main parquet with the log
    # print("Starting merging registration log into parquet...")
    # merge_log_into_parquet_sequential(
    #     base_parquet=full_data_parquet,
    #     log_path=  "/data/rbg/users/duitz/CT-generative-pred/metadata/train/registration_log.jsonl",
    #     key_cols=["exam_id"],
    #     output_parquet="/data/rbg/users/duitz/CT-generative-pred/metadata/train/full_data_single_timepoints_updated.parquet",
    # )
    # print("Finished merging registration log into parquet.")

    








    # Testing code
    # first = pd.read_parquet("/data/rbg/users/duitz/CT-generative-pred/test_data/updated.parquet")
    # second = pd.read_parquet(full_data_parquet)
    # print(f"Length of first: {len(first)}, length of second: {len(second)}")
    # # Locate exam with exam_id 10000402215824639 and print the row for both dataframes
    # exam_id_to_check = "10000402215824639"
    # print("Row in first dataframe:")
    # print(first[first["exam_id"] == exam_id_to_check]["sorted_paths"])
    # print("Row in second dataframe:")
    # print(second[second["exam_id"] == exam_id_to_check]["sorted_paths"])
    # print("Row in first dataframe:")
    # print(first[first["exam_id"] == exam_id_to_check]["nifti_path"])
    # print("Row in second dataframe:")
    # print(second[second["exam_id"] == exam_id_to_check]["nifti_path"])

    # print("Base dtypes:")
    # print(first["exam_id"].dtypes)

    # print("\nLog dtypes:")
    # print(second["exam_id"].dtypes)

    # print("same object?:", first is second)
    # print("same content?:", first.equals(second))