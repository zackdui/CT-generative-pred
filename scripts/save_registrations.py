# This file is used to write nifti files, registrations matrices and update parquet files accordingly.
import json
import torch
import os

from CTFM.utils.config import load_config
from CTFM.utils.custom_loggers import RegistrationLogger, merge_log_into_parquet_sequential
from CTFM.data.processing import write_new_nifti_files, write_registration_matrices


def save_bad_exams_to_pt(json_path:str):
    exam_ids = []
    pids = []
    with open(json_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            exam_ids.append(str(record["exam_id"]))
            pids.append(str(record["pid"]))
    data = {
        "exam_ids": exam_ids,
        "pids": pids,
    }
    dir = os.path.dirname(json_path)
    output_path = os.path.join(dir, "bad_registration_exams.pt")
    torch.save(data, output_path)

if __name__ == "__main__":
    # Variables
    save_nifti_files = False
    update_parquet = False
    write_registration = True
    max_nifti_files_to_save = 5  # Set to None to process all files
    logger_name_nifti_writing = "/data/rbg/users/duitz/CT-generative-pred/metadata/nifti_creation_log.jsonl"
    registration_log_path = "/data/rbg/users/duitz/CT-generative-pred/metadata/train/registration_logv3.jsonl"
    updated_output_parquet = "/data/rbg/users/duitz/CT-generative-pred/metadata/train/full_data_single_timepoints_updated.parquet"
    bad_registration_log = "/data/rbg/users/duitz/CT-generative-pred/metadata/train/bad_registration_exams.jsonl"
    save_bad_exams = False

    # Config loading
    path_yaml = "configs/paths.yaml"
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet
    single_patient_parquet = base_paths.full_data_train_timelines_parquet
    nifti_output_dir = base_paths.nifti_dir

    # Logger setup
    nifti_logger = RegistrationLogger(
        logger_name_nifti_writing,
        key_cols=["exam_id"],
    )
    logger_registration = RegistrationLogger(
        registration_log_path,
        key_cols=["exam_id"],
    )

    # Write new NIfTI files
    if save_nifti_files:
        updated_parquet = write_new_nifti_files(full_data_parquet, nifti_logger, nifti_output_dir, max_files=max_nifti_files_to_save)

    # Write registration matrices
    if write_registration:
        print("Starting writing registration matrices...")
        updated_parquet_registration = write_registration_matrices(
            single_patient_parquet=single_patient_parquet,
            all_data_parquet=full_data_parquet,
            logger=logger_registration,
            resample=True,
            max_transform_nifti_files=0,
            bad_exam_json=bad_registration_log,)
        print("Finished writing registration matrices.")
        print(f"Updated parquet with registrations saved to: {updated_parquet_registration}")

    # Update the main parquet with the log
    if update_parquet:
        print("Starting merging registration log into parquet...")
        merge_log_into_parquet_sequential(
            base_parquet=full_data_parquet,
            log_path=registration_log_path,
            key_cols=["exam_id"],
            output_parquet=updated_output_parquet,
        )
        print("Finished merging registration log into parquet.")
    
    if save_bad_exams:
        save_bad_exams_to_pt(bad_registration_log)

    






