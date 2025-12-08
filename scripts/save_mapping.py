# This file saves mappings from exam IDs to NIfTI file paths

from CTFM.data import save_mapping_from_exams_to_nifti, save_mapping_transformed_exams_to_nifti
from CTFM.utils import load_config

if __name__ == "__main__":
    save_regular_mapping = False
    save_transformed_mapping = True
    paths_config = load_config("configs/paths.yaml")

    if save_regular_mapping:
        save_mapping_from_exams_to_nifti(
            nifti_directory=paths_config.nifti_dir,
            output_path="/data/rbg/users/duitz/CT-generative-pred/metadata/zack_exam_to_nifti.pkl"
        )

    if save_transformed_mapping:
        save_mapping_transformed_exams_to_nifti(
            nifti_directory=paths_config.nifti_transformed_dir,
            output_path="/data/rbg/users/duitz/CT-generative-pred/metadata/exam_to_nifti_transformed.pkl"
        )
        