# Create the mapping of exam_ids to Nifti file paths 
import os
import pickle
import re

def save_mapping_from_exams_to_nifti(nifti_directory="/data/rbg/scratch/lung_ct/nlst_nifti/", output_path="zack_exam_to_nifti.pkl"):
    """
    Create and save a mapping from exam IDs to NIfTI file paths.
    The mapping is saved as a pickle file at the specified output path.
    """
    nifti_files = {}
    for nifti_file in os.listdir(nifti_directory):
        if any(k in nifti_file for k in ["T0", "T1", "T2"]):
            _, pid, tp, series = nifti_file.split("_")
            series = series.split(".nii.gz")[0]
            exam = f"{pid}{tp}{series.split('.')[-1][:5]}{series.split('.')[-1][:5]}"
        else:
            exam = nifti_file.split("_")[1].split(".")[0]
        exam = exam[:17] if len(exam) > 17 else exam
        nifti_files[exam] = os.path.join(
            nifti_directory, nifti_file
        )

    with open(output_path, "wb") as f:
        pickle.dump(nifti_files, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved exam to NIfTI mapping with {len(nifti_files)} entries to {output_path}")

def save_mapping_transformed_exams_to_nifti(nifti_directory="/data/rbg/scratch/lung_ct/nlst_nifti_transformed", output_path="/data/rbg/users/duitz/CT-generative-pred/metadata/exam_to_nifti_transformed.pkl"):
    """
    Create and save a mapping from exam IDs to NIfTI file paths for transformed files.
    The mapping is saved as a pickle file at the specified output path.

    Nifti file format is transformed_{exam_id}.nii.gz or transformed_{exam_id}.nii
    """
    nifti_files = {}
    for nifti_file in os.listdir(nifti_directory):
        base = nifti_file.removesuffix(".nii.gz").removesuffix(".nii")
        prefix, exam_id = base.rsplit("_", 1)
        nifti_files[exam_id] = os.path.join(
            nifti_directory, nifti_file
        )

    with open(output_path, "wb") as f:
        pickle.dump(nifti_files, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved transformed exam to NIfTI mapping with {len(nifti_files)} entries to {output_path}")

def save_exam_registration_mapping(mat_dir: str, output_pkl: str) -> None:
    """
    Scan `mat_dir` for files named:
      rigid_f<fixed_exam>_m<moving_exam>.mat

    Build a mapping from unordered exam pairs to the actual file order and path.
    Key: (min(fixed_exam, moving_exam), max(fixed_exam, moving_exam))
    Value: dict with keys:
        - 'fixed':   exam id used as fixed in the file
        - 'moving':  exam id used as moving in the file
        - 'filename': basename of the .mat file
        - 'path':    full path to the .mat file

    Save the mapping as a .pkl file at `output_pkl`.
    """
    pattern = re.compile(r"rigid_f(\d+)_m(\d+)\.mat$")
    mapping = {}

    for fname in os.listdir(mat_dir):
        m = pattern.match(fname)
        if not m:
            continue

        fixed_exam, moving_exam = m.groups()
        pair_key = tuple(sorted((fixed_exam, moving_exam)))

        # If both directions exist, keep the first one we saw.
        if pair_key not in mapping:
            full_path = os.path.join(mat_dir, fname)
            mapping[pair_key] = {
                "fixed": fixed_exam,
                "moving": moving_exam,
                "filename": fname,
                "path": full_path,
            }

    with open(output_pkl, "wb") as f:
        pickle.dump(mapping, f)