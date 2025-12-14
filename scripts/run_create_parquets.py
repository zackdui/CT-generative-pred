
import pickle

from CTFM.utils import load_config
from CTFM.data import NLST_Survival_Dataset, full_parquet_creation
from pathlib import Path

if __name__ == "__main__":
    #### Variables ####
    dataset_config_yaml = "configs/nlst_large.yaml"
    path_yaml = "configs/paths.yaml"
    split_group = "dev"  # "train", "dev", "test"; dev becomes "val"
    
    #### Load Configs ####
    args = load_config(dataset_config_yaml)
    base_paths = load_config(path_yaml)
    nodule_parquet_path = base_paths.nodule_tracked_parquet

    metadata_path = Path(base_paths.metadata_dir)  # / "nlst_large_metadata.csv"

    ## Load Dataset 
    nlst_raw = NLST_Survival_Dataset(args, split_group=split_group)
    nlst_data = nlst_raw.dataset
    print(f"Loaded NLST dataset with {len(nlst_data)} exams for split group {split_group}.")

    ## create initial parquet with all data
    exam_to_nifti_file = metadata_path / "zack_exam_to_nifti.pkl"
    with open(exam_to_nifti_file, "rb") as f:
        exam_to_nifti = pickle.load(f)

    if split_group == "dev":
        split_group = "val"

    full_parquet_creation(
        nlst_data,
        exam_to_nifti,
        split_group,
        metadata_path=metadata_path,
        nodule_parquet=nodule_parquet_path,
        debug=True,
    )