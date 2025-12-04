

if __name__ == "__main__":
    #### Variables ####
    dataset_config_yaml = "configs/nlst_large.yaml"
    path_yaml = "configs/paths.yaml"
    split_group = "train"
    
    #### Load Configs ####
    args = load_config(dataset_config_yaml)
    base_paths = load_config(path_yaml)

    metadata_path = Path(base_paths.metadata_dir)  # / "nlst_large_metadata.csv"

    ## Load Dataset 
    nlst_raw = NLST_Survival_Dataset(args, split_group=split_group)
    nlst_data = nlst_raw.dataset

    ## create initial parquet with all data
    exam_to_nifti_file = metadata_path / "zack_exam_to_nifti.pkl"
    with open(exam_to_nifti_file, "rb") as f:
        exam_to_nifti = pickle.load(f)

    full_parquet_creation(
        nlst_data,
        exam_to_nifti,
        split_group,
        debug=True,
    )