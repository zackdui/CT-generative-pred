# This file will take the nodule pickle file and convert it to parquet for easier handling
from CTFM.data import pickle_to_parquet
from CTFM.utils import load_config
    


if __name__ == "__main__":
    path_yaml = "configs/paths.yaml"
    base_paths = load_config(path_yaml)
    p_path = base_paths.nodule_tracked_file
    out_parquet = base_paths.nodule_tracked_parquet

    pickle_to_parquet(p_path, out_parquet)