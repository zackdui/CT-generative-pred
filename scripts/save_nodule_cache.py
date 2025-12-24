import os

from CTFM.utils import load_config
from CTFM.data import paired_exams_by_pid_nodule_group, encode_and_cache_nodule, consolidate_indices_nodule


if __name__ == "__main__":
    save_paired_nodule_parquet = True
    save_cache = True
    consolidate_cache = True
    split = "test"  # "train" or "val" or "test"

    # Config loading
    path_yaml = "configs/paths.yaml"
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_test_parquet
    bounding_boxes_parquet_path = base_paths.bounding_boxes_test_parquet
    output_dir = base_paths.raw_cached_nodule_dir

    if save_paired_nodule_parquet:
        direc = os.path.dirname(bounding_boxes_parquet_path)
        output_path = os.path.join(direc, "paired_nodules.parquet")
        paired_exams_by_pid_nodule_group(bounding_boxes_parquet_path,
                                        output_path=output_path)
    
    if save_cache:
        encode_and_cache_nodule(
            full_data_parquet,
            bounding_boxes_parquet_path,
            out_root=output_dir,
            split=split,
            shard_size=100,
            force_patch_size=(128, 128, 32),
            num_workers=2,
        )
    if consolidate_cache:
        consolidate_indices_nodule(
            output_dir
        )
        
    
