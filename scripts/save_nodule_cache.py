import os
from vae3d2d import CustomVAE

from CTFM.utils import load_config
from CTFM.data import paired_exams_by_pid_nodule_group, encode_and_cache_nodule, consolidate_indices_nodule


if __name__ == "__main__":
    # Remember to change the full_data_parquet and bounding_boxes_parquet_path depending on split
    save_paired_nodule_parquet = False
    save_cache = True
    consolidate_cache = True
    split = "test"  # "train" or "val" or "test"
    do_encode = True
    encode_model_checkpoint = "/data/rbg/users/duitz/CT-generative-pred/final_saved_models/vae_fixed_std_no_reg.pt"

    ## Encoder if needed
    if do_encode:
        encode_model = CustomVAE.load_from_checkpoint(encode_model_checkpoint, map_location="cpu")
    else:
        encode_model = None

    # Config loading
    path_yaml = "configs/paths.yaml"
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_test_parquet
    bounding_boxes_parquet_path = base_paths.bounding_boxes_test_parquet
    if not do_encode:
        output_dir = base_paths.raw_cached_nodule_dir
    else:
        output_dir = base_paths.encoded_cached_nodule_dir
        # output_dir = "/data/rbg/scratch/test_nlst_nodule_encoded_cache/"

    if save_paired_nodule_parquet:
        direc = os.path.dirname(bounding_boxes_parquet_path)
        output_path = os.path.join(direc, "paired_nodules.parquet")
        paired_exams_by_pid_nodule_group(bounding_boxes_parquet_path,
                                        output_path=output_path)
    
    rank = int(os.environ.get("LOCAL_RANK", "0"))

    if save_cache:
        encode_and_cache_nodule(
            full_data_parquet,
            bounding_boxes_parquet_path,
            out_root=output_dir,
            encoder=encode_model,
            do_encode=do_encode,
            split=split,
            shard_size=100,
            force_patch_size=(128, 128, 32),
            num_workers=2,
            device=f"cuda:{rank}",
            rank=rank,
        )

    if consolidate_cache and rank == 0:
        consolidate_indices_nodule(
            output_dir
        )


