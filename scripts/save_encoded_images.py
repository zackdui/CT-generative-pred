# save_encoded_images.py
import os

from vae3d2d import CustomVAE

from CTFM.data.cache_encoded import encode_and_cache, consolidate_indices
from CTFM.models.auto_encoder_2d import AutoEncoder_Lightning
from CTFM.utils.config import load_config

def run_encode(dataset_parquet: str, 
               out_root: str, 
               is_3d: bool, 
               checkpoint_path_3d: str = "", 
               split: str = "train"):
    
    if is_3d:
        encoder_model = CustomVAE.load_from_checkpoint(path=checkpoint_path_3d)
        encoder = encoder_model
    else:
        encoder_model = AutoEncoder_Lightning()
        encoder = encoder_model

    encode_and_cache(
        dataset_parquet=dataset_parquet,
        encoder=encoder,
        out_root=out_root,
        split=split,
        is_3d=is_3d,
        batch_size=1,
        num_workers=2,
        shard_size=200,
        device="cuda",
        overwrite=False,
        rank=int(os.environ.get("RANK", "0")),
    )

def run_consolidate(out_root: str):
    consolidate_indices(
        out_root=out_root,
    )

if __name__ == "__main__":
    # run with torchrun --nproc_per_node=k scripts/save_encoded_images.py
    #### Variables ####
    path_yaml = "configs/paths.yaml"
    split_group = "train"
    is_3d = False
    run_saving_encoded_flag = True
    run_consolidate_flag = True

    
    #### Load Configs ####
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet
    checkpoint_path_3d = base_paths.vae_3d_checkpoint_path
    if not is_3d:
        encoded_path = base_paths.encoded_dir_2d
    else:
        encoded_path = base_paths.encoded_dir_3d

    #### Run Encoding and Caching ####
    if run_saving_encoded_flag:
        run_encode(
            dataset_parquet=full_data_parquet,
            out_root=encoded_path,
            is_3d=is_3d,
            checkpoint_path_3d=checkpoint_path_3d,
            split=split_group,
        )

    rank = int(os.environ.get("RANK", "0"))
    if run_consolidate_flag and rank == 0:
        run_consolidate(encoded_path)

