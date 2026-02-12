import sys
import os

# For local use of vae3d2d module
sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE, AttnParams, setup_logger, eval_model_3D

from CTFM.utils import load_config, OPTIMIZERS
from CTFM.data import CachedNoduleDataset, CTNoduleDataset3D, RepeatedImageDataset



if __name__=="__main__":
    model_checkpoint = "experiments/vae_3d/single_small_beta_no_log.pt"
    path_yaml = "configs/paths.yaml"
    eval_3d_yaml = "configs/eval_3d_vae.yaml"
    max_examples = None  # set to None to use all examples

    # Load parameters
    base_paths = load_config(path_yaml)
    full_test_parquet = base_paths.full_data_test_parquet
    full_nodule_parquet = base_paths.bounding_boxes_test_parquet
    raw_nodule_index = base_paths.raw_nodule_index
    data_root = base_paths.raw_cached_nodule_dir

    eval_3d_configs = load_config(eval_3d_yaml)

    model = CustomVAE.load_from_checkpoint(model_checkpoint, map_location="cuda")

    model_name = model.get_name()

    logger_eval = setup_logger(save_dir=f"experiments/vae_3d/{model_name}/eval", name=f"eval_{model_name[:-3]}_logs")
    
    save_dir=f"experiments/vae_3d/{model_name}/eval/examples"
    os.makedirs(save_dir, exist_ok=True)

    test_dataset = CachedNoduleDataset(full_test_parquet, 
                                       full_nodule_parquet, 
                                       raw_nodule_index, 
                                       data_root, 
                                       split="test",
                                       max_length=max_examples,
                                       max_cache_size=50)
    results = eval_model_3D(model,
                            test_dataset,
                            batch_size=eval_3d_configs.batch_size,
                            mode=eval_3d_configs.mode,
                            patch_size=eval_3d_configs.patch_size,
                            stride=eval_3d_configs.stride,
                            save_dir=save_dir,
                            use_blending=eval_3d_configs.use_blending,
                            logger=logger_eval,
                            use_wandb=eval_3d_configs.use_wandb,
                            wandb_run_name=eval_3d_configs.wandb_run_name,
                            wandb_project_name=eval_3d_configs.wandb_project_name,
                            num_examples_to_save=eval_3d_configs.num_examples_to_save,
                            is_hu=eval_3d_configs.is_hu)
