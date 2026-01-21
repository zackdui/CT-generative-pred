# import sys
import os
import shutil
import torch
import torch.distributed as dist

# For local use of vae3d2d module
# sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE, AttnParams, training_3D, setup_logger

from CTFM.utils import load_config, OPTIMIZERS
from CTFM.data import CachedNoduleDataset, CTNoduleDataset3D, RepeatedImageDataset

def is_global_rank_zero() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def setup_model_and_train(encoder_3d_configs, training_3d_configs, base_dataset, val_dataset=None):
    act = tuple(encoder_3d_configs.act)
    attn_params = AttnParams(**encoder_3d_configs.attn_params)

    model_name = encoder_3d_configs.model_name

    encoder_model = CustomVAE(
        model_name=model_name[:-3],
        fixed_std=encoder_3d_configs.fixed_std,
        num_groups=encoder_3d_configs.num_groups,
        in_channels=encoder_3d_configs.in_channels,
        dropout_prob=encoder_3d_configs.dropout_prob,
        spatial_dims=3,
        vae_latent_channels=encoder_3d_configs.vae_latent_channels,
        smallest_filters=encoder_3d_configs.smallest_filters,
        debug_mode=encoder_3d_configs.debug_mode,
        act=act,
        upsample_mode=encoder_3d_configs.upsample_mode,
        init_filters=encoder_3d_configs.init_filters,
        res_block_weight=encoder_3d_configs.res_block_weight,
        beta=encoder_3d_configs.beta,
        vae_use_log_var=encoder_3d_configs.vae_use_log_var,
        blocks_down=tuple(encoder_3d_configs.blocks_down),
        blocks_up=tuple(encoder_3d_configs.blocks_up),
        use_attn=encoder_3d_configs.use_attn,
        attn_params=attn_params,
        use_checkpoint=encoder_3d_configs.use_checkpoint,
        custom_losses=['mse'],
        downsample_strides=encoder_3d_configs.downsample_strides,
        vae_down_stride=encoder_3d_configs.vae_down_stride,
    )

    optimizer_cls = OPTIMIZERS[training_3d_configs.optimizer_cls.lower()]

    optimizer_kwargs = dict(
        lr=training_3d_configs.optimizer_lr,
        weight_decay=training_3d_configs.optimizer_weight_decay,
    )

    logger_train = setup_logger(save_dir=f"experiments/vae_3d/{model_name[:-3]}", name=f"{model_name[:-3]}_logs")
    # build checkpoint dir from model_name 
    checkpoint_dir = f"experiments/vae_3d/{model_name[:-3]}/{model_name[:-3]}{training_3d_configs.checkpoint_dir_suffix}"
    save_dir = f"experiments/vae_3d/{model_name[:-3]}/final_model"

    history = training_3D(
        encoder_model,
        base_dataset,
        val_dataset=val_dataset,
        optimizer_cls=optimizer_cls,
        accum_steps=training_3d_configs.accum_steps,
        patching=training_3d_configs.patching,
        patch_size=tuple(training_3d_configs.patch_size),
        patches_per_volume=training_3d_configs.patches_per_volume,
        train_batch=training_3d_configs.train_batch,
        val_batch=training_3d_configs.val_batch,
        epochs=training_3d_configs.epochs,
        train_split=training_3d_configs.train_split,
        num_workers=training_3d_configs.num_workers,
        optimizer_kwargs=optimizer_kwargs,
        model_file_name=model_name,
        logger=logger_train,
        use_amp=training_3d_configs.use_amp,
        amp_dtype=getattr(torch, training_3d_configs.amp_dtype) if training_3d_configs.use_amp else torch.bfloat16,
        use_wandb=training_3d_configs.use_wandb,
        wandb_project=training_3d_configs.wandb_project,
        wandb_run_name=training_3d_configs.train_wandb_name,
        checkpoint_dir=checkpoint_dir,
        final_save_dir=save_dir,
        save_every_steps=training_3d_configs.save_every_steps,
        best_check_every_steps=training_3d_configs.best_check_every_steps,
    )

    return history

if __name__ == "__main__":
    # run with torchrun --nproc_per_node=k scripts/train_vae_3d_nodules.py
    #### Variables ####
    path_yaml = "configs/paths.yaml"
    vae_3d_model_yaml = "configs/vae_3d_model.yaml"
    training_3d_yaml = "configs/train_vae_3d_nodule.yaml"
    image_size = (128, 128, 32)
    max_examples = None
    repeat_one_image = False
    repeat_count=20000

    
    #### Load Configs ####
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet
    full_nodule_parquet = base_paths.bounding_boxes_train_parquet
    raw_nodule_index = base_paths.raw_nodule_index
    data_root = base_paths.raw_cached_nodule_dir

    full_data_val_parquet = base_paths.full_data_val_parquet
    full_nodule_val_parquet = base_paths.bounding_boxes_val_parquet

    encoder_3d_configs = load_config(vae_3d_model_yaml)
    training_3d_configs = load_config(training_3d_yaml)

    # Copy over config file
    if is_global_rank_zero():
        model_name = encoder_3d_configs.model_name[:-3]
        run_output_dir = f"experiments/vae_3d/{model_name}"
        os.makedirs(run_output_dir, exist_ok=True)
        unet_config_copy_dst = os.path.join(run_output_dir, f"vae_config.yaml")
        train_config_copy_dst = os.path.join(run_output_dir, f"train_config.yaml")
        shutil.copy2(vae_3d_model_yaml, unet_config_copy_dst)
        shutil.copy2(training_3d_yaml, train_config_copy_dst)


    # base_dataset = CTNoduleDataset3D(
    #     parquet_path=full_data_parquet,
    #     nodules_parquet_path=full_nodule_parquet,
    #     mode="single",
    #     force_patch_size=image_size,
    #     return_meta_data=False,
    # )
    base_dataset = CachedNoduleDataset(full_data_parquet, 
                                       full_nodule_parquet, 
                                       raw_nodule_index, 
                                       data_root, 
                                       split="train",
                                       max_length=max_examples,
                                       max_cache_size=50)
    
    val_dataset = CachedNoduleDataset(full_data_val_parquet, 
                                      full_nodule_val_parquet, 
                                      raw_nodule_index, 
                                      data_root, 
                                      split="val",
                                      max_length=max_examples,
                                      max_cache_size=50)
    
    # Option to repeat one image 
    if repeat_one_image:
        first_image = base_dataset[0]
        print(f"First image shape: {first_image.shape}, min: {first_image.min()}, max: {first_image.max()}")
        base_dataset = RepeatedImageDataset(first_image, repeat_count=repeat_count)
        val_dataset=None # This will use the base_dataset in the model

    #### Setup model and train ####
    history = setup_model_and_train(encoder_3d_configs, training_3d_configs, base_dataset, val_dataset=val_dataset)

    


    