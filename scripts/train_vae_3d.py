# encode_latents_main.py
import os
import sys
import pickle
import torch

# For local use of vae3d2d module
sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE, AttnParams, training_3D, setup_logger

from CTFM.utils import load_config, OPTIMIZERS
from CTFM.data import CTOrigDataset3D, RepeatedImageDataset

def setup_model_and_train(encoder_3d_configs, training_3d_configs, base_dataset):
    act = tuple(encoder_3d_configs.act)
    attn_params = AttnParams(**encoder_3d_configs.attn_params)

    encoder_model = CustomVAE(
        num_groups=encoder_3d_configs.num_groups,
        in_channels=encoder_3d_configs.in_channels,
        dropout_prob=encoder_3d_configs.dropout_prob,
        spatial_dims=3,
        vae_latent_channels=encoder_3d_configs.vae_latent_channels,
        debug_mode=encoder_3d_configs.debug_mode,
        act=act,
        upsample_mode=encoder_3d_configs.upsample_mode,
        init_filters=encoder_3d_configs.init_filters,
        beta=encoder_3d_configs.beta,
        vae_use_log_var=encoder_3d_configs.vae_use_log_var,
        blocks_down=tuple(encoder_3d_configs.blocks_down),
        blocks_up=tuple(encoder_3d_configs.blocks_up),
        use_attn=encoder_3d_configs.use_attn,
        attn_params=attn_params,
    )

    model_name = encoder_3d_configs.model_name

    optimizer_cls = OPTIMIZERS[training_3d_configs.optimizer_cls.lower()]

    optimizer_kwargs = dict(
        lr=training_3d_configs.optimizer_lr,
        weight_decay=training_3d_configs.optimizer_weight_decay,
    )

    logger_train = setup_logger(name=f"{model_name[:-3]}_logs")
    # build checkpoint dir from model_name 
    checkpoint_dir = f"{model_name[:-3]}{training_3d_configs.checkpoint_dir_suffix}"

    history = training_3D(
        encoder_model,
        base_dataset,
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
        wandb_run_name=training_3d_configs.train_wandb_name,
        checkpoint_dir=checkpoint_dir,
        save_every_steps=training_3d_configs.save_every_steps,
        best_check_every_steps=training_3d_configs.best_check_every_steps,
    )

    return history

if __name__ == "__main__":
    
    #### Variables ####
    path_yaml = "configs/paths.yaml"
    vae_3d_model_yaml = "configs/vae_3d_model.yaml"
    training_3d_yaml = "configs/train_vae_3d.yaml"
    image_size = (512, 512, 208)
    max_examples = None
    repeat_one_image = False

    
    #### Load Configs ####
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet
    saved_transforms_file = base_paths.saved_transforms_file
    with open(saved_transforms_file, 'rb') as f:
        saved_transforms = pickle.load(f)

    encoder_3d_configs = load_config(vae_3d_model_yaml)
    training_3d_configs = load_config(training_3d_yaml)

    base_dataset = CTOrigDataset3D(full_data_parquet, 
                    saved_transforms=saved_transforms, 
                    max_lenth=max_examples, 
                    return_meta_data=False)
    
    # Option to repeat one image 
    if repeat_one_image:
        first_image = base_dataset[0]
        z0, z1 = 90, 90+32
        y0, y1 = 0, 512
        x0, x1 = 0, 512
        first_image = first_image[:, z0:z1, y0:y1, x0:x1] 
        print(f"First image shape: {first_image.shape}, min: {first_image.min()}, max: {first_image.max()}")
        base_dataset = RepeatedImageDataset(first_image, repeat_count=4000)

    #### Setup model and train ####
    history = setup_model_and_train(encoder_3d_configs, training_3d_configs, base_dataset)

    


    