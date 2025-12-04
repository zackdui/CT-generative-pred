# encode_latents_main.py
import os
from CTFM.data.cache_encoded import encode_and_cache, consolidate_indices
from CTFM.models.auto_encoder_2d import AutoEncoder_Lightning
from vae3d2d import CustomVAE, AttnParams, training_3D
from CTFM.utils.config import load_config, OPTIMIZERS

def setup_model_and_training():
    pass

if __name__ == "__main__":
    
    #### Variables ####
    path_yaml = "configs/paths.yaml"
    vae_3d_model_yaml = "configs/vae_3d_model.yaml"
    training_3d_yaml = "configs/train_vae_3d.yaml"
    split_group = "train"
    is_3d = False
    run_saving_encoded_flag = True
    run_consolidate_flag = True

    
    #### Load Configs ####
    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet

    encoder_3d_configs = load_config(vae_3d_model_yaml)
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

    training_3d_configs = load_config(training_3d_yaml)

    optimizer_cls = OPTIMIZERS[training_3d_configs.optimizer_cls.lower()]

    optimizer_kwargs = dict(
        lr=training_3d_configs.optimizer_lr,
        weight_decay=training_3d_configs.optimizer_weight_decay,
    )

    # build checkpoint dir from model_name if you want to keep your old pattern
    checkpoint_dir = f"{model_name[:-3]}{training_3d_configs.checkpoint_dir_suffix}"

    training_3D(
        model,
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
        wandb_run_name=train_wandb_name,
        checkpoint_dir=checkpoint_dir,
        save_every_steps=training_3d_configs.save_every_steps,
        best_check_every_steps=training_3d_configs.best_check_every_steps,
    )

