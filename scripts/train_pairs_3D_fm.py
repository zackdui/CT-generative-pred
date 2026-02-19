# This file is used to run training of 3D diffusion model on unconditional
import os
import shutil
from monai.networks.nets import DiffusionModelUNet
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import *
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torchinfo import summary
# import sys
# sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE

# This repo imports
from CTFM.data import (CachedNoduleDataset, 
                       CTNoduleDataset3D, 
                       RepeatedImageDataset, 
                       save_slices, 
                       collate_image_meta, 
                       reverse_normalize,
                       save_montage)
from CTFM.utils import (plot_lr_from_metrics, 
                        plot_loss_from_metrics, 
                        load_config, 
                        window_ct_hu_to_png)
from CTFM.models import UnetLightning3D


def main(model_checkpoint, # Can be None
         training_args, 
         unet_3d_cfm, 
         train_dataset, 
         val_dataset, 
         test_dataset, 
         run_output_dir, 
         decode_model=None,
         single_image=False,
         repeat_count=20000,
         train_bboxes=None):

    debug_flag = training_args.debug_flag
    train_batch_size = training_args.train_batch_size
    test_batch_size = training_args.test_batch_size
    lr = training_args.lr
    
    if debug_flag:
        # import pdb; pdb.set_trace()
        train_batch_size = 1
        test_batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers = 2, drop_last=True, collate_fn=collate_image_meta) #, timeout=30)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, collate_fn=collate_image_meta)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_image_meta)

    ## Model Creation ##
    unet_3d_cfm_configs = load_config(unet_3d_cfm)

    ####### For Single Image Training ########
    if single_image:
        # import pdb; pdb.set_trace()
        image_both = train_dataset[0][0]
        image_in = train_dataset[0][0][:unet_3d_cfm_configs.in_channels]
        image_out = train_dataset[0][0][unet_3d_cfm_configs.in_channels:]
        meta_data = train_dataset[0][1]
        train_dataset = RepeatedImageDataset(image_both, repeat_count, meta_data=meta_data)
        val_dataset = RepeatedImageDataset(image_both, 4, meta_data=meta_data)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_image_meta)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_image_meta)
        if decode_model is not None:
            decode_model.eval()
            with torch.no_grad():
                decode_model = decode_model.to('cuda')
                image_out = decode_model.decode(image_out.unsqueeze(0).to(next(decode_model.parameters()).device))
                image_in = decode_model.decode(image_in.unsqueeze(0).to(next(decode_model.parameters()).device))
                image_out = image_out.squeeze(0)
                image_in = image_in.squeeze(0)
        reverse_normalized_image = reverse_normalize(image_out.cpu())
        image_to_save = window_ct_hu_to_png(reverse_normalized_image, center=-600.0, width=1500.0, bit_depth=8)
        reverse_normalized_image_in = reverse_normalize(image_in.cpu())
        image_to_save_in = window_ct_hu_to_png(reverse_normalized_image_in, center=-600.0, width=1500.0, bit_depth=8)
        single_image_output_dir = os.path.join(run_output_dir, "single_image_example")
        os.makedirs(single_image_output_dir, exist_ok=True)
        save_slices(image_to_save, output_dir=single_image_output_dir, prefix="single_image_goal_")
        save_slices(image_to_save_in, output_dir=single_image_output_dir, prefix="single_image_input_")
        save_montage(image_to_save, out_path=os.path.join(single_image_output_dir, "single_image_goal_montage.png"))
        save_montage(image_to_save_in, out_path=os.path.join(single_image_output_dir, "single_image_input_montage.png"))
    ##################################

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    unet_kwargs = dict(
            spatial_dims=unet_3d_cfm_configs.spatial_dims,
            in_channels=unet_3d_cfm_configs.in_channels,
            out_channels=unet_3d_cfm_configs.out_channels,
            num_res_blocks=unet_3d_cfm_configs.num_res_blocks,
            channels=unet_3d_cfm_configs.channels,
            resblock_updown=unet_3d_cfm_configs.resblock_updown,
            attention_levels=unet_3d_cfm_configs.attention_levels,
            num_head_channels=unet_3d_cfm_configs.num_head_channels,
            norm_num_groups=unet_3d_cfm_configs.norm_num_groups,
            use_flash_attention=True,
        )

    if model_checkpoint is None:
        light_model = UnetLightning3D(DiffusionModelUNet, unet_kwargs, paired_input=True, lr=lr, output_dir=run_output_dir, input_channels=unet_3d_cfm_configs.in_channels, decode_model=decode_model, img_size=training_args.img_size, bbox_file=train_bboxes, time_context_dim=training_args.time_context_dim)
    else:
        light_model = UnetLightning3D.load_from_checkpoint(
                                                    model_checkpoint,
                                                    unet_cls=DiffusionModelUNet,
                                                    decode_model=decode_model,
                                                    strict=False, 
                                                    paired_input=True,
                                                    output_dir=run_output_dir,
                                                    bbox_file=train_bboxes,
                                                    time_context_dim=training_args.time_context_dim
                                                )

    ## Loggers and Callbacks ##
    csv_logger = CSVLogger(save_dir=run_output_dir)
    wandb_logger = WandbLogger(
                                project=training_args.wandb_project,
                                name=training_args.wandb_run_name,
                                config=unet_3d_cfm_configs.__dict__ | training_args.__dict__,
                            )
    
    checkpointer_train = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(csv_logger.log_dir, "checkpoints"),
        filename='train-{epoch}-{train_loss:.2f}',
        save_top_k=1,
        mode="min",
    )

    checkpointer_val = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(csv_logger.log_dir, "checkpoints"),
        filename='val-{epoch}-{val_loss:.2f}',
        save_top_k=1,
        mode="min",
        save_last=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    pl.seed_everything(42, workers=True)
    if debug_flag:
        trainer = pl.Trainer(logger=[csv_logger, wandb_logger], 
                        devices=3, 
                        deterministic=False, 
                        accelerator="auto",
                        callbacks=[checkpointer_train, lr_monitor], 
                        log_every_n_steps=10,
                        max_epochs=2,
                        num_sanity_val_steps=1,
                        gradient_clip_val=1.0, 
                        gradient_clip_algorithm="value",
                        strategy=DDPStrategy(),) 
    else:      
        trainer = pl.Trainer(logger=[csv_logger, wandb_logger],
                            devices=training_args.num_devices, 
                            deterministic=False, 
                            # accumulate_grad_batches=4,
                            accelerator="auto",
                            callbacks=[checkpointer_train, checkpointer_val, lr_monitor], 
                            log_every_n_steps=50,
                            max_epochs=training_args.max_epochs,
                            num_sanity_val_steps=2,
                            gradient_clip_val=1.0, 
                            gradient_clip_algorithm="norm",
                            strategy=DDPStrategy(),
                            check_val_every_n_epoch=1,
                            limit_val_batches=10,
                            precision="bf16-mixed") # I only want it to validate one batch each time
                       
    trainer.fit(model=light_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        log_dir = trainer.logger.log_dir
        path = os.path.join(log_dir, "metrics.csv")
        fig = plot_lr_from_metrics(path, show=False)
        fig2 = plot_loss_from_metrics(path, show=False)

def create_datasets(base_paths, encoded_data=False):
    train_data_parquet = base_paths.full_data_train_parquet
    val_data_parquet = base_paths.full_data_val_parquet
    test_data_parquet = base_paths.full_data_test_parquet

    train_nodule_parquet = base_paths.bounding_boxes_train_parquet
    val_nodule_parquet = base_paths.bounding_boxes_val_parquet
    test_nodule_parquet = base_paths.bounding_boxes_test_parquet

    train_paired_parquet = base_paths.paired_nodules_train_parquet
    val_paired_parquet = base_paths.paired_nodules_val_parquet
    test_paired_parquet = base_paths.paired_nodules_test_parquet

    if not encoded_data:   
        nodule_index = base_paths.raw_nodule_index
        data_root = base_paths.raw_cached_nodule_dir
    else:
        nodule_index = base_paths.encoded_nodule_index
        data_root = base_paths.encoded_cached_nodule_dir

    train_dataset = CachedNoduleDataset(train_data_parquet, 
                                        train_nodule_parquet, 
                                        nodule_index, 
                                        data_root, 
                                        paired_nodule_parquet=train_paired_parquet,
                                        mode="paired",
                                        split="train",
                                        max_length=num_samples,
                                        max_cache_size=50,
                                        return_meta_data=True)
    
    val_dataset = CachedNoduleDataset(val_data_parquet, 
                                      val_nodule_parquet, 
                                      nodule_index, 
                                      data_root, 
                                      paired_nodule_parquet=val_paired_parquet,
                                      mode="paired",
                                      split="val",
                                      max_length=num_samples,
                                      max_cache_size=10,
                                      return_meta_data=True)
    
    test_dataset = CachedNoduleDataset(test_data_parquet, 
                                       test_nodule_parquet, 
                                       nodule_index, 
                                       data_root,
                                       paired_nodule_parquet=test_paired_parquet,
                                       mode="paired",
                                       split="test",
                                       max_length=5,
                                       max_cache_size=10,
                                       return_meta_data=True)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    single_image = False  # set to True to train on a single image repeated
    repeat_count = 20000
    path_yaml = "configs/paths.yaml"
    training_configs = "configs/fm_3d_paired.yaml"
    # unet_3d_cfm = "configs/unet_3d_cfm.yaml"
    unet_3d_cfm = "configs/unet_3d_cfm_raw.yaml"
    encode_data = False
    decode_model_checkpoint = None # "/data/rbg/users/duitz/CT-generative-pred/final_saved_models/vae_fixed_std_no_reg.pt"
    pretrained_model_checkpoint = None # "/data/rbg/users/duitz/CT-generative-pred/experiments/fm_3d_pretrain/full_latent/lightning_logs/version_1/checkpoints/val-epoch=429-val_loss=0.32.ckpt"
    train_bboxes = "/data/rbg/users/duitz/CT-generative-pred/metadata/train_raw_data_nodule_original_boxes.json"

    base_paths = load_config(path_yaml)
    training_args = load_config(training_configs)

    num_samples = training_args.num_samples
    experiment_name = training_args.experiment_name

    experiment_dir = base_paths.paired_3d_fm_experiment_dir
    run_output_dir = os.path.join(experiment_dir, experiment_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # Copy over config file
    unet_config_copy_dst = os.path.join(run_output_dir, f"{experiment_name}_unet_config.yaml")
    train_config_copy_dst = os.path.join(run_output_dir, f"{experiment_name}_train_config.yaml")
    shutil.copy2(unet_3d_cfm, unet_config_copy_dst)
    shutil.copy2(training_configs, train_config_copy_dst)

    if encode_data:
        decode_model = CustomVAE.load_from_checkpoint(decode_model_checkpoint, map_location="cpu")
        model = None
    else:
        model = None
        decode_model = None

    train_dataset, val_dataset, test_dataset = create_datasets(base_paths, encoded_data=encode_data)

    main(pretrained_model_checkpoint, 
         training_args, 
         unet_3d_cfm, 
         train_dataset, 
         val_dataset, 
         test_dataset, 
         run_output_dir, 
         decode_model=decode_model, 
         single_image=single_image, 
         repeat_count=repeat_count, 
         train_bboxes=train_bboxes)


