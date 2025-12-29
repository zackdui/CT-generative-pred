# This file is used to run training of 3D diffusion model on unconditional
import os
import sys
from monai.networks.nets import DiffusionModelUNet
# from generative.networks.nets import DiffusionModelUNet
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import *
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torchinfo import summary

sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE

# This repo imports
from CTFM.data import (CachedNoduleDataset, 
                       CTNoduleDataset3D, 
                       RepeatedImageDataset, 
                       save_slices, 
                       collate_image_meta, 
                       reverse_normalize)
from CTFM.utils import (plot_lr_from_metrics, 
                        plot_loss_from_metrics, 
                        load_config, 
                        window_ct_hu_to_png)
from CTFM.models import UnetLightning3D


def main(training_args, 
         unet_3d_cfm, 
         train_dataset, 
         val_dataset, 
         test_dataset, 
         run_output_dir, 
         single_image=False):

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

    ####### For Single Image Training ########
    if single_image:
        image = train_dataset[0][0]
        meta_data = train_dataset[0][1]
        repeat_count = 10
        train_dataset = RepeatedImageDataset(image, repeat_count, meta_data=meta_data)
        val_dataset = RepeatedImageDataset(image, 4, meta_data=meta_data)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_image_meta)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_image_meta)
        reverse_normalized_image = reverse_normalize(image.cpu())
        image_to_save = window_ct_hu_to_png(reverse_normalized_image, center=-600.0, width=1500.0, bit_depth=8)
        single_image_output_dir = os.path.join(run_output_dir, "single_image_example")
        os.makedirs(single_image_output_dir, exist_ok=True)
        save_slices(image_to_save, output_dir=single_image_output_dir, prefix="single_image_goal_")
    ##################################

    ## Model Creation ##
    unet_3d_cfm_configs = load_config(unet_3d_cfm)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = DiffusionModelUNet(spatial_dims=unet_3d_cfm_configs.spatial_dims,
                           in_channels=unet_3d_cfm_configs.in_channels,
                           out_channels=unet_3d_cfm_configs.out_channels,
                           num_res_blocks=unet_3d_cfm_configs.num_res_blocks,
                           channels=unet_3d_cfm_configs.channels,
                           resblock_updown=unet_3d_cfm_configs.resblock_updown,
                           attention_levels=unet_3d_cfm_configs.attention_levels,
                           num_head_channels=unet_3d_cfm_configs.num_head_channels,
                           norm_num_groups=unet_3d_cfm_configs.norm_num_groups,
                           use_flash_attention=True)

    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in model: {model_parameters}")

    light_model = UnetLightning3D(model, lr=lr, output_dir=run_output_dir, input_channels=1)
    
    if debug_flag:
        torchinfo_summary = summary(light_model, input_size=[(8, 1, 32, 128, 128), (8,)])
        print(torchinfo_summary)

     ## Loggers and Callbacks ##
    csv_logger = CSVLogger(save_dir=run_output_dir)
    wandb_logger = WandbLogger(
                                project=training_args.wandb_project,
                                name=training_args.wandb_run_name,
                                config=unet_3d_cfm_configs.__dict__ | training_args.__dict__ | {'model_parameters': model_parameters},
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
        save_last=True,
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
                            devices=1, 
                            deterministic=False, 
                            accelerator="auto",
                            callbacks=[checkpointer_train, checkpointer_val, lr_monitor], 
                            log_every_n_steps=20,
                            max_epochs=20,
                            num_sanity_val_steps=2,
                            gradient_clip_val=1.0, 
                            gradient_clip_algorithm="value",
                            strategy=DDPStrategy(),
                            check_val_every_n_epoch=1,
                            limit_val_batches=1,
                            profiler="simple",
                            precision="bf16-mixed") # I only want it to validate one batch each time
                       
    trainer.fit(model=light_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpointer_val.best_model_path

    trainer.test(dataloaders=test_loader, ckpt_path=best_path)
    
    if trainer.is_global_zero:
        log_dir = trainer.logger.log_dir
        path = os.path.join(log_dir, "metrics.csv")
        fig = plot_lr_from_metrics(path, show=False)
        fig2 = plot_loss_from_metrics(path, show=False)

def create_datasets(base_paths, max_examples, encode_model=None):
    train_data_parquet = base_paths.full_data_train_parquet
    train_nodule_parquet = base_paths.bounding_boxes_train_parquet
    raw_nodule_index = base_paths.raw_nodule_index
    data_root = base_paths.raw_cached_nodule_dir

    val_data_parquet = base_paths.full_data_val_parquet
    val_nodule_parquet = base_paths.bounding_boxes_val_parquet

    test_data_parquet = base_paths.full_data_test_parquet
    test_nodule_parquet = base_paths.bounding_boxes_test_parquet

    train_dataset = CachedNoduleDataset(train_data_parquet, 
                                        train_nodule_parquet, 
                                        raw_nodule_index, 
                                        data_root, 
                                        split="train",
                                        max_length=num_samples,
                                        max_cache_size=50,
                                        return_meta_data=True)
    
    val_dataset = CachedNoduleDataset(val_data_parquet, 
                                      val_nodule_parquet, 
                                      raw_nodule_index, 
                                      data_root, 
                                      split="val",
                                      max_length=num_samples,
                                      max_cache_size=10,
                                      return_meta_data=True)
    
    test_dataset = CachedNoduleDataset(test_data_parquet, 
                                       test_nodule_parquet, 
                                       raw_nodule_index, 
                                       data_root, 
                                       split="test",
                                       max_length=5,
                                       max_cache_size=10,
                                       return_meta_data=True)
    
    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, encoder=None, device="cuda"):
            self.base_ds = base_ds
            self.encoder = encoder.eval()
            self.device = device

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            x, meta = self.base_ds[idx]
            # IMPORTANT: do NOT put GPU encode here if using num_workers>0
            if self.encoder is not None:
                with torch.no_grad():
                    x = x.unsqueeze(0).to(self.device)  # add batch dim
                    z, _, _, _ = self.encoder.encode(x)
                    x = z.squeeze(0).cpu()  # remove batch dim
            return x, meta
    if encode_model is not None:
        train_dataset = EncodedDataset(train_dataset, encoder=encode_model)
        val_dataset = EncodedDataset(val_dataset, encoder=encode_model)
        test_dataset = EncodedDataset(test_dataset, encoder=encode_model)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    single_image = True  # set to True to train on a single image repeated
    experiment_name = "test_more_steps" #"testing_functionality2"
    path_yaml = "configs/paths.yaml"
    training_configs = "configs/fm_3d_pretraining.yaml"
    unet_3d_cfm = "configs/unet_3d_cfm.yaml"
    encode_data = False
    encode_model_checkpoint = "/data/rbg/users/duitz/CT-generative-pred/final_saved_models/ae3d_16_3down.pt"
    

    base_paths = load_config(path_yaml)
    training_args = load_config(training_configs)

    num_samples = training_args.num_samples

    experiment_dir = base_paths.pretraining_3d_fm_experiment_dir
    run_output_dir = os.path.join(experiment_dir, experiment_name)
    os.makedirs(run_output_dir, exist_ok=True)

    if encode_data:
        encode_model = CustomVAE.load_from_checkpoint(encode_model_checkpoint, map_location="cuda")
    else:
        encode_model = None

    train_dataset, val_dataset, test_dataset = create_datasets(base_paths, num_samples, encode_model)
    
    main(training_args, unet_3d_cfm, train_dataset, val_dataset, test_dataset, run_output_dir, single_image=single_image)

