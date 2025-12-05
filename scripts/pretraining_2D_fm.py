# This file is used to run training of 2D diffusion model on unconditional
# generation of CT slices from NLST dataset

import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchcfm.conditional_flow_matching import *
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
import argparse
import pickle

# This repo imports
from CTFM.models import UNetModel
from CTFM.utils import plot_lr_from_metrics, plot_loss_from_metrics
from CTFM.models import UnetLightning
from CTFM.data import Encoded2DSliceDataset
from CTFM.data import CTOrigDataset2D, RepeatedImageDataset
from CTFM.utils import load_config

## Additional imports if needed ##
from torchinfo import summary
from torchvision.utils import save_image
from torch.utils.data import Dataset
import numpy as np
from lightning.pytorch.utilities import rank_zero_only #@rank_zero_only

from CTFM.utils import window_ct_hu_to_png
from CTFM.data import reverse_normalize

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Has a --debug flag (defaults to False)."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging (default: off)."
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        default=500,
        help="Number of samples to use (default: 10000). None is all samples."
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
        help="Training batch size (default: 64)."
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Testing batch size (default: 128)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate (default: 0.001)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="encoded",
        choices=["encoded", "original"],
        help="Choose dataset type: encoded or original"
    )
    return parser.parse_args(argv)



def main(argv=None):
    path_yaml = "configs/paths.yaml"
    run_output_dir = "/data/rbg/users/duitz/CT-generative-pred/outputs/outputs_2D_cfm_pretrain"
    os.makedirs(run_output_dir, exist_ok=True)
    
    args = parse_args(argv)

    debug_flag = args.debug
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    num_samples = args.samples

    base_paths = load_config(path_yaml)
    full_data_parquet = base_paths.full_data_train_parquet
    data_root = base_paths.encoded_dir_2d
    encoded_index_name = os.path.join(data_root, "meta", "index.parquet")
    saved_transforms_file = base_paths.saved_transforms_file
    with open(saved_transforms_file, 'rb') as f:
        saved_transforms = pickle.load(f)

    if debug_flag:
        global_n = 5
    else:
        global_n = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## Dataset Selection ##
    if args.dataset == "encoded":
        dataset = Encoded2DSliceDataset(full_data_parquet=full_data_parquet, 
                                        encoded_index_parquet=encoded_index_name,
                                        encoded_root=data_root,
                                        split='train',
                                        global_n=global_n,
                                        max_length=num_samples,
                                        )
    else:
        if global_n is not None:
            global_n = num_samples
        dataset = CTOrigDataset2D(parquet_path=full_data_parquet,
                                  device=device,
                                  saved_transforms=saved_transforms,
                                  max_length=global_n,)
        
    if debug_flag:
        # import pdb; pdb.set_trace()
        train_batch_size = 1
        test_batch_size = 1
        test_size = 1
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        test_size = 8
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers = 2, drop_last=True) #, timeout=30)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    ####### For Single Image Training ########
    # image = train_dataset[0][0]
    # repeat_count = 10000
    # train_dataset = RepeatedImageDataset(image, repeat_count)
    # test_dataset = RepeatedImageDataset(image, 4)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    # reverse_normalized_image = reverse_normalize(image.cpu(), clip_window=(-1300, 400))
    # image_to_save = window_ct_hu_to_png(reverse_normalized_image, center=-600.0, width=1500.0, bit_depth=8)
    # os.makedirs("outputs_test_single", exist_ok=True)
    # save_image(image_to_save, "outputs_test_single/goal_image.png")

    ##################################

    ## Model Creation ##
    cfg = load_config("configs/unet_2d_cfm.yaml")

    model = UNetModel(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        model_channels=cfg.model_channels,
        out_channels=cfg.out_channels,
        channel_mult=tuple(cfg.channel_mult),
        use_fp16=cfg.use_fp16,
        use_checkpoint=cfg.use_checkpoint,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
    )

    light_model = UnetLightning(model, lr=lr, output_dir=run_output_dir)

    logger = CSVLogger(save_dir=os.getcwd())
    checkpointer = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename='{epoch}-{train_loss:.2f}',
        save_top_k=1,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    pl.seed_everything(42, workers=True)
    if debug_flag:
        trainer = pl.Trainer(logger=logger,
                        devices=1, 
                        deterministic=True, 
                        accelerator="auto",
                        callbacks=[checkpointer, lr_monitor], 
                        log_every_n_steps=1,
                        max_epochs=2,
                        num_sanity_val_steps=1,
                        gradient_clip_val=1.0, 
                        gradient_clip_algorithm="value",
                        strategy=DDPStrategy(),) 
    else:               
        trainer = pl.Trainer(logger=logger,
                            devices=8, 
                            deterministic=True, 
                            accelerator="auto",
                            callbacks=[checkpointer, lr_monitor], 
                            log_every_n_steps=1,
                            max_epochs=20,
                            num_sanity_val_steps=2,
                            gradient_clip_val=1.0, 
                            gradient_clip_algorithm="value",
                            strategy=DDPStrategy(),
                            check_val_every_n_epoch=1,
                            limit_val_batches=1,) # I only want it to validate one batch each time
                       
    trainer.fit(model=light_model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    best_path = checkpointer.best_model_path

    trainer.test(dataloaders=test_loader, ckpt_path=best_path)
    
    if trainer.is_global_zero:
        log_dir = trainer.logger.log_dir
        path = os.path.join(log_dir, "metrics.csv")
        fig = plot_lr_from_metrics(path, show=False)
        fig2 = plot_loss_from_metrics(path, show=False)
    

if __name__ == "__main__":
    main()

