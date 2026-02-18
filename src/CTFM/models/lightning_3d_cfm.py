from torchcfm.conditional_flow_matching import *
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch import tensor
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from typing import Type
import json
from ..data import (reverse_normalize, 
                    save_slices, 
                    save_side_by_side_slices, 
                    save_mp4, 
                    safe_delete,
                    save_montage)
from ..utils import window_ct_hu_to_png, prepare_for_wandb_hu, prepare_for_wandb, volume_to_gif_frames

import time

class UnetLightning3D(pl.LightningModule):
    def __init__(self, 
                 unet_cls: Type[nn.Module], 
                 model_hyperparameters,
                 paired_input=False,
                 lr=.001, 
                 sigma = 0.1, 
                 output_dir="outputs_random", 
                 input_channels = 1, 
                 img_size = (32, 128, 128), 
                 decode_model=None,
                 num_val_images = 1,
                 dummy_image=None, 
                 convert_from_hu=True,
                 debug_flag=False, 
                 bbox_file = None,
                ):
        """
            model: pytorch modeal - input unet model 
            paired_input: Bool - Whether the input is paired images concatenated along channel dimension
            lr: float - optimizer learning rate (In prectice there is a warmup and cosine annealing) 
            sigma: float - flow matching sigma 
            output_dir: str - directory to output validation and test samples
            input_channels: int - number of channels in the input image
            img_size: tuple(int, int, int) - image size
            decode_model: pytorch model with encode and decode functions or None -
                default is AutoEncoder_Lightning(): If it is None then no model will be used to decode
            num_val_images: int - number of validation images to generate in the validation step
                For paired input this can only go up to batch size
            dummy_image=None, 
            debug_flag=False,
            convert_from_hu: bool = True - If the images will originally be in hu values that need to be converted
            bbox_file: str or None - if the path is provided it will heavily weight the loss around the bounding boxes
                This should only be used when passing in the original volumes not the encoded ones
        """
        super().__init__()
        self.save_hyperparameters(ignore=['unet_cls', 'decode_model', 'dummy_image'])
        self.model = unet_cls(**model_hyperparameters)
        model_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in model: {model_parameters}")
        self.paired_input = paired_input
        if self.paired_input:
            self.FM = ConditionalFlowMatcher(sigma=sigma)
        else:
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.output_dir = output_dir
        self.lr = lr
        self.image_size = img_size
        self.input_channels = input_channels
        self.num_val_images = num_val_images
        self.dummy_image = dummy_image
        self.convert_from_hu = convert_from_hu
        self.debug_flag = debug_flag

        # Used for inference only if gernerating a latent space and then decoding it
        self.decode_model = decode_model
        if self.decode_model is not None:
            self.decode_model.eval()
            for param in self.decode_model.parameters():
                param.requires_grad = False

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.validation_path = os.path.join(self.output_dir, "validation_images")
        print(f"Creating validation directory at: {self.validation_path}")
        os.makedirs(self.validation_path, exist_ok=True)

        if dummy_image is not None:
            save_slices(dummy_image, output_dir=self.output_dir, prefix="dummy_image_input")

        self.fixed_noise = None
        self._wandb_logger = None
        self._wandb_exp = None

        if bbox_file is not None:
            with open(bbox_file, "r") as f:
                self.bboxes = json.load(f)
        else:
            self.bboxes = None

    def on_fit_start(self) -> None:
        if not self.trainer.is_global_zero:
            return

        lg = self.trainer.loggers  # safe here

        wlogger = None

        if isinstance(lg, WandbLogger):
            wlogger = lg
        elif isinstance(lg, (list, tuple)):
            for x in lg:
                if isinstance(x, WandbLogger):
                    wlogger = x
                    break
        elif isinstance(lg, (list, tuple)):  # just in case
            for x in lg:
                if isinstance(x, WandbLogger):
                    wlogger = x
                    break

        self._wandb_logger = wlogger
        self._wandb_exp = wlogger.experiment if wlogger is not None else None

    def forward(self, xt, t, y=None):

        ## Given a single imput image get its actual flow
        # if self.dummy_image is not None:
        #     z = self.dummy_image.to(xt.device)
        #     while t.ndim < xt.ndim: t.unsqueeze_(-1)
        #     return (z - xt)/torch.clip(1-t, min=1e-2)
        # Optional change to t
        # t = t * 1000
        if y:
            return self.model(xt, t, y)
        return self.model(xt, t)


    def training_step(self, batch, batch_idx):
        image_data, meta_data = batch
        if self.paired_input:
            x0 = image_data[:, 0:self.input_channels, :, :, :]
            x1 = image_data[:, self.input_channels:, :, :, :]
        else:
            x1 = image_data
            x0 = torch.randn_like(x1)
            # if self.fixed_noise is None:
            #     self.fixed_noise = torch.randn_like(x1).detach()
            # x0 = self.fixed_noise
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        if self.debug_flag:
            # CUDA event timing (accurate)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()


        vt = self(xt, t)

        if self.debug_flag:
            end.record()
            torch.cuda.synchronize()
            fwd_ms = start.elapsed_time(end)

            self.log("debug/fwd_ms", fwd_ms, prog_bar=True, on_step=True, logger=True, sync_dist=False)

        if self.bboxes is not None:
            B, _, D, H, W = vt.shape
            margin = 4
            w_bg, w_roi = 1.0, 10.0
            w = torch.full((B, 1, D, H, W), w_bg, device=vt.device)

            for b in range(B):

                nodule_id_a = f"{meta_data[b]['nodule_group_a']}_{meta_data[b]['exam_a']}_{meta_data[b]['exam_idx_a']}"
                nodule_id_b = f"{meta_data[b]['nodule_group_b']}_{meta_data[b]['exam_b']}_{meta_data[b]['exam_idx_b']}"
                h0a,h1a,w0a,w1a,d0a,d1a = self.bboxes[nodule_id_a]["bbox"]
                h0b,h1b,w0b,w1b,d0b,d1b = self.bboxes[nodule_id_b]["bbox"]

                h0 = max(0, min(h0a, h0b) - margin); h1 = min(H, max(h1a, h1b) + margin)
                w0 = max(0, min(w0a, w0b) - margin); w1 = min(W, max(w1a, w1b) + margin)
                d0 = max(0, min(d0a, d0b) - margin); d1 = min(D, max(d1a, d1b) + margin)

                w[b, :, d0:d1, h0:h1, w0:w1] = w_roi

            loss = (w * (vt - ut).pow(2)).sum() / (w.sum() + 1e-8)
        else:
            loss = torch.mean((vt - ut) ** 2)

        
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr/current", lr, on_step=True, logger=True, prog_bar=False, sync_dist=False)
        self.log("step_train_loss", loss.detach().float(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        self.log("train_loss", loss.detach().float(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
        
    
    def configure_optimizers(self):
        """ Configure the optimizer and learning rate scheduler """
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = total_steps // 10
        decay_steps = total_steps - warmup_steps

        # Only include the model we are training not the decoding model
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # The optimizer will include the max learning rate
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)  # start_factor must > 0
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=5e-5)
        sched  = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    def generate_images(self, x_input: tensor, n_steps=100):
        """ 
        Generate images from the model given an input image x_input 
        x_input: tensor - input to generate from
        n_steps: int - number of steps
        """
        t_vec = torch.linspace(0, 1, n_steps+1, device=x_input.device)
        x_t = x_input
        # x_t = self.fixed_noise
        B = x_t.shape[0]

        for idx, t0 in enumerate(t_vec[:-1]):
            t1 = t_vec[idx+1]
            dt = t1 - t0
            t = t0.expand(B)
            v = self(x_t, t)
            x_t = x_t + v * dt
        return x_t
    

    @torch.no_grad()
    def generate_images_rk2(self, x_input: Tensor, n_steps: int = 50) -> Tensor:
        """
        RK2 (Heun) integrator for dx/dt = v(x,t).
        Integrates t from 0 -> 1 with n_steps steps.

        Note: currently ignores x_input and starts from self.fixed_noise like your code.
        """
        # Start state
        # x_t = self.fixed_noise  # shape (B, C, D, H, W)
        x_t = x_input
        B = x_t.shape[0]

        t_vec = torch.linspace(0.0, 1.0, n_steps + 1, device=x_t.device, dtype=x_t.dtype)

        for t0, t1 in zip(t_vec[:-1], t_vec[1:]):
            dt = t1 - t0
            t0b = t0.expand(B)
            t1b = t1.expand(B)

            # k1 = v(x, t0)
            k1 = self(x_t, t0b)

            # predictor: x_euler = x + dt*k1
            x_euler = x_t + dt * k1

            # k2 = v(x_euler, t1)
            k2 = self(x_euler, t1b)

            # Heun update: x_{t+dt} = x + dt * (k1 + k2)/2
            x_t = x_t + dt * 0.5 * (k1 + k2)

        return x_t

    @torch.no_grad()
    def generate_images_rk4(self, x_input: Tensor, n_steps: int = 50) -> Tensor:
        """
        Classic RK4 integrator for dx/dt = v(x,t).
        Integrates t from 0 -> 1 with n_steps steps.

        Note: currently ignores x_input and starts from self.fixed_noise like your code.
        """
        # x_t = self.fixed_noise
        x_t = x_input
        B = x_t.shape[0]

        t_vec = torch.linspace(0.0, 1.0, n_steps + 1, device=x_t.device, dtype=x_t.dtype)

        for t0, t1 in zip(t_vec[:-1], t_vec[1:]):
            dt = t1 - t0
            th = t0 + 0.5 * dt  # half-step time

            t0b = t0.expand(B)
            thb = th.expand(B)
            t1b = t1.expand(B)

            k1 = self(x_t, t0b)
            k2 = self(x_t + 0.5 * dt * k1, thb)
            k3 = self(x_t + 0.5 * dt * k2, thb)
            k4 = self(x_t + dt * k3, t1b)

            x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return x_t
    
    def save_wandb(self, input, output, target=None):
        if not self.trainer.is_global_zero:
            return
        if self._wandb_exp is None:
            return
        
        for i, (input_img, output_img) in enumerate(zip(input, output)):
            # Remove the channel Dimension
            input_dhw = input_img[0]
            output_dhw = output_img[0]
            mid = input_dhw.shape[0] // 2

            # Save Approximate montage. Fix for correct scaling
            fig = save_montage(input_dhw, out_path=None, save_fig=False, return_fig=True)
            fig2 = save_montage(output_dhw, out_path=None, save_fig=False, return_fig=True)

            self._wandb_exp.log({
                "images/input_montage": wandb.Image(fig),
                "images/output_montage": wandb.Image(fig2),
            }, step=self.global_step)

            plt.close(fig)
            plt.close(fig2)

            if self.convert_from_hu:
                x_slice = prepare_for_wandb_hu(input_dhw[mid])
                out_slice = prepare_for_wandb_hu(output_dhw[mid])
            else:
                x_slice = prepare_for_wandb(input_dhw[mid])
                out_slice = prepare_for_wandb(output_dhw[mid])

            self._wandb_exp.log({
                "images/inputs_mid_slice": wandb.Image(x_slice),
                "images/outputs_mid_slice": wandb.Image(out_slice),
            }, step=self.global_step)

            if target is not None:
                target_dhw = target[i][0]
                if self.convert_from_hu:
                    target_slice = prepare_for_wandb_hu(target_dhw[mid])
                else:
                    target_slice = prepare_for_wandb(target_dhw[mid])
                self._wandb_exp.log({
                    "images/targets_mid_slice": wandb.Image(target_slice),
                }, step=self.global_step)

            # ---- 3D scrollable GIF: input volume ----
            input_frames = volume_to_gif_frames(input_dhw, every_n=2)  # every 2 slices, tweak as needed
            input_mp4_path = save_mp4(input_frames, fps=10)

            # ---- 3D scrollable GIF: side-by-side (input | recon) ----
            recon_frames = volume_to_gif_frames(output_dhw, every_n=2)
            output_mp4_path = save_mp4(recon_frames, fps=10)
            # ensure same length
            n_frames = min(len(input_frames), len(recon_frames))
            side_by_side_frames = [
                np.concatenate([input_frames[i], recon_frames[i]], axis=1)
                for i in range(n_frames)
            ]
            side_mp4_path = save_mp4(side_by_side_frames, fps=10)

            self._wandb_exp.log({
                "videos/volume_input_mp4": wandb.Video(input_mp4_path, format="mp4"),
                "videos/volume_input_by_output_mp4": wandb.Video(side_mp4_path, format="mp4"),
                "videos/volume_output_mp4": wandb.Video(output_mp4_path, format="mp4"),
            }, step=self.global_step)

            safe_delete(input_mp4_path)
            safe_delete(side_mp4_path)
            safe_delete(output_mp4_path)


            if target is not None:
                target_dhw = target[i][0]
                target_frames = volume_to_gif_frames(target_dhw, every_n=2, is_hu=self.convert_from_hu)
                target_mp4_path = save_mp4(target_frames, fps=10)

                self._wandb_exp.log({
                    "videos/volume_target_mp4": wandb.Video(target_mp4_path, format="mp4"),
                }, step=self.global_step)

                safe_delete(target_mp4_path)

                fig3 = save_montage(target_dhw, out_path=None, save_fig=False, return_fig=True)
                self._wandb_exp.log({
                    "images/targets_montage": wandb.Image(fig3),
                }, step=self.global_step)
                plt.close(fig3)
    
    def save_predictions(self, x0, x1, output_dir: str, prefix: str):
        if self.paired_input:
            x_input = x0[:self.num_val_images]
            x_target = x1[:self.num_val_images]
        else:
            x_input = torch.randn(self.num_val_images, x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4]).to(x1.device)
            x_target = None

        x_output = self.generate_images(x_input)

        images = x_output.view([-1, self.input_channels, self.image_size[0], self.image_size[1], self.image_size[2]]) 

        if self.decode_model is not None:
            with torch.no_grad():
                images = self.decode_model.decode(images)
                x_input = self.decode_model.decode(x_input)
                if self.paired_input:
                    x_target = self.decode_model.decode(x_target)

        images = images.clip(-1, 1)
        x_input = x_input.clip(-1, 1)

        # torch.cuda.synchronize()  # ensure previous CUDA ops finish
        # t0 = time.perf_counter()

        self.save_wandb(x_input, images, x_target if self.paired_input else None)

        # torch.cuda.synchronize()  # ensure save_wandb CUDA work finishes
        # dt = time.perf_counter() - t0

        # print(f"[timing] save_wandb took {dt:.3f}s")
        for i, (input_img, output_img) in enumerate(zip(x_input, images)):
            if self.convert_from_hu:
                input_img_hu = reverse_normalize(input_img, clip_window=(-2000, 500))
                input_img = window_ct_hu_to_png(input_img_hu, center=-600, width=1500, bit_depth=8)
            input_slices = save_slices(input_img, output_dir=output_dir, prefix=f"{prefix}_sample_{i}_input_image")
            out_path_input = f"{output_dir}/{prefix}_sample_{i}_input_montage.png"
            save_montage(input_img, out_path=out_path_input)
            # Save the output images
            if self.convert_from_hu:
                output_img_hu = reverse_normalize(output_img, clip_window=(-2000, 500))
                output_img = window_ct_hu_to_png(output_img_hu, center=-600, width=1500, bit_depth=8)
                output_slices = save_slices(output_img, output_dir=output_dir, prefix=f"{prefix}_sample_{i}_sample_out")
            else:
                output_slices = save_slices(output_img, output_dir=output_dir, prefix=f"{prefix}_sample_{i}_sample_out")
            out_path_output = f"{output_dir}/{prefix}_sample_{i}_output_montage.png"
            save_montage(output_img, out_path=out_path_output)
            save_side_by_side_slices(input_slices, output_slices, output_dir=output_dir, prefix=f"{prefix}_sample_{i}_input_vs_output")
            # Save the target images if paired input
            if self.paired_input:
                target_img = x_target[i]
                target_img = target_img.clip(-1, 1)
                if self.convert_from_hu:
                    target_img_hu = reverse_normalize(target_img, clip_window=(-2000, 500))
                    target_img = window_ct_hu_to_png(target_img_hu, center=-600, width=1500, bit_depth=8)
                    save_slices(target_img, output_dir=output_dir, prefix=f"{prefix}_sample_{i}_target_image")
                else:
                    save_slices(target_img, output_dir=output_dir, prefix=f"{prefix}_sample_{i}_target_image")
                save_montage(target_img, out_path=f"{output_dir}/{prefix}_sample_{i}_target_montage.png")

    
    def validation_step(self, batch, batch_idx):
        """
        Validation step will both evaluate the loss on unseen data and generate sample images to monitor the models progress
        """
        image_data, meta_data = batch
        global_step = self.global_step
        rank = self.global_rank


        # First Compute the loss
        if self.paired_input:
            x0 = image_data[:, 0:self.input_channels, :, :, :]
            x1 = image_data[:, self.input_channels:, :, :, :]
        else:
            x1 = image_data
            x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        vt = self(xt, t)
        val_loss = torch.mean((vt - ut) ** 2)
        self.log("val_loss", val_loss.detach().float(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Now save the output of the model on a test image
        # Only want return the evaluation image on the first gpu
        if rank != 0:
            return val_loss
        # Only save on the first batch of each epoch. Sanity checking is to avoid saving on the sanity check
        if not self.trainer.sanity_checking and batch_idx == 0:
            # print("Starting predictions")
            # torch.cuda.synchronize()  # ensure previous CUDA ops finish
            # t0 = time.perf_counter()
            self.save_predictions(x0, x1, output_dir=self.validation_path, prefix=f"validation_step={global_step}_rank={rank}")

            # torch.cuda.synchronize()  # ensure save_wandb CUDA work finishes
            # dt = time.perf_counter() - t0

            # print(f"Prediction [timing] save_wandb took {dt:.3f}s")
       
        return val_loss
        
    def test_step(self, batch, batch_idx):
        image_data, meta_data = batch
        if self.paired_input:
            x_gauss = image_data[:, 0:self.input_channels, :, :, :]
            x_batch = image_data[:, self.input_channels:, :, :, :]
        else:
            x_batch = image_data
            x_gauss = torch.randn_like(x_batch)

        self.save_predictions(x_gauss, x_batch, output_dir=self.output_dir, prefix=f"final_test_batch={batch_idx}_rank={self.trainer.global_rank}")

        

