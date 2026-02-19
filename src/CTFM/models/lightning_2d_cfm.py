from torchcfm.conditional_flow_matching import *
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.utils import save_image
from .unet_2d.unet import UNetModel
from torchinfo import summary
import torch.nn as nn
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor
from CTFM.utils.plot_lr import plot_lr_from_metrics
from lightning.pytorch.utilities import rank_zero_only
from torch import tensor
from .auto_encoder_2d import AutoEncoder_Lightning
from ..data import reverse_normalize
from ..utils import window_ct_hu_to_png



class UnetLightning(pl.LightningModule):
    def __init__(self, 
                 model, 
                 paired_input=False,
                 lr=.001, 
                 sigma = 0.1, 
                 output_dir="outputs_random", 
                 input_channels = 4, 
                 img_size = (64, 64), 
                 decode_model=AutoEncoder_Lightning(),
                 num_val_images = 1,
                #  use_gauss_input = True,
                 dummy_image=None, 
                 convert_to_hu=True,
                 debug_flag=False, 
                ):
        """
            model: pytorch modeal - input unet model 
            paired_input: Bool - Whether the input is paired images concatenated along channel dimension
            lr: float - optimizer learning rate (In prectice there is a warmup and cosine annealing) 
            sigma: float - flow matching sigma 
            output_dir: str - directory to output validation and test samples
            input_channels: int - number of channels in the input image 
            img_size: tuple(int, int) - image size 
            decode_model: pytorch model with encode and decode functions or None - 
                default is AutoEncoder_Lightning(): If it is None then no model will be used to decode
            num_val_images: int - number of validation images to generate in the validation step
            # use_gauss_input: Bool - Whether to input gaussain noise or actual images for validaiton and test
            #     True inputs gaussian noise
            dummy_image=None, 
            debug_flag=False, 
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
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
        # self.use_gauss_input = use_gauss_input
        self.dummy_image = dummy_image
        self.convert_to_hu = convert_to_hu
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
            save_image(dummy_image, os.path.join(self.output_dir, "dummy_image_input.png"))
        
    def forward(self, t, xt, y=None):

        ## Given a single imput image get its actual flow
        # if self.dummy_image is not None:
        #     z = self.dummy_image.to(xt.device)
        #     while t.ndim < xt.ndim: t.unsqueeze_(-1)
        #     return (z - xt)/torch.clip(1-t, min=1e-2)

        if y:
            return self.model(t, xt, y)
        return self.model(t, xt)
    

    def training_step(self, batch, batch_idx):
        if self.paired_input:
            x0 = batch[:, 0:self.input_channels, :, :]
            x1 = batch[:, self.input_channels:, :, :]
        else:
            x1 = batch
            x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        vt = self(t, xt)
        loss = torch.mean((vt - ut) ** 2)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr/current", lr, on_step=True, logger=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
        
    
    def configure_optimizers(self):
        """ Configure the optimizer and learning rate scheduler """
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = total_steps // 10
        decay_steps = total_steps - warmup_steps

        # Only include the model we are training not the decoding model
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)  # start_factor must > 0
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        sched  = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    def generate_images(self, x_input: tensor, n_steps=80):
        """ 
        Generate images from the model given an input image x_input 
        x_input: tensor - input to generate from
        n_steps: int - number of steps
        """
        t_vec = torch.linspace(0, 1, n_steps+1, device=x_input.device)
        x_t = x_input
        for idx, t0 in enumerate(t_vec[:-1]):
            t1 = t_vec[idx+1]
            dt = t1 - t0
            v = self(t0, x_t)
            x_t = x_t + v * dt
        return x_t
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step will both evaluate the loss on unseen data and generate sample images to monitor the models progress
        """
        global_step = self.global_step
        rank = self.global_rank

        # First Compute the loss
        if self.paired_input:
            x0 = batch[:, 0:self.input_channels, :, :]
            x1 = batch[:, self.input_channels:, :, :]
        else:
            x1 = batch
            x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        vt = self(t, xt)
        val_loss = torch.mean((vt - ut) ** 2)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Now save the output of the model on a test image
        # Only want return the evaluation image on the first gpu
        if rank != 0:
            return val_loss
        
        # if self.use_gauss_input:
        #     x_gauss = torch.randn(self.num_val_images, batch.shape[1], batch.shape[2], batch.shape[3]).to(batch.device)
        # else:
        #     x_gauss = x1[:self.num_val_images]

        if self.paired_input:
            x_gauss = x0[:self.num_val_images]
        else:
            x_gauss = torch.randn(self.num_val_images, batch.shape[1], batch.shape[2], batch.shape[3]).to(batch.device)

        x_output = self.generate_images(x_gauss)

        images = x_output.view([-1, self.input_channels, self.image_size[0], self.image_size[1]]) 

        if self.decode_model is not None:
            with torch.no_grad():
                images = self.decode_model.decode(images)

        images = images.clip(-1, 1)
        for i, (input_img, output_img) in enumerate(zip(x_gauss, images)):

            save_path_in = f"{self.validation_path}/validation_step={global_step}_rank={rank}_sample_in.png"
            save_path_out = f"{self.validation_path}/validation_step={global_step}_rank={rank}_sample_out.png"
            save_path_target = f"{self.validation_path}/validation_step={global_step}_rank={rank}_sample_target.png"

            save_image(input_img, save_path_in, nrow=1, normalize=True, value_range=(-1, 1))
            if self.convert_to_hu:
                output_img_hu = reverse_normalize(output_img, clip_window=(-2000, 500))
                output_img = window_ct_hu_to_png(output_img_hu, center=-600, width=1500, bit_depth=8, return_float=True)
                save_image(output_img, save_path_out, nrow=1, normalize=True, value_range=(0, 255))
            else:
                save_image(output_img, save_path_out, nrow=1, normalize=True, value_range=(-1, 1))
            if self.paired_input:
                target_img = x1[i]
                if self.decode_model is not None:
                    with torch.no_grad():
                        target_img = self.decode_model.decode(target_img.unsqueeze(0))
                    target_img = target_img.squeeze(0)
                target_img = target_img.clip(-1, 1)
                if self.convert_to_hu:
                    target_img_hu = reverse_normalize(target_img, clip_window=(-2000, 500))
                    target_img = window_ct_hu_to_png(target_img_hu, center=-600, width=1500, bit_depth=8, return_float=True)
                    save_image(target_img, save_path_target, nrow=1, normalize=True, value_range=(0, 255))
                else:
                    save_image(target_img, save_path_target, nrow=1, normalize=True, value_range=(-1, 1))

        return val_loss
        
    
    def test_step(self, batch, batch_idx):
        if self.paired_input:
            x_gauss = batch[:, 0:self.input_channels, :, :]
            x_batch = batch[:, self.input_channels:, :, :]
        else:
            x_batch = batch
            x_gauss = torch.randn_like(x_batch)

        x_out = self.generate_images(x_gauss)

        images = x_out.view([-1, self.input_channels, self.image_size[0], self.image_size[1]]) 

        if self.decode_model is not None:
            with torch.no_grad():
                images = self.decode_model.decode(images)

        images = images.clip(-1, 1)
        for i, (input_img, output_img) in enumerate(zip(x_gauss, images)):
            rank = self.trainer.global_rank
            save_path_in = f"{self.output_dir}/final_test_batch={batch_idx}_rank={rank}_sample_{i}_sample_in.png"
            save_path_out = f"{self.output_dir}/final_test_batch={batch_idx}_rank={rank}_sample_{i}_sample_out.png"
            save_path_target = f"{self.output_dir}/final_test_batch={batch_idx}_rank={rank}_sample_{i}_sample_target.png"

            save_image(input_img, save_path_in, nrow=1,normalize=True, value_range=(-1, 1))
            if self.convert_to_hu:
                output_img_hu = reverse_normalize(output_img, clip_window=(-2000, 500))
                output_img = window_ct_hu_to_png(output_img_hu, center=-600, width=1500, bit_depth=8)
                save_image(output_img, save_path_out, nrow=1,normalize=True, value_range=(0, 255))
            else:
                save_image(output_img, save_path_out, nrow=1,normalize=True, value_range=(-1, 1))
            if self.paired_input:
                target_img = x_batch[i]
                if self.decode_model is not None:
                    with torch.no_grad():
                        target_img = self.decode_model.decode(target_img.unsqueeze(0))
                    target_img = target_img.squeeze(0)
                target_img = target_img.clip(-1, 1)
                if self.convert_to_hu:
                    target_img_hu = reverse_normalize(target_img, clip_window=(-2000, 500))
                    target_img = window_ct_hu_to_png(target_img_hu, center=-600, width=1500, bit_depth=8)
                    save_image(target_img, save_path_target, nrow=1,normalize=True, value_range=(0, 255))
                else:  
                    save_image(target_img, save_path_target, nrow=1,normalize=True, value_range=(-1, 1))



