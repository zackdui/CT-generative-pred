from diffusers.models import AutoencoderKL
import torch
from torch import Tensor
import lightning.pytorch as pl


class AutoEncoder_Lightning(pl.LightningModule):
    """
    AutoEncoder Lightning Module wrapping the AutoencoderKL from diffusers.
    The encoder will just return the mean rather than sampling from the latent distribution.
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
        # Other model options "stabilityai/sd-vae-ft-mse", "stabilityai/sdxl-vae"
        # model_params = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
        # print(f"Model Parameters: {model_params/1e6:.2f} Million")
        
    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """
        Encode will return the mean latent representation of x.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # return self.autoencoder.encode(x, return_dict=True).latent_dist.sample().mul_(0.18215)
        return self.autoencoder.encode(x, return_dict=True).latent_dist.mode().mul_(0.18215)
    
    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.autoencoder.decode(z / 0.18215, return_dict=True).sample
    
    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encode(x)
        x_recon = self.decode(z)
        loss = torch.nn.functional.mse_loss(x_recon, x)
        self.log("train_loss", loss)
        return loss
 
if __name__ == "__main__":
    model = AutoEncoder_Lightning()

    