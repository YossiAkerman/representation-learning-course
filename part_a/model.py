import torch
from torch import nn, optim
import pytorch_lightning as pl


class DenoisingAutoEncoder(pl.LightningModule):
    """Convolutional denoising autoencoder for 64x64 RGB images.

    The encoder is the lecture-4 autoencoder, adapted from one grayscale
    channel and 28x28 to three channels and 64x64. Two stride-2 convolutions
    halve the map twice (64 -> 32 -> 16). A 3x3 kernel with padding 1 does
    that exactly, so no pooling layer is needed. Channels go 3 -> 32 -> 64.
    A linear layer then compresses 64*16*16 numbers into h.

    The decoder is the mirror image. output_padding=1 makes the transposed
    convolutions land on 32 and 64 rather than 31 and 63. Sigmoid keeps every
    pixel in [0, 1], which is the range to_tensor produces.

    Training adds Gaussian noise to the input and asks the decoder to
    reconstruct the clean image. Validation reseeds its own generator at
    every epoch, and the validation loader is not shuffled, so each epoch
    is scored on the same noisy inputs.
    """

    def __init__(self, learning_rate, latent_dim, noise_sigma, noise_seed):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, 64, 64] -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32] -> [B, 64, 16, 16]
            nn.ReLU(),
            nn.Flatten(),  # [B, 64, 16, 16] -> [B, 16384]
            nn.Linear(64 * 16 * 16, latent_dim),  # [B, 16384] -> [B, latent_dim]
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),  # [B, latent_dim] -> [B, 16384]
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),  # [B, 16384] -> [B, 64, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 3, 64, 64]
            nn.Sigmoid(),  # values in (0, 1)
        )
        self.loss_fn = nn.MSELoss()

    def encode(self, x):
        """Representation h, for later evaluation. x is [B, 3, 64, 64]."""
        return self.encoder(x)

    def forward(self, x):
        """Reconstruct a (usually noisy) image. Output is [B, 3, 64, 64]."""
        return self.decoder(self.encode(x))

    def add_noise(self, x, generator=None):
        """Gaussian noise with the configured sigma, clamped back to [0, 1].

        generator=None draws from the process RNG, which training uses.
        A seeded generator makes validation noise the same on every epoch.
        """
        if generator is None:
            noise = torch.randn_like(x)
        else:
            # Generator samples are produced on CPU, then moved to the batch device.
            noise = torch.randn(x.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(device=x.device, dtype=x.dtype)
        return (x + self.noise_sigma * noise).clamp(0, 1)

    def on_validation_epoch_start(self):
        self._reset_eval_noise()

    def on_test_epoch_start(self):
        self._reset_eval_noise()

    def _reset_eval_noise(self):
        self._eval_noise_generator = torch.Generator()
        self._eval_noise_generator.manual_seed(self.noise_seed)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # label is unused
        recon = self.forward(self.add_noise(x))
        loss = self.loss_fn(recon, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._eval_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _eval_loss(self, batch):
        x, _ = batch
        recon = self.forward(self.add_noise(x, generator=self._eval_noise_generator))
        return self.loss_fn(recon, x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
