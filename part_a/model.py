import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics


def build_encoder(latent_dim):
    """Lecture-4 convolutional encoder for a 64x64 RGB image.

    Two stride-2 convolutions halve the map twice (64 -> 32 -> 16). A linear
    layer then compresses 64*16*16 numbers into h. The same stack is the
    denoising autoencoder, the vanilla autoencoder, and every baseline.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, 64, 64] -> [B, 32, 32, 32]
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32] -> [B, 64, 16, 16]
        nn.ReLU(),
        nn.Flatten(),  # [B, 64, 16, 16] -> [B, 16384]
        nn.Linear(64 * 16 * 16, latent_dim),  # [B, 16384] -> [B, latent_dim]
    )


def build_decoder(latent_dim):
    """Mirror of build_encoder. Sigmoid keeps pixels in (0, 1)."""
    return nn.Sequential(
        nn.Linear(latent_dim, 64 * 16 * 16),  # [B, latent_dim] -> [B, 16384]
        nn.ReLU(),
        nn.Unflatten(1, (64, 16, 16)),  # [B, 16384] -> [B, 64, 16, 16]
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 32, 32, 32]
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 3, 64, 64]
        nn.Sigmoid(),
    )


def _accuracy_metrics(num_classes):
    """Separate train, validation, and test meters so their state does not mix."""
    metrics = {}
    for stage in ("train", "val", "test"):
        metrics[f"{stage}_top1"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        metrics[f"{stage}_top5"] = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
    return metrics


def _classification_step(module, batch, stage):
    x, y = batch
    logits = module(x)
    loss = module.loss_fn(logits, y)
    top1 = getattr(module, f"{stage}_top1")(logits, y)
    top5 = getattr(module, f"{stage}_top5")(logits, y)
    module.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
    module.log(f"{stage}_accuracy", top1, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
    module.log(f"{stage}_top5", top5, on_step=False, on_epoch=True, prog_bar=True)
    return loss


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

        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(latent_dim)
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


class EncoderClassifier(pl.LightningModule):
    """Pretrained encoder plus a new linear head.

    freeze_encoder=True trains only the head. False trains the encoder and the
    head. The encoder file is the representation being evaluated; this module
    does not load a decoder.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, encoder_path, freeze_encoder):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.freeze_encoder = freeze_encoder

        self.encoder = build_encoder(latent_dim)
        state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict)
        self.head = nn.Linear(latent_dim, num_classes)

        if self.freeze_encoder:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        for name, metric in _accuracy_metrics(num_classes).items():
            setattr(self, name, metric)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, x):
        """Frozen or fine-tuned representation h."""
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encode(x))

    def training_step(self, batch, batch_idx):
        return _classification_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return _classification_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return _classification_step(self, batch, "test")

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return optim.Adam(parameters, lr=self.lr)


class SupervisedEncoder(pl.LightningModule):
    """Same encoder as the autoencoder, trained from scratch with labels.

    No autoencoder checkpoint is loaded. This is the supervised reference
    for label efficiency.
    """

    def __init__(self, learning_rate, num_classes, latent_dim):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.encoder = build_encoder(latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        for name, metric in _accuracy_metrics(num_classes).items():
            setattr(self, name, metric)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encode(x))

    def training_step(self, batch, batch_idx):
        return _classification_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return _classification_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return _classification_step(self, batch, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
