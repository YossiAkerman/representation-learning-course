"""Report figures for the denoising autoencoder.

Run this after training. The script reads a checkpoint and a TensorBoard log.
It does not train and it does not invent missing numbers.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import config
from model import DenoisingAutoEncoder
from train import build_datamodule


def latest_logdir(root):
    """Most recent tb_logs/<timestamp>/denoising_ae directory."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"No TensorBoard directory at {root}. Train before plotting.")
    candidates = [path / "denoising_ae" for path in root.iterdir() if (path / "denoising_ae").is_dir()]
    if not candidates:
        raise SystemExit(f"No denoising_ae run under {root}. Train before plotting.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _scalar_values(logdir, names):
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    for name in names:
        if name in available:
            return [event.value for event in accumulator.Scalars(name)]
    raise SystemExit(
        f"None of {names} are logged in {logdir}. Scalars present: {available}. "
        "Refusing to draw a curve without recorded losses."
    )


def save_reconstructions(ckpt, out_path):
    """Eight validation images: original, the noisy input, and the reconstruction."""
    model = DenoisingAutoEncoder.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    datamodule = build_datamodule()
    datamodule.setup()
    images, _ = next(iter(datamodule.val_dataloader()))
    images = images[:8]
    if images.shape[0] < 8:
        raise SystemExit(f"Need 8 validation images, found {images.shape[0]}.")

    # Same seed the validation epoch uses, so this row is the corruption the loss sees.
    generator = torch.Generator().manual_seed(config.SEED)
    with torch.no_grad():
        noisy = model.add_noise(images, generator=generator)
        reconstructed = model(noisy)

    rows = (
        ("Original", images),
        ("Noisy", noisy),
        ("Reconstructed", reconstructed),
    )
    fig, axes = plt.subplots(3, 8, figsize=(12, 5.2))
    for row, (title, batch) in enumerate(rows):
        for col in range(8):
            ax = axes[row, col]
            image = batch[col].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=11)
    fig.suptitle("Validation reconstructions", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(logdir, out_path):
    """Epoch training and validation reconstruction loss from TensorBoard."""
    train_loss = _scalar_values(logdir, ("train_loss_epoch",))
    val_loss = _scalar_values(logdir, ("val_loss", "val_loss_epoch"))
    if not train_loss or not val_loss:
        raise SystemExit(f"No epoch losses in {logdir}. Refusing to draw an empty figure.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(train_loss) + 1), train_loss, marker="o", label="Training")
    ax.plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE)")
    ax.set_title("Denoising autoencoder reconstruction loss")
    ax.set_xticks(range(1, max(len(train_loss), len(val_loss)) + 1))
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Save the Part A1 report figures from a finished run.")
    parser.add_argument("--ckpt", type=str, default=str(Path(config.CHECKPOINT_DIR) / "last.ckpt"))
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard version directory. Default: latest run.")
    parser.add_argument("--out-dir", type=str, default=config.FIGURES_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint {ckpt}. Train before saving figures.")
    logdir = Path(args.logdir) if args.logdir else latest_logdir(config.LOG_DIR)
    if not logdir.is_dir():
        raise SystemExit(f"Missing TensorBoard directory {logdir}.")

    out_dir = Path(args.out_dir)
    save_reconstructions(ckpt, out_dir / "a1_reconstructions.png")
    save_loss_curve(logdir, out_dir / "a1_training_loss.png")
    print(f"Saved {out_dir / 'a1_reconstructions.png'}")
    print(f"Saved {out_dir / 'a1_training_loss.png'}")


if __name__ == "__main__":
    main()
