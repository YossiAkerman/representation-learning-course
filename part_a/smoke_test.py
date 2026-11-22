"""Checks for the denoising-autoencoder setup.

Runs the real data split, one forward and backward pass, and one batch through
train.py. Does not train for 10 epochs and does not write the report figures.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import torch

import config
from datamodule import TinyImageNetDataModule, verify_dataset
from model import DenoisingAutoEncoder


def check(passed, name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS  {name}", flush=True)
    passed.append(name)


def main():
    root = Path(__file__).resolve().parent
    passed = []

    datamodule = TinyImageNetDataModule()
    report = verify_dataset(datamodule)
    covered = set(datamodule.train_indices) | set(datamodule.val_indices)

    check(passed, "train size is 90000", report["train_size"] == 90000)
    check(passed, "validation size is 10000", report["val_size"] == 10000)
    check(passed, "test size is 10000", report["test_size"] == 10000)
    check(passed, "train and validation indices do not overlap", report["overlap"] == 0)
    check(passed, "train and validation partition the original training split", len(covered) == 100000)
    check(passed, "validation holds out 50 images per class", report["val_count_min"] == 50 and report["val_count_max"] == 50)
    check(passed, "training batch is RGB 64x64", report["sample_shape"][1:] == (3, 64, 64))
    check(passed, "validation batch is RGB 64x64", report["val_shape"][1:] == (3, 64, 64))
    check(passed, "test batch is RGB 64x64", report["test_shape"][1:] == (3, 64, 64))
    check(passed, "pixels are in [0, 1]", 0.0 <= report["pixel_min"] and report["pixel_max"] <= 1.0)

    images, _ = next(iter(datamodule.train_dataloader()))
    model = DenoisingAutoEncoder(
        learning_rate=config.LEARNING_RATE,
        latent_dim=config.LATENT_DIM,
        noise_sigma=config.NOISE_SIGMA,
        noise_seed=config.SEED,
    )
    model.train()
    representation = model.encode(images)
    noisy = model.add_noise(images)
    reconstruction = model.forward(noisy)
    loss = torch.nn.functional.mse_loss(reconstruction, images)

    check(passed, "encoder output is [B, 128]", tuple(representation.shape) == (images.shape[0], config.LATENT_DIM))
    check(passed, "decoder output is [B, 3, 64, 64]", tuple(reconstruction.shape) == tuple(images.shape))
    check(passed, "reconstruction loss is finite", bool(torch.isfinite(loss)))
    loss.backward()
    grads_ok = any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.requires_grad
    )
    check(passed, "backward pass produces finite gradients", grads_ok)

    ckpt_dir = root / "checkpoints" / "smoke"
    log_dir = root / "tb_logs" / "smoke"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "a1",
            "--max-epochs",
            "1",
            "--limit-train-batches",
            "1",
            "--limit-val-batches",
            "1",
            "--ckpt-dir",
            str(ckpt_dir),
            "--log-dir",
            str(log_dir),
        ],
        cwd=root,
        check=True,
    )
    ckpt = ckpt_dir / "last.ckpt"
    check(passed, "train.py writes a checkpoint", ckpt.is_file() and ckpt.stat().st_size > 0)
    restored = DenoisingAutoEncoder.load_from_checkpoint(ckpt, map_location="cpu")
    restored_h = restored.encode(images[:2])
    check(passed, "checkpoint reloads and encodes", tuple(restored_h.shape) == (2, config.LATENT_DIM))

    shutil.rmtree(ckpt_dir)
    shutil.rmtree(log_dir)
    print(f"\n{len(passed)} checks passed.")


if __name__ == "__main__":
    main()
