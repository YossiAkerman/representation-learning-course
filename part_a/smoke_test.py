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
from datamodule import CIFAR100DataModule, TinyImageNetDataModule, balanced_train_val_split, verify_dataset
from evaluate import accuracy_from_scores, corrupt, corruption_curve, retrieval_indices, weighted_knn
from model import DenoisingAutoEncoder, EncoderClassifier, build_encoder
from search import experiment_settings, rank_trials, render_config, sample_trials
from splits import class_balanced_subset, load_or_create_subset, subset_counts


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

    again = TinyImageNetDataModule()
    again.setup()
    check(passed, "the Tiny-ImageNet split is reproducible", again.train_indices == datamodule.train_indices and again.val_indices == datamodule.val_indices)

    one = class_balanced_subset(datamodule.train_indices, datamodule.train_labels, 900, 200, config.SEED)
    one_again = class_balanced_subset(datamodule.train_indices, datamodule.train_labels, 900, 200, config.SEED)
    one_counts = subset_counts(datamodule.train_indices, datamodule.train_labels, one, 200)
    ten = class_balanced_subset(datamodule.train_indices, datamodule.train_labels, 9000, 200, config.SEED)
    ten_counts = subset_counts(datamodule.train_indices, datamodule.train_labels, ten, 200)
    full = class_balanced_subset(datamodule.train_indices, datamodule.train_labels, 90000, 200, config.SEED)
    stored = load_or_create_subset(
        root / "splits" / "tinyimagenet_fraction_1.json",
        datamodule.train_indices,
        datamodule.train_labels,
        900,
        200,
        config.SEED,
        "fraction_1",
    )
    check(passed, "1 percent subset is reproducible and nearly balanced", one == one_again == stored and len(one) == 900 and int((one_counts == 5).sum()) == 100 and int((one_counts == 4).sum()) == 100)
    check(passed, "10 percent subset has 45 images per class", len(ten) == 9000 and int(ten_counts.min()) == 45 and int(ten_counts.max()) == 45)
    check(passed, "100 percent subset is the full training pool", full == list(datamodule.train_indices))

    bank = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    knn_scores = weighted_knn(torch.tensor([[1.0, 0.0]]), bank, torch.tensor([0, 0, 1]), k=1, num_classes=2)
    knn_result = accuracy_from_scores(knn_scores, torch.tensor([0]))
    score_row = torch.tensor([[0.1, 0.2, 0.9, 0.05, 0.0, 0.4]])
    top5_hit = accuracy_from_scores(score_row, torch.tensor([5]))
    top5_miss = accuracy_from_scores(score_row, torch.tensor([4]))
    check(passed, "weighted k-NN returns the nearest class", int(knn_result["predictions"][0]) == 0 and knn_result["top1"] == 1.0)
    check(passed, "top-5 counts a class in the shortlist and top-1 does not", top5_hit["top1"] == 0.0 and top5_hit["top5"] == 1.0 and top5_miss["top5"] == 0.0)
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    normalized_bank = torch.nn.functional.normalize(bank, dim=-1)
    check(passed, "nearest neighbor retrieval returns the closest row", int(retrieval_indices(query, normalized_bank, k=1)[0, 0]) == 0)

    encoder_path = root / "checkpoints" / "smoke_encoder.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_encoder(8).state_dict(), encoder_path)
    images_small = torch.rand(2, 3, 64, 64)
    probe_labels = torch.tensor([0, 1])
    probe = EncoderClassifier(0.001, 10, 8, encoder_path, freeze_encoder=True)
    probe.train()
    probe_loss = torch.nn.functional.cross_entropy(probe(images_small), probe_labels)
    probe_loss.backward()
    frozen_ok = (not probe.encoder.training) and all(parameter.grad is None for parameter in probe.encoder.parameters()) and probe.head.weight.grad is not None
    check(passed, "a linear probe keeps the encoder frozen", frozen_ok and torch.isfinite(probe.head.weight.grad).all())

    finetune = EncoderClassifier(0.0001, 10, 8, encoder_path, freeze_encoder=False)
    finetune.train()
    finetune_loss = torch.nn.functional.cross_entropy(finetune(images_small), probe_labels)
    finetune_loss.backward()
    check(
        passed,
        "fine-tuning trains the encoder",
        finetune.encoder.training and all(parameter.requires_grad for parameter in finetune.encoder.parameters()) and finetune.encoder[0].weight.grad is not None and torch.isfinite(finetune.encoder[0].weight.grad).all(),
    )
    encoder_path.unlink()

    synthetic_labels = [class_id for class_id in range(100) for _ in range(500)]
    cifar_train, cifar_val = balanced_train_val_split(synthetic_labels, 50, 100, config.SEED)
    cifar_pool_labels = [synthetic_labels[index] for index in cifar_train]
    cifar_one = class_balanced_subset(cifar_train, cifar_pool_labels, 450, 100, config.SEED)
    cifar_counts = subset_counts(cifar_train, cifar_pool_labels, cifar_one, 100)
    check(
        passed,
        "synthetic CIFAR-100 holdout is 45000, 5000, and disjoint",
        len(cifar_train) == 45000 and len(cifar_val) == 5000 and not (set(cifar_train) & set(cifar_val)),
    )
    check(
        passed,
        "synthetic CIFAR-100 1 percent subset totals 450",
        len(cifar_one) == 450 and int((cifar_counts == 5).sum()) == 50 and int((cifar_counts == 4).sum()) == 50,
    )

    cifar = CIFAR100DataModule(batch_size=32, seed=config.SEED)
    cifar.prepare_data()
    cifar.setup()
    cifar_batch, _ = next(iter(cifar.val_dataloader()))
    check(passed, "CIFAR-100 splits are 45000, 5000, and 10000", len(cifar.train_ds) == 45000 and len(cifar.val_ds) == 5000 and len(cifar.test_ds) == 10000)
    check(passed, "CIFAR-100 test data is a separate dataset from training", cifar.train_source is not cifar.test_source)
    check(passed, "CIFAR-100 batch is RGB 64x64 in [0, 1]", tuple(cifar_batch.shape[1:]) == (3, 64, 64) and float(cifar_batch.min()) >= 0.0 and float(cifar_batch.max()) <= 1.0)

    trials = sample_trials()
    check(passed, "the proxy search draws 12 unique grid points", trials == sample_trials() and len(trials) == 12 and len(set(trials)) == 12)
    tied = rank_trials([
        {"knn_top1": 0.4, "val_mse": 0.2, "latent_dim": 256},
        {"knn_top1": 0.4, "val_mse": 0.2, "latent_dim": 64},
        {"knn_top1": 0.5, "val_mse": 0.9, "latent_dim": 128},
    ])
    check(passed, "model selection uses k-NN, then MSE, then latent size", tied[0]["knn_top1"] == 0.5 and rank_trials(tied[1:])[0]["latent_dim"] == 64)
    first = render_config(experiment_settings(EXPERIMENT_NAME="a2_00", CHECKPOINT_DIR="checkpoints/a2_00", ENCODER_PATH="checkpoints/a2_00/encoder.pt"))
    second = render_config(experiment_settings(EXPERIMENT_NAME="a2_01", CHECKPOINT_DIR="checkpoints/a2_01", ENCODER_PATH="checkpoints/a2_01/encoder.pt"))
    check(passed, "experiment configs keep separate checkpoint directories", "checkpoints/a2_00" in first and "checkpoints/a2_01" in second and "checkpoints/a2_00" not in second)

    clean_images = torch.rand(4, 3, 8, 8)
    shared_noise = torch.randn_like(clean_images)
    unchanged = corrupt(clean_images, 0, shared_noise)
    corrupted = corrupt(clean_images, 0.2, shared_noise)

    class _MeanClassifier(torch.nn.Module):
        def forward(self, batch):
            mean = batch.mean(dim=(1, 2, 3))
            return torch.stack((1 - mean, mean), dim=1)

    curve = corruption_curve(_MeanClassifier(), clean_images, torch.zeros(4, dtype=torch.long), (0.0, 0.2), seed=config.SEED)
    curve_again = corruption_curve(_MeanClassifier(), clean_images, torch.zeros(4, dtype=torch.long), (0.0, 0.2), seed=config.SEED)
    check(passed, "sigma 0 keeps the image and paired noise is deterministic", torch.equal(unchanged, clean_images.clamp(0, 1)) and torch.equal(corrupted, corrupt(clean_images, 0.2, shared_noise)) and curve == curve_again)

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
