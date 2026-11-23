import argparse
import json
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
from datamodule import CIFAR100DataModule, TinyImageNetDataModule
from model import (
    DenoisingAutoEncoder,
    EncoderClassifier,
    SupervisedEncoder,
    build_encoder,
)

torch.set_float32_matmul_precision("medium")  # to make lightning happy


def build_model():
    """Build the Lightning module named by config.MODEL."""
    model_name = getattr(config, "MODEL", "dae")
    if model_name == "dae":
        return DenoisingAutoEncoder(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            noise_sigma=config.NOISE_SIGMA,
            noise_seed=config.SEED,
        )
    if model_name == "supervised":
        return SupervisedEncoder(
            learning_rate=config.LEARNING_RATE,
            num_classes=config.NUM_CLASSES,
            latent_dim=config.LATENT_DIM,
        )
    if model_name in ("linear_probe", "finetune"):
        source = getattr(config, "PRETRAINED_ENCODER", None) or config.ENCODER_PATH
        if not Path(source).is_file():
            raise FileNotFoundError(f"No encoder at {source}. Train that encoder before this run.")
        learning_rate = config.LEARNING_RATE
        if model_name == "finetune":
            learning_rate = getattr(config, "FINETUNE_LEARNING_RATE", 0.0001)
        return EncoderClassifier(
            learning_rate=learning_rate,
            num_classes=config.NUM_CLASSES,
            latent_dim=config.LATENT_DIM,
            encoder_path=source,
            freeze_encoder=(model_name == "linear_probe"),
        )
    if model_name == "random":
        return None
    raise ValueError(f"Unknown MODEL {model_name!r}.")


def build_datamodule():
    """Tiny-ImageNet or CIFAR-100, using the active experiment's subset settings."""
    dataset = getattr(config, "DATASET", "tinyimagenet")
    settings = dict(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
        label_fraction=getattr(config, "LABEL_FRACTION", 1.0),
        proxy_per_class=getattr(config, "PROXY_PER_CLASS", None),
        splits_dir=getattr(config, "SPLITS_DIR", "splits"),
    )
    if dataset == "cifar100":
        return CIFAR100DataModule(
            **settings,
            val_per_class=getattr(config, "VAL_PER_CLASS", 50),
            num_classes=config.NUM_CLASSES,
        )
    return TinyImageNetDataModule(
        **settings,
        val_per_class=config.VAL_PER_CLASS,
        num_classes=config.NUM_CLASSES,
        dataset_name=config.DATASET_NAME,
    )


def _limit_batches(value):
    """Lightning treats 1 as one batch and 1.0 as the whole loader."""
    if "." in value:
        return float(value)
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Tiny-ImageNet denoising autoencoder.")
    parser.add_argument(
        "--config",
        default="a1",
        help="Experiment file in configs/, without the .py suffix. Default: a1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/last.ckpt.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Score the reserved official test split with checkpoints/last.ckpt. Does not train.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=_limit_batches, default=1.0)
    parser.add_argument("--limit-val-batches", type=_limit_batches, default=1.0)
    parser.add_argument("--ckpt-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=str, default=config.LOG_DIR)
    return parser.parse_args()


def run(args):
    """Train, or score the held-out test split. Settings come from the loaded experiment."""
    pl.seed_everything(config.SEED, workers=True)
    model_name = getattr(config, "MODEL", "dae")

    if model_name == "random":
        _save_random_encoder()
        return

    if args.test:
        ckpt_path = Path(config.CHECKPOINT_DIR) / "last.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train before testing.")
        logger = TensorBoardLogger(args.log_dir, name=time.strftime("%Y-%m-%d_%H-%M-%S"), version="test")
        trainer = _make_trainer(args, logger, max_epochs=1, checkpointing=False)
        results = trainer.test(build_model(), datamodule=build_datamodule(), ckpt_path=str(ckpt_path))
        _write_test_summary(results)
        return

    ckpt_path = None
    if args.resume:
        ckpt_path = str(Path(args.ckpt_dir) / "last.ckpt")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}.")

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger(
        args.log_dir, name=run_id, version=config.EXPERIMENT_NAME, default_hp_metric=False
    )
    model = build_model()
    dm = build_datamodule()
    max_epochs = config.NUM_EPOCHS if args.max_epochs is None else args.max_epochs
    trainer = _make_trainer(args, logger, max_epochs)

    print(model)
    print(
        f"Experiment {config.EXPERIMENT_NAME}: {model_name} on {getattr(config, 'DATASET', 'tinyimagenet')} "
        f"for {max_epochs} epochs, batch size {config.BATCH_SIZE}, "
        f"latent dim {config.LATENT_DIM}, noise sigma {config.NOISE_SIGMA}, lr {config.LEARNING_RATE}"
    )
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    # A smoke run passes another checkpoint directory, so it does not replace
    # this experiment's encoder file.
    if args.ckpt_dir == config.CHECKPOINT_DIR and model_name != "linear_probe":
        destination = Path(config.ENCODER_PATH)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), destination)
        print(f"Saved encoder to {destination}")
    if args.ckpt_dir == config.CHECKPOINT_DIR:
        _write_fit_summary(trainer)


def _save_random_encoder():
    """Write the seeded encoder without training it."""
    encoder = build_encoder(config.LATENT_DIM)
    destination = Path(config.ENCODER_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), destination)
    print(f"Saved random encoder to {destination}")


def _write_fit_summary(trainer):
    """Record the epoch metrics from this run. Smoke runs skip this file."""
    metrics = {}
    for key, value in trainer.callback_metrics.items():
        if hasattr(value, "item"):
            metrics[key] = float(value.item())
    payload = {
        "experiment": config.EXPERIMENT_NAME,
        "model": getattr(config, "MODEL", "dae"),
        "dataset": getattr(config, "DATASET", "tinyimagenet"),
        "latent_dim": config.LATENT_DIM,
        "noise_sigma": config.NOISE_SIGMA,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "metrics": metrics,
    }
    destination = Path(getattr(config, "RESULTS_DIR", "results")) / f"{config.EXPERIMENT_NAME}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(f"Saved fit summary to {destination}")


def _write_test_summary(results):
    metrics = {}
    if results:
        for key, value in results[0].items():
            metrics[key] = float(value)
    payload = {
        "experiment": config.EXPERIMENT_NAME,
        "model": getattr(config, "MODEL", "dae"),
        "dataset": getattr(config, "DATASET", "tinyimagenet"),
        "split": "test",
        "metrics": metrics,
    }
    destination = Path(getattr(config, "RESULTS_DIR", "results")) / f"{config.EXPERIMENT_NAME}_test.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(f"Saved test summary to {destination}")


def _make_trainer(args, logger, max_epochs, checkpointing=True):
    # save_top_k=0 keeps only last.ckpt, which is what --resume loads.
    callbacks = []
    if checkpointing:
        callbacks.append(ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=0))
    return pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=max_epochs,
        precision=config.PRECISION,
        log_every_n_steps=10,  # default is 50: short runs would get no step points
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
        enable_checkpointing=checkpointing,
    )


def main():
    # Load the experiment before parse_args reads checkpoint and log defaults from it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="a1")
    known, _ = pre.parse_known_args()
    config.load(known.config)
    run(parse_args())


if __name__ == "__main__":
    main()
