import argparse
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
from datamodule import TinyImageNetDataModule
from model import DenoisingAutoEncoder

torch.set_float32_matmul_precision("medium")  # to make lightning happy


def build_model():
    return DenoisingAutoEncoder(
        learning_rate=config.LEARNING_RATE,
        latent_dim=config.LATENT_DIM,
        noise_sigma=config.NOISE_SIGMA,
        noise_seed=config.SEED,
    )


def build_datamodule():
    return TinyImageNetDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
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

    if args.test:
        ckpt_path = Path(config.CHECKPOINT_DIR) / "last.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train before testing.")
        logger = TensorBoardLogger(args.log_dir, name=time.strftime("%Y-%m-%d_%H-%M-%S"), version="test")
        trainer = _make_trainer(args, logger, max_epochs=1, checkpointing=False)
        trainer.test(build_model(), datamodule=build_datamodule(), ckpt_path=str(ckpt_path))
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
        f"Experiment {config.EXPERIMENT_NAME}: denoising autoencoder on Tiny-ImageNet "
        f"for {max_epochs} epochs, batch size {config.BATCH_SIZE}, "
        f"latent dim {config.LATENT_DIM}, noise sigma {config.NOISE_SIGMA}, lr {config.LEARNING_RATE}"
    )
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    # The lecture pipeline hands the next stage an encoder file. A smoke run
    # uses another checkpoint directory, so it does not overwrite this file.
    if args.ckpt_dir == config.CHECKPOINT_DIR:
        torch.save(model.encoder.state_dict(), config.ENCODER_PATH)
        print(f"Saved encoder to {config.ENCODER_PATH}")


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
