"""Proxy search for the denoising autoencoder, and the configs that follow it.

Running this file trains the 12 proxy models. Importing it does not.
"""

import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import torch

SIGMAS = (0.05, 0.10, 0.20)
LATENT_DIMS = (64, 128, 256)
LEARNING_RATES = (0.0001, 0.0003, 0.001)

_BASE = {
    "EXPERIMENT_NAME": "a1",
    "MODEL": "dae",
    "DATASET": "tinyimagenet",
    "DATASET_NAME": "zh-plus/tiny-imagenet",
    "LATENT_DIM": 128,
    "NOISE_SIGMA": 0.10,
    "LEARNING_RATE": 0.001,
    "FINETUNE_LEARNING_RATE": 0.0001,
    "BATCH_SIZE": 128,
    "NUM_EPOCHS": 10,
    "SEED": 42,
    "NUM_CLASSES": 200,
    "VAL_PER_CLASS": 50,
    "NUM_WORKERS": 0,
    "LABEL_FRACTION": 1.0,
    "PROXY_PER_CLASS": None,
    "DATA_DIR": "dataset/",
    "SPLITS_DIR": "splits",
    "ACCELERATOR": "auto",
    "DEVICES": 1,
    "PRECISION": "16-mixed",
    "CHECKPOINT_DIR": "checkpoints",
    "ENCODER_PATH": "tiny_imagenet_encoder.pt",
    "PRETRAINED_ENCODER": None,
    "LOG_DIR": "tb_logs",
    "RESULTS_DIR": "results",
    "FIGURES_DIR": "figures",
    "KNN_K": 20,
}


def sample_trials(seed=42, count=12):
    """Twelve reproducible draws from the 27-point grid."""
    grid = list(itertools.product(SIGMAS, LATENT_DIMS, LEARNING_RATES))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(grid), generator=generator).tolist()
    return [grid[index] for index in order[:count]]


def rank_trials(rows):
    """Best validation k-NN, then lower reconstruction MSE, then the smaller latent size."""
    return sorted(rows, key=lambda row: (-row["knn_top1"], row["val_mse"], row["latent_dim"]))


def render_config(values):
    """One assignment per line, in a stable order."""
    lines = ["# Generated experiment configuration.", ""]
    for key in _BASE:
        lines.append(f"{key} = {values[key]!r}")
    return "\n".join(lines) + "\n"


def experiment_settings(**overrides):
    values = dict(_BASE)
    values.update(overrides)
    name = values["EXPERIMENT_NAME"]
    values.setdefault("CHECKPOINT_DIR", f"checkpoints/{name}")
    return values


def write_config(path, values):
    Path(path).write_text(render_config(values))


def write_selected_configs(selected, directory):
    """Write the final DAE, the vanilla counterpart, and every downstream run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    latent = selected["latent_dim"]
    learning_rate = selected["learning_rate"]
    sigma = selected["noise_sigma"]

    def named(name, **overrides):
        values = {
            "EXPERIMENT_NAME": name,
            "LATENT_DIM": latent,
            "LEARNING_RATE": learning_rate,
            "NOISE_SIGMA": sigma,
            "CHECKPOINT_DIR": f"checkpoints/{name}",
            "ENCODER_PATH": f"checkpoints/{name}/encoder.pt",
        }
        values.update(overrides)
        write_config(directory / f"{name}.py", experiment_settings(**values))
        return values

    named("dae_selected", MODEL="dae", NUM_EPOCHS=10, PROXY_PER_CLASS=None)
    named("vanilla", MODEL="dae", NOISE_SIGMA=0.0, NUM_EPOCHS=10, PROXY_PER_CLASS=None)
    named("random", MODEL="random", NOISE_SIGMA=0.0, NUM_EPOCHS=0, PROXY_PER_CLASS=None)

    sources = {
        "dae": "checkpoints/dae_selected/encoder.pt",
        "vanilla": "checkpoints/vanilla/encoder.pt",
        "random": "checkpoints/random/encoder.pt",
    }
    for dataset, prefix, num_classes in (
        ("tinyimagenet", "", 200),
        ("cifar100", "cifar_", 100),
    ):
        for percent in (1, 10, 100):
            named(
                f"{prefix}supervised_{percent}",
                MODEL="supervised",
                DATASET=dataset,
                NUM_CLASSES=num_classes,
                LABEL_FRACTION=percent / 100,
                LEARNING_RATE=0.001,
                NOISE_SIGMA=0.0,
            )
            for model_name in ("dae", "vanilla", "random"):
                named(
                    f"{prefix}probe_{model_name}_{percent}",
                    MODEL="linear_probe",
                    DATASET=dataset,
                    NUM_CLASSES=num_classes,
                    LABEL_FRACTION=percent / 100,
                    LEARNING_RATE=0.001,
                    NOISE_SIGMA=0.0,
                    PRETRAINED_ENCODER=sources[model_name],
                )
            for model_name in ("dae", "vanilla"):
                named(
                    f"{prefix}finetune_{model_name}_{percent}",
                    MODEL="finetune",
                    DATASET=dataset,
                    NUM_CLASSES=num_classes,
                    LABEL_FRACTION=percent / 100,
                    LEARNING_RATE=0.0001,
                    NOISE_SIGMA=0.0,
                    PRETRAINED_ENCODER=sources[model_name],
                )
            if percent < 100:
                named(
                    f"{prefix}finetune_random_{percent}",
                    MODEL="finetune",
                    DATASET=dataset,
                    NUM_CLASSES=num_classes,
                    LABEL_FRACTION=percent / 100,
                    LEARNING_RATE=0.0001,
                    NOISE_SIGMA=0.0,
                    PRETRAINED_ENCODER=sources["random"],
                )


def _write_trial_csv(path, rows):
    fieldnames = [
        "experiment",
        "noise_sigma",
        "latent_dim",
        "learning_rate",
        "val_mse",
        "knn_top1_percent",
        "knn_top5_percent",
        "selected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Train 12 proxy autoencoders and write the selected configuration."""
    root = Path(__file__).resolve().parent
    rows = []
    for index, (sigma, latent, learning_rate) in enumerate(sample_trials()):
        name = f"a2_{index:02d}"
        values = experiment_settings(
            EXPERIMENT_NAME=name,
            NOISE_SIGMA=sigma,
            LATENT_DIM=latent,
            LEARNING_RATE=learning_rate,
            NUM_EPOCHS=2,
            PROXY_PER_CLASS=20,
            CHECKPOINT_DIR=f"checkpoints/{name}",
            ENCODER_PATH=f"checkpoints/{name}/encoder.pt",
        )
        write_config(root / "configs" / f"{name}.py", values)
        subprocess.run([sys.executable, "train.py", "--config", name], cwd=root, check=True)
        subprocess.run(
            [sys.executable, "evaluate.py", "--config", name, "--protocol", "knn"],
            cwd=root,
            check=True,
        )
        fit = json.loads((root / "results" / f"{name}.json").read_text())
        knn = json.loads((root / "results" / f"{name}_knn_val.json").read_text())
        if "val_loss" not in fit["metrics"]:
            raise RuntimeError(f"{name} did not record a validation loss.")
        rows.append({
            "experiment": name,
            "noise_sigma": sigma,
            "latent_dim": latent,
            "learning_rate": learning_rate,
            "val_mse": fit["metrics"]["val_loss"],
            "knn_top1": knn["top1"],
            "knn_top5": knn["top5"],
        })

    ranked = rank_trials(rows)
    table = []
    for position, row in enumerate(ranked):
        table.append({
            "experiment": row["experiment"],
            "noise_sigma": row["noise_sigma"],
            "latent_dim": row["latent_dim"],
            "learning_rate": row["learning_rate"],
            "val_mse": row["val_mse"],
            "knn_top1_percent": 100 * row["knn_top1"],
            "knn_top5_percent": 100 * row["knn_top5"],
            "selected": int(position == 0),
        })
    _write_trial_csv(root / "results" / "a2_trials.csv", table)
    write_selected_configs(ranked[0], root / "configs")
    print(f"Selected {ranked[0]['experiment']} and wrote configs/dae_selected.py")


if __name__ == "__main__":
    main()
