"""Report figures for the denoising autoencoder.

Run this after training. The script reads a checkpoint and a TensorBoard log.
It does not train and it does not invent missing numbers.
"""

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import config
from model import DenoisingAutoEncoder
from train import build_datamodule


def latest_logdir(root):
    """Most recent tb_logs/<timestamp>/<experiment> directory."""
    root = Path(root)
    name = config.EXPERIMENT_NAME
    if not root.is_dir():
        raise SystemExit(f"No TensorBoard directory at {root}. Train before plotting.")
    candidates = [path / name for path in root.iterdir() if (path / name).is_dir()]
    if not candidates:
        raise SystemExit(f"No {name} run under {root}. Train before plotting.")
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
    parser = argparse.ArgumentParser(description="Save report figures from recorded results.")
    parser.add_argument("--config", default="a1")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard version directory. Default: latest run.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--results", type=str, default=None, help="Directory of result CSVs. Missing files are skipped.")
    return parser.parse_args()


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows. Refusing to draw an empty figure.")
    return rows


def _number(row, key):
    return float(row[key])


def _save(fig, path, source_csv):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if source_csv is not None:
        shutil.copy(source_csv, path.with_suffix(".csv"))
    print(f"Saved {path}")


def _correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) < 3 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return None, None
    pearson = float(np.corrcoef(x_values, y_values)[0, 1])
    x_rank = np.argsort(np.argsort(x_values)).astype(float)
    y_rank = np.argsort(np.argsort(y_values)).astype(float)
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman


def plot_recon_vs_knn(csv_path, out_path):
    """One point per proxy trial. The selected trial is marked."""
    rows = _read_csv(csv_path)
    x_values = [_number(row, "val_mse") for row in rows]
    y_values = [_number(row, "knn_top1_percent") for row in rows]
    selected = [row.get("selected", "0") in ("1", "True", "true") for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        [x for x, keep in zip(x_values, selected) if not keep],
        [y for y, keep in zip(y_values, selected) if not keep],
        label="Trial",
    )
    if any(selected):
        ax.scatter(
            [x for x, keep in zip(x_values, selected) if keep],
            [y for y, keep in zip(y_values, selected) if keep],
            marker="*",
            s=160,
            label="Selected",
        )
    pearson, spearman = _correlation(x_values, y_values)
    if pearson is not None:
        ax.text(
            0.02,
            0.98,
            f"Pearson {pearson:.2f}\nSpearman {spearman:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    ax.set_xlabel("Validation reconstruction MSE")
    ax.set_ylabel("Validation k-NN top-1 accuracy (%)")
    ax.set_title("Reconstruction error and representation quality")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_path, csv_path)


def plot_grouped_bars(csv_path, out_path, title, protocols):
    rows = _read_csv(csv_path)
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(protocols), 1)
    for index, protocol in enumerate(protocols):
        heights = []
        for model in models:
            match = [
                row for row in rows
                if row["model"] == model and row["protocol"] == protocol and row.get("fraction", "100") in ("100", "100.0")
            ]
            heights.append(_number(match[0], "top1_percent") if match else float("nan"))
        positions = np.arange(len(models)) + index * width
        ax.bar(positions, heights, width=width, label=protocol)
    ax.set_xticks(np.arange(len(models)) + width * (len(protocols) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_path, csv_path)


def plot_label_efficiency(csv_path, out_path, split):
    rows = [row for row in _read_csv(csv_path) if row.get("split", split) == split]
    if not rows:
        raise SystemExit(f"No {split} rows in {csv_path}.")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    for model in models:
        points = [row for row in rows if row["model"] == model]
        points.sort(key=lambda row: _number(row, "fraction"))
        ax.plot(
            [_number(row, "fraction") for row in points],
            [_number(row, "top1_percent") for row in points],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Labeled training data (%)")
    ax.set_ylabel(f"Top-1 {split} accuracy (%)")
    ax.set_title(f"Label efficiency ({split})")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_path, csv_path)


def plot_corruption(csv_path, out_path):
    rows = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    for model in models:
        points = [row for row in rows if row["model"] == model]
        points.sort(key=lambda row: _number(row, "sigma"))
        ax.plot(
            [_number(row, "sigma") for row in points],
            [_number(row, "top1_percent") for row in points],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Classification accuracy under input noise")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_path, csv_path)


def write_result_figures(results_dir, out_dir):
    """Draw each figure whose CSV exists. Missing files are left pending."""
    results_dir = Path(results_dir)
    jobs = (
        ("a2_trials.csv", "a2_recon_vs_knn.png", plot_recon_vs_knn),
        (
            "baselines.csv",
            "a3_baselines.png",
            lambda source, target: plot_grouped_bars(
                source, target, "Tiny-ImageNet baselines", ("knn", "linear_probe", "finetune", "supervised")
            ),
        ),
        (
            "transfer.csv",
            "a3_transfer.png",
            lambda source, target: plot_grouped_bars(
                source, target, "CIFAR-100 transfer", ("knn", "linear_probe", "finetune", "supervised")
            ),
        ),
        (
            "label_efficiency_val.csv",
            "a4_label_efficiency_val.png",
            lambda source, target: plot_label_efficiency(source, target, "val"),
        ),
        (
            "label_efficiency_test.csv",
            "a4_label_efficiency_test.png",
            lambda source, target: plot_label_efficiency(source, target, "test"),
        ),
        ("corruption.csv", "a4_corruption.png", lambda source, target: plot_corruption(source, target)),
    )
    drawn = 0
    for filename, figure_name, plot in jobs:
        source = results_dir / filename
        if not source.is_file():
            print(f"Skipping {figure_name}: {source} is not available yet.")
            continue
        plot(source, out_dir / figure_name)
        drawn += 1
    if drawn == 0:
        print("No result CSVs were available. No figures were drawn.")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="a1")
    known, _ = pre.parse_known_args()
    config.load(known.config)
    args = parse_args()
    out_dir = Path(args.out_dir or getattr(config, "FIGURES_DIR", "figures"))
    if args.results:
        write_result_figures(args.results, out_dir)
        return

    ckpt = Path(args.ckpt) if args.ckpt else Path(config.CHECKPOINT_DIR) / "last.ckpt"
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint {ckpt}. Train before saving figures.")
    logdir = Path(args.logdir) if args.logdir else latest_logdir(config.LOG_DIR)
    if not logdir.is_dir():
        raise SystemExit(f"Missing TensorBoard directory {logdir}.")

    save_reconstructions(ckpt, out_dir / "a1_reconstructions.png")
    save_loss_curve(logdir, out_dir / "a1_training_loss.png")
    print(f"Saved {out_dir / 'a1_reconstructions.png'}")
    print(f"Saved {out_dir / 'a1_training_loss.png'}")


if __name__ == "__main__":
    main()
