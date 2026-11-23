"""Shared evaluation for every encoder: k-NN, retrieval, and corruption."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from model import build_encoder
from train import build_datamodule, build_model


def l2_normalize(vectors):
    """Unit-length rows, so a dot product is cosine similarity."""
    return torch.nn.functional.normalize(vectors.float(), dim=-1)


def weighted_knn(query, bank, bank_labels, k, num_classes, chunk=256):
    """Cosine k-NN with similarity-weighted votes.

    Weights are the top-k cosine values clamped at 0, so a negative neighbor
    cannot cast a negative vote. Similarity is computed in chunks.
    """
    query = l2_normalize(query)
    bank = l2_normalize(bank)
    bank_labels = bank_labels.long()
    k = min(k, bank.shape[0])
    votes = []
    for start in range(0, query.shape[0], chunk):
        similarity = query[start:start + chunk] @ bank.T
        top_similarity, top_index = similarity.topk(k, dim=1)
        weights = top_similarity.clamp(min=0)
        labels = bank_labels[top_index]
        vote = similarity.new_zeros(similarity.shape[0], num_classes)
        vote.scatter_add_(1, labels, weights)
        votes.append(vote)
    return torch.cat(votes, dim=0)


def accuracy_from_scores(scores, targets):
    """Top-1 and Top-5 accuracy from class scores. Top-5 uses every class when there are fewer than five."""
    targets = targets.long()
    top1 = int((scores.argmax(dim=1) == targets).sum())
    k = min(5, scores.shape[1])
    shortlist = scores.topk(k, dim=1).indices
    top5 = int((shortlist == targets.unsqueeze(1)).any(dim=1).sum())
    total = targets.numel()
    return {
        "top1": top1 / total,
        "top5": top5 / total,
        "predictions": scores.argmax(dim=1),
    }


def per_class_accuracy(predictions, targets, num_classes):
    """Accuracy of each class. A class with no examples is omitted."""
    rows = []
    for class_id in range(num_classes):
        mask = targets == class_id
        if int(mask.sum()) == 0:
            continue
        rows.append({
            "class_id": class_id,
            "count": int(mask.sum()),
            "accuracy": float((predictions[mask] == class_id).float().mean()),
        })
    return rows


def retrieval_indices(query, bank, k):
    """Index of the k nearest bank rows for each query, by cosine similarity."""
    query = l2_normalize(query)
    bank = l2_normalize(bank)
    return (query @ bank.T).topk(min(k, bank.shape[0]), dim=1).indices


def corrupt(images, sigma, noise):
    """Scale one shared noise tensor. Sigma 0 returns the clean image."""
    if sigma == 0:
        return images.clamp(0, 1)
    return (images + float(sigma) * noise).clamp(0, 1)


def corruption_curve(model, images, labels, sigmas, seed):
    """Score one batch under several noise levels without retraining.

    The noise tensor is drawn once, then scaled, so the corruptions are paired.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator)
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for sigma in sigmas:
            logits = model(corrupt(images, sigma, noise))
            accuracy = float((logits.argmax(dim=1) == labels).float().mean())
            rows.append({"sigma": float(sigma), "top1": accuracy})
    if was_training:
        model.train()
    return rows


def corruption_loader(model, loader, sigmas, seed):
    """Validation corruption curve. Each batch reuses one noise draw across sigmas."""
    counts = {float(sigma): 0 for sigma in sigmas}
    total = 0
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            noise = torch.randn(images.shape, generator=generator)
            total += labels.numel()
            for sigma in sigmas:
                logits = model(corrupt(images, sigma, noise))
                counts[float(sigma)] += int((logits.argmax(dim=1) == labels).sum())
    return [{"sigma": sigma, "top1": counts[sigma] / total} for sigma in counts]


def _unshuffled_loader(datamodule, dataset):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=datamodule.batch_size,
        num_workers=datamodule.num_workers,
        shuffle=False,
    )


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_loader(module, loader, device):
    """Clean representations. The encoder is in eval mode and batches stay small."""
    module = module.to(device)
    module.eval()
    features = []
    labels = []
    for images, batch_labels in loader:
        features.append(module.encode(images.to(device)).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels)


def _results_path(name):
    directory = Path(getattr(config, "RESULTS_DIR", "results"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def run_knn(label_fraction, split, chunk=256):
    """k-NN of clean embeddings. The bank is the labeled training subset."""
    if getattr(config, "PROXY_PER_CLASS", None):
        config.LABEL_FRACTION = 1.0
    else:
        config.LABEL_FRACTION = label_fraction / 100
    datamodule = build_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    module = _load_encoder_module()
    device = pick_device()
    bank, bank_labels = embed_loader(module, _unshuffled_loader(datamodule, datamodule.train_ds), device)
    query_set = datamodule.val_ds if split == "val" else datamodule.test_ds
    query, query_labels = embed_loader(module, _unshuffled_loader(datamodule, query_set), device)
    scores = weighted_knn(
        query,
        bank,
        bank_labels,
        k=getattr(config, "KNN_K", 20),
        num_classes=config.NUM_CLASSES,
        chunk=chunk,
    )
    measured = accuracy_from_scores(scores, query_labels)
    payload = {
        "experiment": config.EXPERIMENT_NAME,
        "protocol": "knn",
        "dataset": getattr(config, "DATASET", "tinyimagenet"),
        "split": split,
        "label_fraction": label_fraction,
        "top1": measured["top1"],
        "top5": measured["top5"],
    }
    suffix = "knn_val" if getattr(config, "PROXY_PER_CLASS", None) else f"knn_{split}_f{label_fraction}"
    path = _results_path(f"{config.EXPERIMENT_NAME}_{suffix}.json")
    path.write_text(json.dumps(payload, indent=2))
    print(f"k-NN {split} top-1 {measured['top1']:.4f}, top-5 {measured['top5']:.4f}")
    print(f"Saved {path}")
    return payload


def run_per_class(label_fraction, split):
    """Per-class k-NN accuracy on the chosen query split."""
    if getattr(config, "PROXY_PER_CLASS", None):
        config.LABEL_FRACTION = 1.0
    else:
        config.LABEL_FRACTION = label_fraction / 100
    datamodule = build_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    module = _load_encoder_module()
    device = pick_device()
    bank, bank_labels = embed_loader(module, _unshuffled_loader(datamodule, datamodule.train_ds), device)
    query_set = datamodule.val_ds if split == "val" else datamodule.test_ds
    query, query_labels = embed_loader(module, _unshuffled_loader(datamodule, query_set), device)
    scores = weighted_knn(
        query, bank, bank_labels, k=getattr(config, "KNN_K", 20), num_classes=config.NUM_CLASSES
    )
    predictions = scores.argmax(dim=1)
    rows = per_class_accuracy(predictions, query_labels, config.NUM_CLASSES)
    path = _results_path(f"{config.EXPERIMENT_NAME}_per_class_{split}.json")
    path.write_text(json.dumps(rows, indent=2))
    print(f"Saved {path}")
    return rows


def run_retrieval(out_dir, queries=8, neighbors=5):
    """Show validation queries from distinct classes and their nearest training images.

    A 1-NN label match means the closest training image has the query label.
    That is separate from weighted k-NN classification accuracy.
    """
    config.LABEL_FRACTION = 1.0
    config.PROXY_PER_CLASS = None
    datamodule = build_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    module = _load_encoder_module()
    device = pick_device()
    bank, bank_labels = embed_loader(module, _unshuffled_loader(datamodule, datamodule.train_ds), device)
    query, query_labels = embed_loader(module, _unshuffled_loader(datamodule, datamodule.val_ds), device)
    neighbor_index = _nearest_indices(query, bank, neighbors)
    neighbor_labels = bank_labels[neighbor_index]
    collected = []
    for row in range(query_labels.shape[0]):
        collected.append({
            "query_index": row,
            "label": int(query_labels[row]),
            "neighbor_index": neighbor_index[row],
            "neighbor_labels": neighbor_labels[row],
            "correct": bool(int(neighbor_labels[row, 0]) == int(query_labels[row])),
        })
    chosen = _pick_retrieval_examples(collected, queries, config.NUM_CLASSES, config.SEED)
    for item in chosen:
        image, label = datamodule.val_ds[item["query_index"]]
        item["image"] = image
        if int(label) != item["label"]:
            raise RuntimeError(f"Validation row {item['query_index']} has label {label}, expected {item['label']}.")
    classes = len({item["label"] for item in chosen})
    matches = sum(item["correct"] for item in chosen)
    print(
        f"Selected {len(chosen)} validation queries from {classes} classes "
        f"({matches} 1-NN label matches, {len(chosen) - matches} 1-NN label mismatches)."
    )
    neighbor_images = _gather_neighbors(datamodule, chosen, neighbors)
    path = Path(out_dir) / "a5_1nn_retrieval.png"
    _save_retrieval_figure(chosen, neighbor_images, path)
    _save_retrieval_csv(chosen, path.with_suffix(".csv"))
    print(f"Saved {path}")
    print(f"Saved {path.with_suffix('.csv')}")


def _nearest_indices(query, bank, neighbors, chunk=128):
    """Cosine nearest neighbors, one chunk of queries at a time."""
    parts = []
    for start in range(0, query.shape[0], chunk):
        parts.append(retrieval_indices(query[start:start + chunk], bank, neighbors))
    return torch.cat(parts)


def _pick_retrieval_examples(collected, queries, num_classes, seed):
    """One query per class, aiming for half 1-NN matches and half mismatches.

    The class order is a seeded shuffle, so the sample is not the first
    validation batches. A class is used at most once.
    """
    generator = torch.Generator().manual_seed(seed)
    class_order = torch.randperm(num_classes, generator=generator).tolist()
    by_class = {}
    for position, item in enumerate(collected):
        by_class.setdefault(item["label"], []).append(position)
    used = set()

    def pick(class_id, want_match):
        candidates = [
            position for position in by_class.get(class_id, [])
            if collected[position]["correct"] is want_match
        ]
        if not candidates:
            return None
        offset = int(torch.randint(len(candidates), (1,), generator=generator))
        return candidates[offset]

    def fill(want_match, count):
        picked = []
        for class_id in class_order:
            if class_id in used or len(picked) >= count:
                continue
            position = pick(class_id, want_match)
            if position is None:
                continue
            used.add(class_id)
            picked.append(position)
        return picked

    half = queries // 2
    match_rows = fill(True, half)
    mismatch_rows = fill(False, half)
    if len(match_rows) < half:
        match_rows.extend(fill(True, half - len(match_rows)))
    if len(mismatch_rows) < half:
        mismatch_rows.extend(fill(False, half - len(mismatch_rows)))
    chosen_rows = match_rows + mismatch_rows
    if len(chosen_rows) < queries:
        for class_id in class_order:
            if class_id in used or len(chosen_rows) >= queries:
                continue
            rows = by_class.get(class_id, [])
            if not rows:
                continue
            offset = int(torch.randint(len(rows), (1,), generator=generator))
            used.add(class_id)
            chosen_rows.append(rows[offset])
    chosen = [collected[position] for position in chosen_rows[:queries]]
    chosen.sort(key=lambda item: not item["correct"])
    return chosen


def _gather_neighbors(datamodule, chosen, neighbors):
    """Load the training images that retrieval selected."""
    panels = []
    for item in chosen:
        images = [datamodule.train_ds[int(index)][0] for index in item["neighbor_index"].tolist()]
        panels.append(torch.stack(images))
    return panels


def _save_retrieval_figure(chosen, neighbor_images, path):
    columns = 1 + neighbor_images[0].shape[0]
    fig, axes = plt.subplots(len(chosen), columns, figsize=(1.5 * columns, 1.7 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)
    for row, item in enumerate(chosen):
        _show(axes[row, 0], item["image"])
        status = "1-NN label match" if item["correct"] else "1-NN label mismatch"
        nearest = int(item["neighbor_labels"][0])
        axes[row, 0].set_ylabel(f"Query class {item['label']}\n{status}", fontsize=7)
        for column in range(neighbor_images[row].shape[0]):
            _show(axes[row, column + 1], neighbor_images[row][column])
            if column == 0:
                axes[row, column + 1].set_title(f"NN class {nearest}", fontsize=8)
    fig.suptitle("1-NN retrieval in representation h", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_retrieval_csv(chosen, path):
    """Query class, shown neighbor classes, and whether the 1-NN label matches."""
    neighbor_count = len(chosen[0]["neighbor_labels"]) if chosen else 0
    fieldnames = [
        "query_index",
        "query_class_id",
        "nearest_neighbor_class_id",
        "nearest_neighbor_matches",
    ]
    fieldnames.extend(f"neighbor_{rank}_class_id" for rank in range(1, neighbor_count + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in chosen:
            labels = [int(label) for label in item["neighbor_labels"].tolist()]
            row = {
                "query_index": item["query_index"],
                "query_class_id": item["label"],
                "nearest_neighbor_class_id": labels[0],
                "nearest_neighbor_matches": int(item["correct"]),
            }
            for rank, label in enumerate(labels, start=1):
                row[f"neighbor_{rank}_class_id"] = label
            writer.writerow(row)


def _show(axis, image):
    axis.imshow(image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    axis.set_xticks([])
    axis.set_yticks([])


def _load_encoder_module():
    model_name = getattr(config, "MODEL", "dae")
    if model_name == "random":
        module = build_encoder(config.LATENT_DIM)
        state = torch.load(config.ENCODER_PATH, map_location="cpu", weights_only=True)
        module.load_state_dict(state)

        class _Wrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def encode(self, images):
                return self.encoder(images)

        return _Wrapper(module)
    if model_name == "dae":
        checkpoint = Path(config.CHECKPOINT_DIR) / "last.ckpt"
        if checkpoint.is_file():
            from model import DenoisingAutoEncoder
            return DenoisingAutoEncoder.load_from_checkpoint(checkpoint, map_location="cpu")
        module = build_model()
        state = torch.load(config.ENCODER_PATH, map_location="cpu", weights_only=True)
        module.encoder.load_state_dict(state)
        return module
    return build_model()


def run_corruption(split):
    """Score a finished classifier at several noise levels. The weights stay fixed."""
    from model import EncoderClassifier, SupervisedEncoder

    checkpoint = Path(config.CHECKPOINT_DIR) / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No classifier checkpoint at {checkpoint}.")
    model_name = getattr(config, "MODEL", "")
    if model_name == "supervised":
        model = SupervisedEncoder.load_from_checkpoint(checkpoint, map_location="cpu")
    else:
        model = EncoderClassifier.load_from_checkpoint(checkpoint, map_location="cpu")
    datamodule = build_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    dataset = datamodule.val_ds if split == "val" else datamodule.test_ds
    rows = corruption_loader(
        model,
        _unshuffled_loader(datamodule, dataset),
        sigmas=(0.0, 0.05, 0.10, 0.20),
        seed=config.SEED,
    )
    for row in rows:
        row["experiment"] = config.EXPERIMENT_NAME
        row["split"] = split
        row["top1_percent"] = 100 * row["top1"]
    path = _results_path(f"{config.EXPERIMENT_NAME}_corruption_{split}.json")
    path.write_text(json.dumps(rows, indent=2))
    print(f"Saved {path}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved encoder. This does not train.")
    parser.add_argument("--config", default="a1")
    parser.add_argument("--protocol", choices=("knn", "per_class", "retrieval", "corruption"), required=True)
    parser.add_argument("--label-fraction", type=int, default=100)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="a1")
    known, _ = pre.parse_known_args()
    config.load(known.config)
    args = parse_args()
    if args.protocol == "knn":
        run_knn(args.label_fraction, args.split)
    elif args.protocol == "per_class":
        run_per_class(args.label_fraction, args.split)
    elif args.protocol == "retrieval":
        run_retrieval(args.out_dir or getattr(config, "FIGURES_DIR", "figures"))
    else:
        run_corruption(args.split)


if __name__ == "__main__":
    main()
