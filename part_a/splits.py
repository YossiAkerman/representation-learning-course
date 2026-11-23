"""Class-balanced labeled subsets shared by every model."""

import json
from pathlib import Path

import torch


def class_balanced_subset(pool_indices, pool_labels, total, num_classes, seed):
    """Choose `total` indices from the pool, as evenly as possible per class.

    A seeded shuffle decides which classes receive one extra image when the
    total does not divide evenly. The same seed returns the same indices.
    """
    pool_indices = torch.as_tensor(pool_indices, dtype=torch.long)
    pool_labels = torch.as_tensor(pool_labels, dtype=torch.long)
    if total == pool_indices.numel():
        return pool_indices.sort().values.tolist()
    if total > pool_indices.numel():
        raise ValueError(f"Asked for {total} images, but the pool has {pool_indices.numel()}.")

    base, extra = divmod(int(total), num_classes)
    generator = torch.Generator().manual_seed(seed)
    class_order = torch.randperm(num_classes, generator=generator).tolist()
    extra_classes = set(class_order[:extra])
    chosen = []
    for class_id in range(num_classes):
        class_pos = torch.where(pool_labels == class_id)[0]
        need = base + (1 if class_id in extra_classes else 0)
        if class_pos.numel() < need:
            raise ValueError(
                f"Class {class_id} has {class_pos.numel()} images in the pool, "
                f"which is fewer than the {need} requested."
            )
        perm = torch.randperm(class_pos.numel(), generator=generator)
        chosen.append(pool_indices[class_pos[perm[:need]]])
    return torch.cat(chosen).sort().values.tolist()


def subset_counts(pool_indices, pool_labels, chosen, num_classes):
    """How many chosen images belong to each class."""
    label_of = {int(index): int(label) for index, label in zip(pool_indices, pool_labels)}
    labels = torch.tensor([label_of[int(index)] for index in chosen], dtype=torch.long)
    return torch.bincount(labels, minlength=num_classes)


def load_or_create_subset(path, pool_indices, pool_labels, total, num_classes, seed, kind):
    """Return a saved subset, creating it when the file is missing or stale.

    The file is the source of truth so every model sees the same indices.
    """
    path = Path(path)
    expected = {
        "total": int(total),
        "num_classes": int(num_classes),
        "seed": int(seed),
        "kind": kind,
    }
    if path.is_file():
        saved = json.loads(path.read_text())
        if all(saved.get(key) == value for key, value in expected.items()):
            return saved["indices"]

    indices = class_balanced_subset(pool_indices, pool_labels, total, num_classes, seed)
    counts = subset_counts(pool_indices, pool_labels, indices, num_classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected,
        "counts": [int(value) for value in counts.tolist()],
        "indices": [int(index) for index in indices],
    }
    path.write_text(json.dumps(payload))
    return payload["indices"]
