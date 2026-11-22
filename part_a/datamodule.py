import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.transforms import functional as TF
from datasets import load_dataset

import config


class TinyImageNetSubset(Dataset):
    """Images from one Hugging Face split, restricted to a list of row indices.

    Each image is converted to RGB and then to a float tensor in [0, 1].
    The label is returned so a later probe can reuse this loader. The
    autoencoder does not train on it.
    """

    def __init__(self, hf_split, indices):
        self.hf_split = hf_split
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # Index the row first. Reading the image column as a whole decodes every file.
        row = self.hf_split[int(self.indices[i])]
        image = row["image"].convert("RGB")
        x = TF.to_tensor(image)  # [3, H, W], values in [0, 1]
        y = int(row["label"])
        return x, y


def balanced_train_val_split(labels, val_per_class, num_classes, seed):
    """Hold out exactly val_per_class indices from each class.

    One generator walks the classes in order, so the same seed always
    yields the same sets. The two sets partition the original training
    indices: nothing is shared, and nothing is dropped.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    val_parts = []
    for class_id in range(num_classes):
        class_indices = torch.where(labels == class_id)[0]
        if class_indices.numel() < val_per_class:
            raise ValueError(
                f"Class {class_id} has {class_indices.numel()} images, "
                f"which is fewer than the {val_per_class} required for validation."
            )
        perm = torch.randperm(class_indices.numel(), generator=generator)
        shuffled = class_indices[perm]
        val_parts.append(shuffled[:val_per_class])
        train_parts.append(shuffled[val_per_class:])

    # Sorted indices keep the validation order fixed across runs. The training
    # loader still shuffles each epoch.
    train_indices = torch.cat(train_parts).sort().values.tolist()
    val_indices = torch.cat(val_parts).sort().values.tolist()
    return train_indices, val_indices


def class_counts(labels, num_classes):
    """Per-class counts, or None when a split has no class labels."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.numel() == 0 or int(labels.min()) < 0:
        return None
    return torch.bincount(labels, minlength=num_classes)


class TinyImageNetDataModule(pl.LightningDataModule):
    """Tiny-ImageNet from Hugging Face, with a balanced validation holdout.

    The original training split has 100,000 images. 50 images per class are
    held out for validation (10,000). The remaining 90,000 are the training
    set. The dataset's valid split (10,000) is the official test set and is
    not used to build the other two.
    """

    def __init__(
        self,
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
        val_per_class=config.VAL_PER_CLASS,
        num_classes=config.NUM_CLASSES,
        dataset_name=config.DATASET_NAME,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_per_class = val_per_class
        self.num_classes = num_classes
        self.dataset_name = dataset_name

    def prepare_data(self):
        # Download once. setup() reads the cached copy.
        load_dataset(self.dataset_name, cache_dir=self.data_dir)

    def setup(self, stage=None):
        ds = load_dataset(self.dataset_name, cache_dir=self.data_dir)
        train_labels = ds["train"]["label"]
        self.train_indices, self.val_indices = balanced_train_val_split(
            train_labels,
            val_per_class=self.val_per_class,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        self.test_indices = list(range(len(ds["valid"])))

        label_tensor = torch.as_tensor(train_labels, dtype=torch.long)
        self.train_labels = label_tensor[self.train_indices]
        self.val_labels = label_tensor[self.val_indices]
        self.test_labels = torch.as_tensor(ds["valid"]["label"], dtype=torch.long)

        self.train_ds = TinyImageNetSubset(ds["train"], self.train_indices)
        self.val_ds = TinyImageNetSubset(ds["train"], self.val_indices)
        self.test_ds = TinyImageNetSubset(ds["valid"], self.test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def _batch_stats(loader):
    images, _ = next(iter(loader))
    return {
        "shape": tuple(images.shape),
        "min": float(images.min()),
        "max": float(images.max()),
    }


def verify_dataset(dm):
    """Load the splits and print sizes, class balance, shape, and pixel range.

    Pixel range is measured on one batch from each split. ToTensor maps an
    8-bit image onto [0, 1], so this checks that conversion rather than
    scanning every file.
    """
    dm.prepare_data()
    dm.setup()

    overlap = len(set(dm.train_indices) & set(dm.val_indices))
    train_counts = class_counts(dm.train_labels, dm.num_classes)
    val_counts = class_counts(dm.val_labels, dm.num_classes)
    test_counts = class_counts(dm.test_labels, dm.num_classes)
    train_batch = _batch_stats(dm.train_dataloader())
    val_batch = _batch_stats(dm.val_dataloader())
    test_batch = _batch_stats(dm.test_dataloader())
    pixel_min = min(train_batch["min"], val_batch["min"], test_batch["min"])
    pixel_max = max(train_batch["max"], val_batch["max"], test_batch["max"])

    report = {
        "train_size": len(dm.train_ds),
        "val_size": len(dm.val_ds),
        "test_size": len(dm.test_ds),
        "overlap": overlap,
        "train_count_min": None if train_counts is None else int(train_counts.min()),
        "train_count_max": None if train_counts is None else int(train_counts.max()),
        "val_count_min": None if val_counts is None else int(val_counts.min()),
        "val_count_max": None if val_counts is None else int(val_counts.max()),
        "test_count_min": None if test_counts is None else int(test_counts.min()),
        "test_count_max": None if test_counts is None else int(test_counts.max()),
        "sample_shape": train_batch["shape"],
        "val_shape": val_batch["shape"],
        "test_shape": test_batch["shape"],
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
    }

    print("Tiny-ImageNet split check")
    print(f"  train: {report['train_size']} (per class {report['train_count_min']}..{report['train_count_max']})")
    print(f"  val:   {report['val_size']} (per class {report['val_count_min']}..{report['val_count_max']})")
    test_balance = (
        "labels unavailable"
        if report["test_count_min"] is None
        else f"per class {report['test_count_min']}..{report['test_count_max']}"
    )
    print(f"  test:  {report['test_size']} ({test_balance})")
    print(f"  train/val index overlap: {report['overlap']}")
    print(f"  batch shape: {report['sample_shape']}")
    print(f"  pixel range over one batch per split: [{report['pixel_min']:.4f}, {report['pixel_max']:.4f}]")
    return report
