# Part A1: Denoising Autoencoder on Tiny-ImageNet

This directory trains a convolutional denoising autoencoder on Tiny-ImageNet. The encoder is the small two-layer network from lecture 4, adapted to 64×64 RGB images. Pretraining uses reconstruction only. Class labels are used to build a balanced validation split and are not training targets.

Run every command from this directory.

## Dataset preparation

Images come from the Hugging Face dataset `zh-plus/tiny-imagenet`. The first run downloads it into `dataset/`.

The published training split contains 100,000 images across 200 classes. A generator seeded with 42 holds out 50 images from each class, giving:

- 90,000 training images
- 10,000 validation images, 50 per class, with no index in common with the training set
- 10,000 official test images, taken from the dataset's `valid` split and not used to form the other two sets

Every image is converted to RGB and then to a float tensor in [0, 1], with shape `[3, 64, 64]`. `NUM_WORKERS` is 0 because decoding these files in forked workers is unreliable on macOS. On a Linux machine or Colab, that value can be raised in `configs/a1.py`.

## Model architecture

The encoder maps `[B, 3, 64, 64]` to a 128-dimensional vector `h`:

1. Convolution, 3 to 32 channels, kernel 3, stride 2, padding 1, then ReLU. Spatial size 64 to 32.
2. Convolution, 32 to 64 channels, kernel 3, stride 2, padding 1, then ReLU. Spatial size 32 to 16.
3. Flatten, then a linear layer from `64 × 16 × 16` to 128.

`encode(x)` returns `h`. The decoder is the mirror image: a linear layer, ReLU, unflatten to `[B, 64, 16, 16]`, and two transposed convolutions with `output_padding=1` that restore `[B, 3, 64, 64]`. A sigmoid keeps pixels in (0, 1).

## Denoising

Each training image is corrupted with Gaussian noise of standard deviation 0.10. The noisy image is clamped to [0, 1], and the network is trained to reconstruct the clean image. Training noise follows the process RNG after the global seed is set, so a full run is reproducible.

Validation and test noise use a separate generator that is reseeded to 42 at the start of every evaluation epoch. The validation loader is not shuffled, so each epoch is scored on the same corrupted inputs and the validation loss can be compared across epochs.

## Reconstruction loss

The objective is the mean squared error between the reconstruction and the clean image. The label is ignored.

## Training configuration

| Setting | Value |
| --- | --- |
| Latent dimension | 128 |
| Noise sigma | 0.10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 10 |
| Seed | 42 |
| Loss | MSE |
| Accelerator | `auto` (GPU when available, otherwise CPU) |
| Precision | `16-mixed` (bf16 fallback on CPU) |

The settings for this run live in `configs/a1.py`. `python train.py --config a1` loads that file and trains for 10 epochs on the held-out 10,000 validation images. It does not score the official test split. There is no early stopping, so the run uses the full epoch budget. The last checkpoint is written to `checkpoints/last.ckpt`. When that file is in the default checkpoint directory, the encoder weights are also written to `tiny_imagenet_encoder.pt`. A later experiment adds `configs/<name>.py` and is started with `--config <name>`.

An interrupted run continues with `python train.py --config a1 --resume`. Resume starts a new TensorBoard directory. `visualize.py` reads the latest directory unless `--logdir` points at an earlier one.

The official test split is reserved for final evaluation:

```bash
python train.py --config a1 --test
```

## Commands

Install dependencies, then run the smoke test. The smoke test downloads Tiny-ImageNet, checks the splits and the network, and runs one training batch and one validation batch through `train.py`. It does not start the 10-epoch run.

```bash
pip install -r requirements.txt
python smoke_test.py
```

Initial training:

```bash
python train.py --config a1
```

TensorBoard:

```bash
tensorboard --logdir tb_logs
```

After that training run has finished, write the two report figures from the checkpoint and the log. Do not run this on a smoke-test log; the smoke artifacts are deleted, and the script will not invent a curve.

```bash
python visualize.py
```

## Output locations

- `dataset/` — Hugging Face cache
- `tb_logs/<timestamp>/a1/` — training and validation loss
- `checkpoints/last.ckpt` — last training checkpoint
- `tiny_imagenet_encoder.pt` — encoder weights from a run that used the default checkpoint directory
- `figures/a1_reconstructions.png` — eight validation images, original, noisy, and reconstructed
- `figures/a1_training_loss.png` — training and validation MSE by epoch

Datasets, checkpoints, logs, encoder files, and generated figures are gitignored.
