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

The initial, untuned denoising run is `configs/a1.py`: latent dimension 128, noise sigma 0.10, Adam with learning rate 0.001, batch size 128, 10 epochs, and seed 42. The loss is mean squared error. The accelerator is `auto` and the precision is `16-mixed`, so the run uses MPS or CUDA when one is present and the CPU otherwise.

`python train.py --config a1` trains for 10 epochs and validates on the held-out 10,000 images. It does not score the official test split. There is no early stopping. The last checkpoint is `checkpoints/last.ckpt`, and the encoder is also written to `tiny_imagenet_encoder.pt`.

Every later experiment is another file in `configs/`. Each file has its own checkpoint directory, so runs do not overwrite one another. An interrupted run continues with `--resume` and the same config name.

## Later experiments

Do not start these until you intend to spend the compute. `python search.py` trains 12 proxy autoencoders on a fixed subset of 20 images per class (4,000 images) for 2 epochs. The validation set stays the full 10,000-image holdout. Selection uses validation k-NN top-1 accuracy, then lower validation MSE, then the smaller latent size. The script writes `configs/dae_selected.py`, `configs/vanilla.py`, `configs/random.py`, and the downstream configs.

The vanilla autoencoder is the selected denoising model with sigma 0. The random encoder is the same architecture at the selected width, saved from the seed and not trained. Supervised runs train that encoder from scratch on a labeled fraction. Linear probes freeze it. Fine-tuning trains it, at learning rate 0.0001. Random fine-tuning at 100% is omitted because it is the same experiment as supervised training on all labels.

Labeled fractions are 1%, 10%, and 100% of the training pool. On Tiny-ImageNet those totals are 900, 9,000, and 90,000. The 900-image subset gives five images to 100 classes and four to the other 100. CIFAR-100 is resized to 64×64. Fifty images per class are held out of its official training split, leaving 45,000 training images, 5,000 validation images, and the 10,000 official test images. Its 1% subset has 450 images.

k-NN uses clean images, an encoder in eval mode, cosine similarity of L2-normalized `h`, `k = 20`, and a vote weighted by cosine values clamped at 0. Proxy trials use the 4,000-image subset as the bank. Final k-NN uses the labeled fraction as the bank. The official test split is not used to choose hyperparameters.

## Commands

Install dependencies, then run the smoke test. It does not start the 10-epoch run or the 12 proxy trials.

```bash
pip install -r requirements.txt
python smoke_test.py
```

Initial A1 training, which is not the selected model:

```bash
python train.py --config a1
python visualize.py --config a1
```

Proxy search. This depends on nothing except the dataset cache, and it writes `configs/dae_selected.py`.

```bash
python search.py
```

Final pretraining. The vanilla run depends on `configs/vanilla.py` from the search.

```bash
python train.py --config dae_selected
python train.py --config vanilla
python train.py --config random
```

Tiny-ImageNet supervised training, probes, and fine-tunes. Probes and fine-tunes depend on the encoder file named in the config. Repeat the same names with `_1` and `_10`.

```bash
python train.py --config supervised_100
python train.py --config probe_dae_100
python train.py --config probe_vanilla_100
python train.py --config probe_random_100
python train.py --config finetune_dae_100
python train.py --config finetune_vanilla_100
```

Validation k-NN, then the official test split only after those configs are frozen:

```bash
python evaluate.py --config dae_selected --protocol knn --label-fraction 100 --split val
python train.py --config probe_dae_100 --test
```

CIFAR-100 uses the same commands with a `cifar_` prefix, for example `python train.py --config cifar_probe_dae_100`. There is no extra self-supervised pretraining on CIFAR-100.

Nearest neighbors, per-class accuracy, and the optional corruption curve:

```bash
python evaluate.py --config dae_selected --protocol retrieval
python evaluate.py --config dae_selected --protocol per_class --split val
python evaluate.py --config probe_dae_100 --protocol corruption --split val
```

Figures are drawn only from CSVs that already exist:

```bash
python visualize.py --results results
tensorboard --logdir tb_logs
```

Resume an interrupted run with `python train.py --config <name> --resume`.

## Output locations

- `dataset/` — Tiny-ImageNet and CIFAR-100 downloads
- `splits/` — shared labeled-subset indices
- `tb_logs/<timestamp>/<experiment>/` — TensorBoard logs
- `checkpoints/<experiment>/last.ckpt` — last checkpoint, except A1, which uses `checkpoints/last.ckpt`
- `checkpoints/<experiment>/encoder.pt` — encoder weights for that run
- `results/` — metrics JSON and the CSVs read by the figures
- `figures/` — reconstruction, loss, search, baseline, transfer, label-efficiency, retrieval, and corruption figures
- `REPORT.md` — methods, with results marked pending until the runs exist

Datasets, checkpoints, logs, and encoder files are gitignored. Figures and result tables are not.

The planned training count is 54 runs: 12 proxy models, 2 full autoencoders, and 20 Tiny-ImageNet plus 20 CIFAR-100 downstream runs. Saving the random encoder does not train. k-NN, retrieval, and corruption scoring do not train. The code is in place; those experiments have not been run.

