# Part A: Denoising autoencoders on Tiny-ImageNet

This directory contains the completed Part A experiments. Pretraining learns a representation `h` by reconstruction. Labels are used for the splits and for the downstream evaluations. Run every command from this directory.

```bash
pip install -r requirements.txt
python smoke_test.py
```

The smoke test checks the data split, models, and one training batch. It does not repeat the full experiments.

## Splits

Tiny-ImageNet comes from `zh-plus/tiny-imagenet` and is cached in `dataset/`. The published training split has 100,000 images and 200 classes. Seed 42 holds out 50 images per class:

- 90,000 training images
- 10,000 validation images, with no index shared with training
- 10,000 official test images, the dataset's `valid` split

The labeled subsets of the training pool are saved in `splits/`. They contain 900 images (1%), 9,000 images (10%), and 90,000 images (100%). The 1% subset gives five images to 100 classes and four images to the other 100.

CIFAR-100 is resized to 64×64 RGB with values in [0, 1]. Seed 42 holds out 50 images per class from the official training split: 45,000 train, 5,000 validation, and 10,000 official test images. The 1% and 10% subsets contain 450 and 4,500 training images. CIFAR-100 does not receive a new self-supervised pretraining run. Its probes and fine-tunes load the Tiny-ImageNet encoders named in the config.

Validation chooses hyperparameters and is reported for every run. The official test split is scored only by `python train.py --config <name> --test` or `python evaluate.py --split test`.

## Workflow

Each experiment is one file in `configs/`. Training is `python train.py --config <name>`. Resume with the same name and `--resume`. The commands below are the runs that produced `results/`.

1. **A1.** `configs/a1.py` trains the initial denoising autoencoder: latent size 128, noise sigma 0.10, learning rate 0.001, batch size 128, 10 epochs, seed 42. `python visualize.py --config a1` writes the reconstruction and loss figures from the checkpoint and TensorBoard log.
2. **A2.** `python search.py` trains 12 proxy models for 2 epochs on 20 images per class. It ranks validation k-NN Top-1, then lower validation MSE, then smaller latent size. The selected setting is sigma 0.20, latent size 256, and learning rate 0.001 (`configs/a2_05.py`, recorded in `results/a2_trials.csv`).
3. **A3 pretraining and baselines.** `dae_selected`, `vanilla` (the same model with sigma 0), and `random` (the seeded encoder, not trained). Downstream configs use learning rate 0.001 for a linear probe and 0.0001 for fine-tuning. The supervised comparison at that fine-tuning rate is `supervised_100_lr1e4`, `supervised_10_lr1e4`, and `supervised_1_lr1e4`. `supervised_100` is the same architecture at learning rate 0.001. There is no separate `finetune_random_100` config.
4. **A4.** Repeat the probe, fine-tune, and `supervised_*_lr1e4` names with `_10` and `_1`.
5. **A5.** `python evaluate.py --config dae_selected --protocol per_class --split val` and the same command for `vanilla`. Retrieval writes a class-diverse 1-NN panel:

```bash
python evaluate.py --config dae_selected --protocol retrieval --out-dir figures/a5_dae_1nn
python evaluate.py --config vanilla --protocol retrieval --out-dir figures/a5_vanilla_1nn
```

k-NN uses clean images, cosine similarity of L2-normalized `h`, `k = 20`, and votes weighted by cosine values clamped at 0. The bank is the labeled training fraction.

```bash
python evaluate.py --config dae_selected --protocol knn --label-fraction 100 --split val
python train.py --config probe_dae_100 --test
```

CIFAR-100 uses the `cifar_` configs, including `cifar_supervised_100_lr1e4`, `cifar_supervised_10_lr1e4`, and `cifar_supervised_1_lr1e4`. There is no `cifar_finetune_random_100` config.

## Where the files are

- `results/` — fit summaries, test summaries, k-NN JSON, `a2_trials.csv`, and per-class JSON
- `figures/a1_reconstructions.png`, `figures/a1_training_loss.png` — initial A1 run
- `figures/a2_recon_vs_knn.png` and the CSV beside it — proxy search
- `figures/a5_dae/` and `figures/a5_vanilla/` — first retrieval panels, one query category
- `figures/a5_dae_1nn/` and `figures/a5_vanilla_1nn/` — eight query classes, with a CSV of class IDs
- `figures/a5_dae_diverse/` and `figures/a5_vanilla_diverse/` — the same retrieval table as the `a5_*_1nn` directories
- `checkpoints/last.ckpt` and `tiny_imagenet_encoder.pt` — A1
- `checkpoints/<experiment>/last.ckpt` and `checkpoints/<experiment>/encoder.pt` — later runs
- `tb_logs/<timestamp>/<experiment>/` — TensorBoard event files
- `REPORT.md` — methods and the recorded numbers

`dataset/`, `tb_logs/`, checkpoints, and `*.pt` files are gitignored. Result JSON, result CSV, and figures are kept.

## TensorBoard

TensorBoard logs are written only on the machine that trains. Each run creates `tb_logs/<timestamp>/<experiment>/`. The root `.gitignore` excludes `**/tb_logs/`, so the event files are not in the Git repository and this project does not record a public TensorBoard URL. From this directory, the local curves are opened with:

```bash
tensorboard --logdir tb_logs
```

The scalar values used in `REPORT.md` are the JSON files in `results/`, not a remote dashboard.


### TensorBoard Training Logs

The training logs for Part A are provided in `tensorboard_logs_part_a.zip`.

To inspect the recorded training and validation metrics, extract the archive from the `part_a` directory:

```bash
unzip tensorboard_logs_part_a.zip
tensorboard --logdir tb_logs
```

The archive contains the recorded TensorBoard event files. Datasets and model checkpoints are excluded from the repository.