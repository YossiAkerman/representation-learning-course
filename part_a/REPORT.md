# Part A report

Percentages below are 100 times the fraction stored in `results/`, rounded to two decimal places. The source file for each table is named in the section. This file is not a PDF.

## 1. Methods

The question is whether a denoising autoencoder learns a representation that is useful beyond its reconstruction loss. The encoder maps a 64×64 RGB image to a vector `h`. A decoder reconstructs the clean image. Labels are not used during autoencoder training.

Tiny-ImageNet is `zh-plus/tiny-imagenet`: 100,000 published training images and 200 classes. Seed 42 holds out 50 images per class, leaving 90,000 training images and 10,000 validation images. The dataset's `valid` split, 10,000 images, is the official test set. CIFAR-100 is resized to 64×64. The same seed holds out 50 images per class from its official training split, leaving 45,000 training images and 5,000 validation images. Its official test split has 10,000 images. Labeled subsets are class-balanced and shared across models: 1%, 10%, and 100% of the training pool.

The encoder is two stride-2 convolutions, 3 to 32 and 32 to 64, each followed by ReLU, then a linear layer to `h`. The decoder is the mirror image and ends in a sigmoid. The representation used downstream is `h`. Decoder weights are discarded after pretraining.

Training corrupts the input as `clamp(x + sigma * noise, 0, 1)` and minimizes

```text
L_rec = MSE(decoder(encoder(x_noisy)), x_clean)
```

`configs/a1.py` is the initial run: sigma 0.10, latent size 128, Adam learning rate 0.001, batch size 128, 10 epochs, seed 42. A seeded search then draws 12 of the 27 combinations of sigma in {0.05, 0.10, 0.20}, latent size in {64, 128, 256}, and learning rate in {0.0001, 0.0003, 0.001}. Each proxy model trains for 2 epochs on 20 images per class. Selection uses validation k-NN Top-1, then lower validation MSE, then the smaller latent size. The selected denoising model is retrained on all 90,000 training images. The vanilla autoencoder uses that setting with sigma 0. The random encoder is the same width, saved from seed 42 and not trained.

k-NN reads clean images with the encoder in evaluation mode. It L2-normalizes `h`, uses cosine similarity, sets `k = 20`, and weights the vote by the cosine value after clamping it at 0. A linear probe freezes the encoder and trains a new head at learning rate 0.001. Fine-tuning trains the encoder and a new head at learning rate 0.0001. The supervised baseline trains the same encoder from random initialization. The comparison reported with the fine-tunes uses learning rate 0.0001 (`supervised_*_lr1e4` and `cifar_supervised_*_lr1e4`). `supervised_100` is an additional run at learning rate 0.001. CIFAR-100 probes and fine-tunes load `checkpoints/dae_selected/encoder.pt`, `checkpoints/vanilla/encoder.pt`, or `checkpoints/random/encoder.pt`. No autoencoder is trained on CIFAR-100.

A 1-NN label match means the nearest training image has the query label. It is not the weighted k-NN accuracy.

## 2. Results

### A1 and A2

`results/a1.json` records the initial denoising run. Final training MSE is 0.0150 and validation MSE is 0.0152. The reconstruction panel is `figures/a1_reconstructions.png`, and the loss curve is `figures/a1_training_loss.png`.

`results/a2_trials.csv` selects `a2_05`: sigma 0.20, latent size 256, learning rate 0.001, proxy validation MSE 0.0427, and proxy k-NN Top-1 2.15%. The scatter plot is `figures/a2_recon_vs_knn.png`.

After full training, `results/dae_selected.json` records training MSE 0.0125 and validation MSE 0.0128. `results/vanilla.json` records training MSE 0.0117 and validation MSE 0.0119.

### Representation quality

Validation and test k-NN for the three Tiny-ImageNet encoders, as Top-1 / Top-5:

| Encoder | Val 1% | Val 10% | Val 100% | Test 1% | Test 10% | Test 100% |
|---|---:|---:|---:|---:|---:|---:|
| DAE | 2.56 / 8.28 | 4.50 / 13.10 | 6.95 / 17.95 | 2.47 / 8.43 | 4.48 / 12.66 | 6.85 / 17.81 |
| Vanilla | 2.47 / 8.62 | 4.36 / 13.51 | 7.07 / 17.93 | 2.64 / 8.29 | 4.29 / 12.88 | 6.88 / 17.85 |
| Random | 1.87 / 6.38 | 2.73 / 8.92 | 3.88 / 11.31 | 1.78 / 6.35 | 2.84 / 9.02 | 3.95 / 11.79 |

The files are `results/{dae_selected,vanilla,random}_knn_{val,test}_f{1,10,100}.json`.

On the full validation bank, per-class k-NN accuracy is zero for 53 of 200 classes with the DAE and 52 of 200 with the vanilla autoencoder. Each class has 50 validation images. The Pearson correlation of the two per-class vectors is 0.962. The files are `results/dae_selected_per_class_val.json` and `results/vanilla_per_class_val.json`.

The class-diverse retrieval panels use eight query classes, four 1-NN label matches, and four 1-NN label mismatches. They are `figures/a5_dae_1nn/` and `figures/a5_vanilla_1nn/`, including the CSV of query and neighbor class IDs. `figures/a5_*_diverse/` stores the same CSV. The earlier panels in `figures/a5_dae/` and `figures/a5_vanilla/` use one query category.

### Tiny-ImageNet classifiers

Validation Top-1 / Top-5 from the fit JSON. The learning rate is 0.001 for probes and for `supervised_100`, and 0.0001 for fine-tuning and `supervised_*_lr1e4`.

| Model | 1% | 10% | 100% |
|---|---:|---:|---:|
| Supervised, lr 0.0001 | 0.97 / 4.65 | 7.64 / 20.83 | 17.41 / 38.03 |
| Supervised, lr 0.001 | — | — | 11.79 / 29.11 |
| Probe, DAE | 2.19 / 7.76 | 4.73 / 14.21 | 6.66 / 19.07 |
| Probe, vanilla | 2.20 / 7.85 | 5.00 / 15.25 | 7.02 / 19.65 |
| Probe, random | 0.86 / 2.96 | 2.22 / 9.02 | 4.80 / 14.72 |
| Fine-tune, DAE | 0.94 / 4.04 | 6.36 / 18.37 | 16.72 / 37.18 |
| Fine-tune, vanilla | 0.92 / 4.17 | 6.56 / 18.06 | 17.10 / 38.20 |
| Fine-tune, random | 0.97 / 4.65 | 7.64 / 20.83 | — |

`supervised_100` has no test file. Its validation loss is 10.9205 and its training accuracy is 94.24%. There is no `finetune_random_100` result. The recorded 1% and 10% random fine-tune metrics match `supervised_1_lr1e4` and `supervised_10_lr1e4`.

Official test Top-1 / Top-5:

| Model | 1% | 10% | 100% |
|---|---:|---:|---:|
| Supervised, lr 0.0001 | 1.04 / 4.50 | 7.47 / 20.56 | 17.54 / 38.79 |
| Probe, DAE | 2.01 / 7.76 | 4.74 / 14.68 | 6.99 / 19.60 |
| Probe, vanilla | 2.20 / 7.97 | 5.27 / 15.81 | 7.19 / 19.76 |
| Probe, random | 0.73 / 2.82 | 2.36 / 9.14 | 4.74 / 15.38 |
| Fine-tune, DAE | 1.19 / 4.17 | 6.04 / 18.15 | 17.50 / 38.37 |
| Fine-tune, vanilla | 1.08 / 4.27 | 6.05 / 18.55 | 17.25 / 39.20 |
| Fine-tune, random | 1.04 / 4.50 | 7.47 / 20.56 | — |

### CIFAR-100 transfer

k-NN of the Tiny-ImageNet encoders on CIFAR-100. Validation files are `results/cifar_probe_{dae,vanilla,random}_{1,10,100}_knn_val_f{1,10,100}.json`. Test files are `results/cifar_probe_{dae,vanilla,random}_100_knn_test_f{1,10,100}.json`.

| Encoder | Val 1% | Val 10% | Val 100% | Test 1% | Test 10% | Test 100% |
|---|---:|---:|---:|---:|---:|---:|
| DAE | 7.40 / 21.84 | 13.70 / 33.16 | 21.08 / 43.56 | 6.79 / 21.02 | 12.92 / 32.66 | 20.48 / 43.53 |
| Vanilla | 7.86 / 21.76 | 13.34 / 33.00 | 21.04 / 43.88 | 6.87 / 21.14 | 12.96 / 32.33 | 20.47 / 43.61 |
| Random | 5.94 / 18.86 | 11.06 / 28.02 | 17.46 / 37.60 | 5.81 / 17.79 | 10.59 / 27.26 | 17.05 / 37.33 |

Classifier validation Top-1 / Top-5:

| Model | 1% | 10% | 100% |
|---|---:|---:|---:|
| Supervised, lr 0.0001 | 3.44 / 14.00 | 15.44 / 36.80 | 28.18 / 55.64 |
| Probe, DAE | 5.98 / 19.28 | 14.76 / 35.66 | 18.84 / 41.50 |
| Probe, vanilla | 6.10 / 19.06 | 15.04 / 36.18 | 19.14 / 42.58 |
| Probe, random | 1.28 / 5.64 | 4.08 / 16.18 | 11.20 / 31.08 |
| Fine-tune, DAE | 2.80 / 9.72 | 14.84 / 36.32 | 31.00 / 58.52 |
| Fine-tune, vanilla | 2.84 / 10.64 | 15.68 / 36.58 | 31.18 / 58.24 |
| Fine-tune, random | 3.44 / 14.00 | 15.44 / 36.80 | — |

Official test Top-1 / Top-5:

| Model | 1% | 10% | 100% |
|---|---:|---:|---:|
| Supervised, lr 0.0001 | 3.34 / 14.10 | 14.79 / 36.50 | 28.39 / 56.07 |
| Probe, DAE | 6.03 / 18.98 | 13.57 / 33.80 | 18.50 / 40.97 |
| Probe, vanilla | 5.53 / 17.61 | 13.87 / 34.47 | 18.94 / 41.03 |
| Probe, random | 1.28 / 5.78 | 3.53 / 16.32 | 10.94 / 30.56 |
| Fine-tune, DAE | 2.75 / 10.46 | 13.99 / 35.31 | 30.52 / 59.17 |
| Fine-tune, vanilla | 2.71 / 10.66 | 13.65 / 35.55 | 29.94 / 58.14 |
| Fine-tune, random | 3.34 / 14.10 | 14.79 / 36.50 | — |

No corruption-curve file is present in `results/`.

## 3. Discussion

Reconstruction pretraining beats a random encoder on every k-NN column above. It does not produce a consistent gain for denoising over the vanilla autoencoder. On Tiny-ImageNet at 100% labels, vanilla k-NN is 7.07% validation Top-1 and the DAE is 6.95%. The test scores are 6.88% and 6.85%. The vanilla validation reconstruction MSE is also lower, 0.0119 against 0.0128. Linear probes follow the same pattern on both datasets: the vanilla probe is ahead by a fraction of a percentage point at 100% labels.

Fine-tuning closes most of the gap with supervised training at learning rate 0.0001. On Tiny-ImageNet at 100% labels, validation Top-1 is 17.41% for supervised training, 17.10% for the vanilla fine-tune, and 16.72% for the DAE fine-tune. The DAE fine-tune's test Top-1, 17.50%, is slightly above the vanilla fine-tune, 17.25%, and just below the supervised test score, 17.54%. On CIFAR-100 at 100% labels, both fine-tunes exceed the supervised baseline: 31.18% and 31.00% validation Top-1, against 28.18%. The transferred encoders are therefore useful, while denoising is not uniformly better than removing the noise term.

The learning rate matters for the supervised baseline. At 0.001, Tiny-ImageNet training accuracy reaches 94.24% and validation Top-1 falls to 11.79%, with validation loss 10.9205. At 0.0001, validation Top-1 is 17.41%. The 0.0001 runs are the comparison used against fine-tuning.

Label fraction changes which protocol is strongest. At 1% on Tiny-ImageNet, the frozen probes outperform both fine-tuning and supervised training. At 100%, fine-tuning and supervised training are well above the probes. The same ordering appears on CIFAR-100. Accuracy increases from 1% to 10% to 100% for every completed protocol.

The per-class results show that the two autoencoders fail together. More than 50 classes have zero validation k-NN accuracy for each model, and the class-wise accuracies correlate at 0.962. The 1-NN panels include neighbors that share color or texture with the query while carrying a different class ID. The learned representation separates some categories, and it also keeps low-level appearance close together.

## References

Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th International Conference on Machine Learning.
