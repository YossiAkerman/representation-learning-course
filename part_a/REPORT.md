# Part A report

The numerical results in this file are pending. No experiment beyond the smoke test has been run, and this file is not a PDF.

## 1. Methods

The question is whether a denoising autoencoder learns a representation that is useful beyond its reconstruction loss. The encoder is trained to map a corrupted 64×64 RGB image to a vector `h`, and a light decoder is trained to reconstruct the clean image. Labels are not used during this training. They are used to build the validation split, the labeled subsets, and the downstream evaluations.

Tiny-ImageNet is loaded from `zh-plus/tiny-imagenet`. The original training split has 100,000 images and 200 classes. A generator with seed 42 holds out 50 images per class for validation (10,000). The remaining 90,000 images are the training pool. The dataset's `valid` split (10,000) is the official test set. It is not used to choose hyperparameters, epochs, or model variants. CIFAR-100 is the transfer dataset. Images are resized to 64×64. Fifty images per class are held out of the official training split, leaving 45,000 training images and 5,000 validation images. The official CIFAR-100 test split has 10,000 images and is kept separate.

The encoder is the two-convolution network from lecture 4, with three input channels:

1. Convolution, 3 to 32 channels, kernel 3, stride 2, padding 1, then ReLU.
2. Convolution, 32 to 64 channels, kernel 3, stride 2, padding 1, then ReLU.
3. Flatten, then a fully connected layer to `h`.

The decoder is the mirror image and ends in a sigmoid, so the reconstruction has shape `[B, 3, 64, 64]` and values in (0, 1). The representation used for every evaluation is `h`, the encoder output before the decoder. It is the bottleneck the later classifiers can read, and the decoder is discarded after pretraining.

A corrupted image is `x_noisy = clamp(x + sigma * noise, 0, 1)`, with `noise` drawn from a standard normal. The training objective is

```text
L_rec = MSE(decoder(encoder(x_noisy)), x_clean)
```

The initial configuration, which is not the selected one, uses sigma 0.10, latent dimension 128, Adam with learning rate 0.001, batch size 128, and 10 epochs. Hyperparameters are chosen by a seeded random search of 12 configurations from sigma in {0.05, 0.10, 0.20}, latent dimension in {64, 128, 256}, and learning rate in {0.0001, 0.0003, 0.001}. Proxy training uses 20 images per class (4,000) for 2 epochs. The validation holdout stays at 10,000 images. The selection metric is validation k-NN top-1 accuracy. Ties are broken by lower validation reconstruction MSE, then by the smaller latent dimension. The selected denoising model is then trained on the full 90,000-image pool.

k-NN uses clean images, an encoder in evaluation mode, cosine similarity of L2-normalized `h`, `k = 20`, and a vote weighted by those cosine values after clamping them at 0. A linear probe freezes the encoder and trains a linear layer with cross-entropy. Fine-tuning trains the encoder and the linear layer with learning rate 0.0001. Both report top-1 and top-5 accuracy. The same class-balanced labeled subsets are used for every model: 1%, 10%, and 100% of the training pool.

The baselines are a vanilla autoencoder, which is the selected model with sigma 0; a randomly initialized encoder at the same width; and a supervised encoder trained from scratch on each labeled fraction. Random fine-tuning on 100% of the labels is not run separately, because it is the supervised run.

## 2. Results

Pending. The smoke test checks the implementation. It does not produce the reconstruction figures, the search scatter, the baseline tables, the label-efficiency curves, the transfer results, or the nearest-neighbor panel.

## 3. Discussion

Pending. In particular, the comparison of reconstruction error with k-NN accuracy, the ranking under k-NN, linear probing, and fine-tuning, the failure cases, and the transfer results all require the recorded runs. No numerical claim is made here.

## References

Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th International Conference on Machine Learning.
