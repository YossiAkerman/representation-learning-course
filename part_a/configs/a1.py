# Initial denoising-autoencoder experiment. Another run adds configs/<name>.py.
EXPERIMENT_NAME = "a1"

# Training hyperparameters.
LATENT_DIM = 128  # length of h. Later probes must use this same width.
NOISE_SIGMA = 0.10  # standard deviation of the Gaussian corruption
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
SEED = 42

# Dataset
DATASET_NAME = "zh-plus/tiny-imagenet"
DATA_DIR = "dataset/"  # Hugging Face cache; gitignored
NUM_CLASSES = 200
VAL_PER_CLASS = 50  # held out of the original training split
NUM_WORKERS = 0  # Hugging Face image decoding and macOS process forking do not mix. Raise this on Colab.

# Compute. "auto" uses a GPU when one is present, and the CPU otherwise.
ACCELERATOR = "auto"
DEVICES = 1
PRECISION = "16-mixed"  # mixed precision on GPU; Lightning falls back to bf16 on CPU

# Outputs. Checkpoints and the encoder file are gitignored.
CHECKPOINT_DIR = "checkpoints"
ENCODER_PATH = "tiny_imagenet_encoder.pt"
LOG_DIR = "tb_logs"
FIGURES_DIR = "figures"
