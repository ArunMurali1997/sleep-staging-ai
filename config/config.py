from pathlib import Path
import numpy as np
import torch

# =========================================================
# BASE PATHS
# =========================================================

BASE_DIR = Path("/content/sleep_model")

EDF_DIR = BASE_DIR / "Data" / "edf"
XML_DIR = BASE_DIR / "Data" / "annot"

OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "cwt_cache"

META_FILE = OUTPUT_DIR / "metadata.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# SIGNAL PARAMETERS
# =========================================================

TARGET_FS = 128
EPOCH_SEC = 30
SAMPLES_PER_EPOCH = TARGET_FS * EPOCH_SEC

LOWCUT = 1.0
HIGHCUT = 32.0

IMG_SIZE = (96, 96)
CHANNEL_NAMES = ["EEG1", "EEG2", "EEG3"]

SCALES = np.arange(1, 32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30


# =========================================================
# TRAINING CONFIG
# =========================================================

T = 4
ALPHA = 0.5