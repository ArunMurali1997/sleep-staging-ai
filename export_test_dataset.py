import random
from pathlib import Path

import pandas as pd
import torch


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path("/content/sleep_model")

OUTPUT_DIR = BASE_DIR / "output"

CACHE_DIR = OUTPUT_DIR / "cwt_cache"

META_FILE = OUTPUT_DIR / "metadata.csv"

EXPORT_DIR = BASE_DIR / "share_package"

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_SUBJECTS_TO_EXPORT = 5


# =========================================================
# LOAD METADATA
# =========================================================

df = pd.read_csv(META_FILE)

print(f"Total samples: {len(df)}")

subjects = sorted(df["subject"].unique())

print(f"Subjects found: {len(subjects)}")

random.seed(42)

random.shuffle(subjects)

subjects = subjects[:NUM_SUBJECTS_TO_EXPORT]


# =========================================================
# EXPORT SUBJECT FILES
# =========================================================

for subject in subjects:

    print(f"\nExporting {subject}")

    subject_df = df[
        df["subject"] == subject
    ].sort_values("global_epoch")

    eeg_list = []

    label_list = []

    for _, row in subject_df.iterrows():

        tensor = torch.load(
            CACHE_DIR / row["filename"],
            map_location="cpu"
        )

        eeg_list.append(
            tensor.unsqueeze(0)
        )

        label_list.append(
            int(row["label"])
        )

    eeg_cwt = torch.cat(
        eeg_list,
        dim=0
    )

    labels = torch.tensor(
        label_list,
        dtype=torch.long
    )

    save_path = (
        EXPORT_DIR /
        f"{subject}.pt"
    )

    torch.save(
        {
            "eeg_cwt": eeg_cwt.half(),
            "labels": labels,
        },
        save_path,
    )

    print(
        f"Saved {save_path.name}"
    )

    print(
        f"Shape: {tuple(eeg_cwt.shape)}"
    )


print("\nExport complete")