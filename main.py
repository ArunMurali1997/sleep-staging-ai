# =========================================================
# COMPLETE MAIN.PY
# MESA PREPROCESSING + TRAINING + DISTILLATION
# =========================================================

from __future__ import annotations

import os
import gc
import random
import argparse
import hashlib
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

import mne
import seaborn as sns
import matplotlib.pyplot as plt

from ssqueezepy import cwt as ssq_cwt
from train_models import (
    train_pipeline,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path("/content/sleep_model")

EDF_DIR = BASE_DIR / "Data" / "edf"
XML_DIR = BASE_DIR / "Data" / "annot"

PREPROCESSED_DIR = BASE_DIR / "preprocessed"

OUTPUT_DIR = BASE_DIR / "output"

PREPROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# PREPROCESSING CONFIG
# =========================================================

STAGE_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 4,
}

FS = 128
NV = 8

KEEP_WAKE_EPOCHS = 2

EPOCH_DURATION_SEC = 30

REQUIRED_CHANNELS = [
    "EEG1",
    "EEG2",
    "EEG3",
]


# =========================================================
# TRAINING CONFIG
# =========================================================

BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 30

NUM_CLASSES = 5

T = 4
ALPHA = 0.5


# =========================================================
# RANDOM SEED
# =========================================================

random.seed(42)

np.random.seed(42)

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



# =========================================================
# DATASET
# =========================================================

class SleepDataset(Dataset):

    def __init__(
        self,
        pt_files,
        augment=False,
    ):

        self.samples = []

        self.augment = augment

        for pt_file in pt_files:

            data = torch.load(
                pt_file,
                map_location="cpu",
            )

            eeg_cwt = data["eeg_cwt"]

            labels = data["labels"]

            for i in range(len(labels)):

                self.samples.append((
                    eeg_cwt[i].float(),
                    int(labels[i]),
                ))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        x, y = self.samples[idx]

        if self.augment and random.random() < 0.5:

            shift = random.randint(-8, 8)

            x = torch.roll(
                x,
                shifts=shift,
                dims=2,
            )

            noise = (
                torch.randn_like(x)
                * 0.015
            )

            x = x + noise

        return x, y


# =========================================================
# CNN
# =========================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(
            128,
            NUM_CLASSES,
        )

    def forward(self, x):

        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)


# =========================================================
# VIT
# =========================================================

class EnhancedViT(nn.Module):

    def __init__(
        self,
        dim=256,
        depth=8,
        heads=8,
    ):

        super().__init__()

        self.patch_embed = nn.Conv2d(
            3,
            dim,
            kernel_size=16,
            stride=16,
        )

        self.cls_token = nn.Parameter(
            torch.randn(1,1,dim)
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1,200,dim)
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                batch_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=depth,
            )
        )

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(
            dim,
            NUM_CLASSES,
        )

    def forward(self, x):

        B = x.shape[0]

        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1,2)

        cls = self.cls_token.expand(
            B,
            -1,
            -1,
        )

        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1)]

        x = self.transformer(x)

        x = self.norm(x[:,0])

        return self.head(x)


# =========================================================
# TRAIN
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
):

    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    best_f1 = -1

    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        for x, y in train_loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

        model.eval()

        preds = []

        trues = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)

                out = model(x)

                p = out.argmax(1).cpu().numpy()

                preds.extend(p)

                trues.extend(y.numpy())

        macro_f1 = f1_score(
            trues,
            preds,
            average="macro",
        )

        print(
            f"Epoch {epoch+1} "
            f"| Val F1 {macro_f1:.4f}"
        )

        if macro_f1 > best_f1:

            best_f1 = macro_f1

            best_state = model.state_dict()

    model.load_state_dict(best_state)

    return model


# =========================================================
# EVALUATE
# =========================================================

def evaluate(
    model,
    loader,
    name,
):

    model.eval()

    preds = []

    trues = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            out = model(x)

            p = out.argmax(1).cpu().numpy()

            preds.extend(p)

            trues.extend(y.numpy())

    acc = accuracy_score(
        trues,
        preds,
    )

    f1 = f1_score(
        trues,
        preds,
        average="macro",
    )

    print(f"\n{name}")

    print(
        classification_report(
            trues,
            preds,
            digits=3,
        )
    )

    print(
        f"Accuracy: {acc:.4f}"
    )

    print(
        f"Macro F1: {f1:.4f}"
    )

    return acc, f1


# =========================================================
# DATA LOADERS
# =========================================================

def get_data_loaders():

    pt_files = sorted(
        PREPROCESSED_DIR.glob("*.pt")
    )

    train_files, test_files = train_test_split(
        pt_files,
        test_size=0.2,
        random_state=42,
    )

    train_files, val_files = train_test_split(
        train_files,
        test_size=0.125,
        random_state=42,
    )

    train_dataset = SleepDataset(
        train_files,
        augment=True,
    )

    # val_dataset = SleepDataset(
    #     val_files,
    # )

    test_dataset = SleepDataset(
        test_files,
    )

    labels = []

    for _, y in train_dataset.samples:

        labels.append(y)

    labels = np.array(labels)

    class_counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )

    weights = 1 / np.maximum(
        class_counts,
        1,
    )

    weights = weights / weights.mean()

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    # )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        test_loader,
        class_weights,
    )


# =========================================================
# DISTILLATION
# =========================================================

def train_distillation(
    train_loader,
    test_loader,
    class_weights,
    vit_pth_path,
):

    teacher = EnhancedViT().to(DEVICE)

    teacher.load_state_dict(
        torch.load(
            vit_pth_path,
            map_location=DEVICE,
        )
    )

    teacher.eval()

    student = CNN().to(DEVICE)

    optimizer = optim.AdamW(
        student.parameters(),
        lr=3e-4,
    )

    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    kl_loss = nn.KLDivLoss(
        reduction="batchmean"
    )

    for epoch in range(EPOCHS):

        student.train()

        for x, y in train_loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():

                teacher_logits = teacher(x)

            student_logits = student(x)

            loss_ce = ce_loss(
                student_logits,
                y,
            )

            loss_kd = kl_loss(
                F.log_softmax(
                    student_logits / T,
                    dim=1,
                ),
                F.softmax(
                    teacher_logits / T,
                    dim=1,
                ),
            ) * (T * T)

            loss = (
                ALPHA * loss_ce
                +
                (1 - ALPHA) * loss_kd
            )

            loss.backward()

            optimizer.step()

        print(
            f"Distillation Epoch {epoch+1}"
        )

    evaluate(
        student,
        test_loader,
        "Distilled CNN",
    )

    torch.save(
        student.state_dict(),
        OUTPUT_DIR / "model_cnn_distilled.pth",
    )


# =========================================================
# TRAIN PIPELINE
# =========================================================

def train_pipeline_distillation():


    (
        train_loader,
        test_loader,
        class_weights,
    ) = get_data_loaders()


    print("\nRunning Distillation")

    train_distillation(
        train_loader,
        test_loader,
        class_weights,
        vit_pth_path=OUTPUT_DIR / "model_vit.pth",
    )


# =========================================================
# MAIN
# =========================================================

def main():
    # Train ViT teacher
    train_pipeline()
    train_pipeline_distillation()


if __name__ == "__main__":
    main()