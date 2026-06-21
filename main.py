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
    cohen_kappa_score
)

import mne
import seaborn as sns
import matplotlib.pyplot as plt

from ssqueezepy import cwt as ssq_cwt
from thop import profile
from thop import clever_format
import copy
# from train_models import (
#     train_pipeline,
# )


# =========================================================
# CONFIG
# =========================================================


BASE_DIR = Path("/content/sleep_model")

EDF_DIR = BASE_DIR / "Data" / "edf"
XML_DIR = BASE_DIR / "Data" / "annot"
OUTPUT_DIR = BASE_DIR / "output"
PREPROCESSED_DIR = OUTPUT_DIR / "cwt_cache"



# PREPROCESSED_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPORT_DIR = (
    BASE_DIR /
    "share_package"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", DEVICE)
# print("PREPROCESSED_DIR:", PREPROCESSED_DIR)
# print("PT FILES:", len(list(PREPROCESSED_DIR.glob("*.pt"))))

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

ALPHA = 0.7
T = 4

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

    def __init__(self, file_list, augment=False):

        self.augment = augment

        self.index_map = []

        self.cache = {}

        print("Loading dataset...")

        for file_path in file_list:

            data = torch.load(
                file_path,
                map_location="cpu"
            )

            self.cache[file_path] = data

            for i in range(
                len(data["labels"])
            ):
                self.index_map.append(
                    (file_path, i)
                )

        print(
            f"Total epochs: {len(self.index_map)}"
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        file_path, ep_idx = self.index_map[idx]

        data = self.cache[file_path]

        x = data["eeg_cwt"][ep_idx].float()

        y = int(
            data["labels"][ep_idx]
        )

        if self.augment and random.random() < 0.5:

            shift = random.randint(-8, 8)

            x = torch.roll(
                x,
                shifts=shift,
                dims=2
            )

            x = x + (
                torch.randn_like(x)
                * 0.015
            )

        return x, y
# =========================================================
# CNN
# =========================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64,5)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class EnhancedViT(nn.Module):
    def __init__(self, img_size=96, patch=8, dim=384, depth=10, heads=8):
        super().__init__()
        num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)

        # FIX: norm_first=True switches this to a Pre-LN transformer.
        # Pre-LN keeps gradients well-scaled from the very first step, which
        # is what makes a deep (8-layer) transformer trainable from scratch
        # on a small dataset without LR warmup collapsing it onto the
        # majority class. Post-LN (the previous default) is known to need
        # warmup to avoid exactly that failure mode.
        encoder_layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=1024, dropout=0.1,
            activation="gelu", batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(dim, 5)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.dropout(x)
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

        for batch_idx, (x, y) in enumerate(train_loader):

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

    classes = [
        "Wake",
        "N1",
        "N2",
        "N3",
        "REM"
    ]

    # ==========================================
    # CALCULATE METRICS
    # ==========================================

    accuracy = accuracy_score(
        trues,
        preds
    )

    macro_f1 = f1_score(
        trues,
        preds,
        average="macro"
    )

    kappa = cohen_kappa_score(
        trues,
        preds
    )

    cm = confusion_matrix(
        trues,
        preds
    )

    report = classification_report(
        trues,
        preds,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    # ==========================================
    # PRINT FINAL RESULTS
    # ==========================================

    print("\n" + "=" * 70)
    print(f"{name} FINAL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Cohen Kappa  : {kappa:.4f}"
    )

    print(
        f"Macro F1     : {macro_f1:.4f}"
    )

    # ==========================================
    # STAGE-WISE METRICS
    # ==========================================

    print("\n" + "=" * 70)
    print("STAGE-WISE METRICS")
    print("=" * 70)

    for cls in classes:

        precision = report[
            cls
        ]["precision"]

        recall = report[
            cls
        ]["recall"]

        stage_f1 = report[
            cls
        ]["f1-score"]

        print(
            f"{cls:5} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1 Score: {stage_f1:.4f}"
        )

    # ==========================================
    # FULL CLASSIFICATION REPORT
    # ==========================================

    print("\n" + "=" * 70)
    print("FULL CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            trues,
            preds,
            target_names=classes,
            digits=4,
            zero_division=0
        )
    )

    # ==========================================
    # CONFUSION MATRIX TABLE
    # ==========================================

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{c}"
            for c in classes
        ],
        columns=[
            f"Pred_{c}"
            for c in classes
        ]
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm_df)

    # ==========================================
    # VISUAL CONFUSION MATRIX
    # ==========================================

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title(
        f"{name} Confusion Matrix"
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.show()

    return (
        accuracy,
        macro_f1
    )

def print_model_stats(model, name):

    temp_model = copy.deepcopy(
        model
    ).to(DEVICE)

    temp_model.eval()

    dummy_input = torch.randn(
        1, 3, 96, 96
    ).to(DEVICE)

    flops, params = profile(
        temp_model,
        inputs=(dummy_input,),
        verbose=False
    )

    flops, params = clever_format(
        [flops, params],
        "%.3f"
    )

    print("\n" + "=" * 50)
    print(f"{name} Model Statistics")
    print("=" * 50)
    print(f"Parameters: {params}")
    print(f"FLOPs: {flops}")
    print("=" * 50)


# =========================================================
# DATA LOADERS
# =========================================================

def get_data_loaders():

    pt_files = sorted(
        EXPORT_DIR.glob("*.pt")
    )

    print(
        f"Found {len(pt_files)} files"
    )

    train_files, test_files = train_test_split(
        pt_files,
        test_size=0.2,
        random_state=42
    )

    train_dataset = SleepDataset(
        train_files,
        augment=True
    )

    test_dataset = SleepDataset(
        test_files
    )

    all_labels = []

    for file_path in train_files:
        data = torch.load(file_path, map_location="cpu")
        all_labels.extend(data["labels"])

    label_counts = np.bincount(all_labels, minlength=NUM_CLASSES)
    weights = len(all_labels) / (NUM_CLASSES * label_counts)

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    print("\nClass Weights:")
    print(class_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        test_loader,
        class_weights
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



    print("Loading teacher...")
    state_dict = torch.load(
        vit_pth_path,
        map_location=DEVICE
    )
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if "total_ops" not in k
        and "total_params" not in k
    }
    teacher.load_state_dict(
        state_dict
    )

    print_model_stats(
        teacher,
        "Teacher ViT"
    )
    
    print("Teacher loaded successfully")
    

    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    student = CNN().to(DEVICE)

    print_model_stats(
        student,
        "Student CNN"
    )

    optimizer = optim.AdamW(
        student.parameters(),
        lr=3e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    kl_loss = nn.KLDivLoss(
        reduction="batchmean"
    )
    best_f1 = -1
    best_state = None

    for epoch in range(EPOCHS):

        student.train()

        total_loss = 0
        total_ce = 0
        total_kd = 0
        total_samples = 0

        for batch_idx, (x, y) in enumerate(train_loader):

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():

                teacher_logits = teacher(x)

            student_logits = student(x)

            if epoch == 0 and batch_idx == 0:

                print("\n===== DISTILLATION SAMPLE =====")

                print("Ground Truth:")
                print(y[:5].cpu().numpy())

                print("\nTeacher Logits:")
                print(teacher_logits[:2].detach().cpu())

                print("\nTeacher Soft Targets:")
                print(
                    F.softmax(
                        teacher_logits[:2] / T,
                        dim=1
                    ).cpu()
                )

                print("\nStudent Logits:")
                print(student_logits[:2].detach().cpu())

                print("\nStudent Soft Targets:")
                print(
                    F.softmax(
                        student_logits[:2] / T,
                        dim=1
                    ).cpu()
                )

                print("===============================\n")

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
            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_ce += loss_ce.item() * batch_size
            total_kd += loss_kd.item() * batch_size
            total_samples += batch_size

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                1.0
            )

            optimizer.step()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f} CE {loss_ce.item():.4f} KD {loss_kd.item():.4f}"
                )

            avg_loss = total_loss / total_samples
            avg_ce = total_ce / total_samples
            avg_kd = total_kd / total_samples

        student.eval()

        preds = []
        trues = []

        with torch.no_grad():

            for x, y in test_loader:

                x = x.to(DEVICE)

                out = student(x)

                p = out.argmax(1).cpu().numpy()

                preds.extend(p)
                trues.extend(y.numpy())

        acc = accuracy_score(trues, preds)

        kappa = cohen_kappa_score(trues, preds)

        f1 = f1_score(
            trues,
            preds,
            average="macro"
)


        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}"
            f" | LR {current_lr:.6f}"
            f" | Loss {avg_loss:.4f}"
            f" | CE {avg_ce:.4f}"
            f" | KD {avg_kd:.4f}"
            f" | Acc {acc:.4f}"
            f" | Kappa {kappa:.4f}"
            f" | F1 {f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            best_state = {
                k: v.cpu().clone()
                for k, v in student.state_dict().items()
            }
            print("\n" + "="*60)
            print("NEW BEST MODEL")
            print("="*60)
            print(f"Accuracy : {acc:.4f}")
            print(f"Kappa    : {kappa:.4f}")
            print(f"Macro F1 : {best_f1:.4f}")
            print("="*60)





    student.load_state_dict(best_state)

    acc, f1 = evaluate(
        student,
        test_loader,
        "Best Distilled CNN"
    )

    torch.save(
        student.state_dict(),
        OUTPUT_DIR / "model_cnn_distilled.pth"  
    )

    print(
        f"Best Distilled CNN saved with F1={best_f1:.4f}"
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
    # train_pipeline()
    train_pipeline_distillation()


if __name__ == "__main__":
    main()