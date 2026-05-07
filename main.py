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
# PREPROCESSING FUNCTIONS
# EXACT SAME AS MESA FILE
# =========================================================

def remap_labels(labels: list) -> list:

    remapped = []

    unexpected = set()

    for lbl in labels:

        if lbl in STAGE_REMAP:

            remapped.append(STAGE_REMAP[lbl])

        else:

            remapped.append(-1)

            unexpected.add(lbl)

    if unexpected:

        print(
            f"[WARN] Unexpected labels: {unexpected}"
        )

    return remapped


def compute_and_save_cwt(
    epochs_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    output_path: str,
    fs: int = FS,
    nv: int = NV,
):

    signals = epochs_tensor.float()

    num_epochs = signals.shape[0]

    eeg_cwt_epochs = []

    for ep in tqdm(
        range(num_epochs),
        desc="CWT epochs",
        leave=False,
    ):

        epoch = signals[ep]

        cwt_channels = []

        for ch in epoch:

            sig = ch.numpy()

            Wx, _ = ssq_cwt(
                sig,
                fs=fs,
                nv=nv,
            )

            cwt_channels.append(
                torch.tensor(
                    np.abs(Wx),
                    dtype=torch.float32,
                )
            )

        eeg_cwt = torch.stack(cwt_channels)

        eeg_cwt_epochs.append(eeg_cwt)

    eeg_cwt_tensor = torch.stack(
        eeg_cwt_epochs
    )

    eeg_cwt_tensor = eeg_cwt_tensor.half()

    eeg_cwt_ds = eeg_cwt_tensor[:, :, :, ::4]

    torch.save({

        "eeg_cwt": eeg_cwt_ds,

        "labels": labels_tensor,

    }, output_path)

    print(
        f"Saved {output_path}"
    )


def find_annotation_file(
    edf_path: Path,
    annotation_dir: Path,
) -> Optional[Path]:

    base_name = edf_path.stem

    if "mesa-sleep-" in base_name:

        nsrrid = base_name.replace(
            "mesa-sleep-",
            ""
        )

        xml_filename = f"{nsrrid}-nsrr.xml"

        xml_path = annotation_dir / xml_filename

        if xml_path.exists():
            return xml_path

    for f in annotation_dir.glob("*.xml"):

        if base_name[:17] in f.name:
            return f

    return None


def parse_first_stage_from_xml(
    xml_path: Path
):

    try:

        tree = ET.parse(str(xml_path))

        root = tree.getroot()

        for event in root.findall(".//ScoredEvent"):

            start_el = event.find("Start")

            duration_el = event.find("Duration")

            type_el = event.find("EventType")

            concept_el = event.find("EventConcept")

            if None in (
                start_el,
                duration_el,
                type_el,
                concept_el,
            ):
                continue

            if "Stages|Stages" not in (
                type_el.text or ""
            ):
                continue

            return (
                float(start_el.text),
                float(duration_el.text),
                concept_el.text or "",
            )

    except Exception as e:

        print(e)

    return None


def parse_sleep_stages(xml_path: Path):

    annotations = []

    try:

        tree = ET.parse(str(xml_path))

        root = tree.getroot()

        for event in root.findall(".//ScoredEvent"):

            start_el = event.find("Start")

            duration_el = event.find("Duration")

            type_el = event.find("EventType")

            concept_el = event.find("EventConcept")

            if None in (
                start_el,
                duration_el,
                type_el,
                concept_el,
            ):
                continue

            if "Stages|Stages" not in (
                type_el.text or ""
            ):
                continue

            annotations.append({

                "start":
                    float(start_el.text),

                "duration":
                    float(duration_el.text),

                "stage":
                    (concept_el.text or "").strip(),
            })

    except Exception as e:

        print(e)

    annotations.sort(
        key=lambda x: x["start"]
    )

    return annotations


def adjust_annotations_for_clip(
    annotations,
    clip_t0,
):

    adjusted = []

    for item in annotations:

        start = item["start"]

        duration = item["duration"]

        stage = item["stage"]

        end = start + duration

        if end <= clip_t0:
            continue

        new_start = max(
            start,
            clip_t0,
        )

        new_duration = end - new_start

        if new_duration <= 0:
            continue

        adjusted.append((
            new_start - clip_t0,
            new_duration,
            stage,
        ))

    return adjusted


def expand_annotations_to_epochs(
    annotations,
    epoch_duration=30,
):

    epoch_labels = []

    for start, duration, stage in annotations:

        num_epochs = int(
            duration // epoch_duration
        )

        epoch_labels.extend(
            [stage] * num_epochs
        )

    return np.array(epoch_labels)


def extract_and_preprocess_signal(
    edf_path: Path,
    xml_path: Optional[Path],
    target_eeg_fs: float = 128.0,
    normalization: str = "zscore",
    keep_wake_epochs: int = KEEP_WAKE_EPOCHS,
    epoch_duration: int = EPOCH_DURATION_SEC,
):

    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=True,
        verbose=False,
    )

    clip_t0 = 0.0

    if xml_path is not None and xml_path.exists():

        first_stage = parse_first_stage_from_xml(
            xml_path
        )

        if first_stage is not None:

            start, duration, stage = first_stage

            if "wake" in stage.lower():

                keep_seconds = (
                    keep_wake_epochs
                    * epoch_duration
                )

                if duration > keep_seconds:

                    clip_t0 = (
                        start
                        +
                        (duration - keep_seconds)
                    )

                    try:

                        raw.crop(
                            tmin=clip_t0,
                            tmax=None,
                        )

                    except Exception as e:

                        print(e)

                        clip_t0 = 0.0

    channels_read = [

        ch for ch in REQUIRED_CHANNELS

        if ch in raw.ch_names
    ]

    if not channels_read:

        raise RuntimeError(
            f"No EEG channels found"
        )

    channel_dict = {}

    for ch in channels_read:

        ch_raw = raw.copy().pick([ch])

        ch_raw.filter(0.3, 40)

        ch_raw.resample(target_eeg_fs)

        data = ch_raw.get_data()[0]

        if normalization == "zscore":

            std = np.std(data)

            if std > 0:

                data = (
                    data - np.mean(data)
                ) / std

        channel_dict[ch] = data

    for ch in REQUIRED_CHANNELS:

        if ch not in channel_dict:

            channel_dict[ch] = None

    lengths = [

        len(v)

        for v in channel_dict.values()

        if v is not None
    ]

    target_length = min(lengths)

    for ch in channel_dict:

        v = channel_dict[ch]

        if v is None:

            channel_dict[ch] = np.zeros(
                target_length
            )

        elif len(v) > target_length:

            channel_dict[ch] = v[:target_length]

        elif len(v) < target_length:

            channel_dict[ch] = np.pad(
                v,
                (0, target_length - len(v)),
            )

    stacked = np.stack([
        channel_dict[ch]
        for ch in REQUIRED_CHANNELS
    ])

    return stacked, clip_t0


def segment_signal_and_labels(
    stacked,
    sfreq,
    xml_path,
    clip_t0=0.0,
    epoch_duration=30,
):

    annotations = parse_sleep_stages(
        xml_path
    )

    annotations = adjust_annotations_for_clip(
        annotations,
        clip_t0,
    )

    epoch_labels = expand_annotations_to_epochs(
        annotations,
        epoch_duration,
    )

    C, T = stacked.shape

    epoch_samples = int(
        epoch_duration * sfreq
    )

    signal_epochs = T // epoch_samples

    stacked = stacked[
        :,
        :signal_epochs * epoch_samples
    ]

    signal_epochs_data = stacked.reshape(
        C,
        signal_epochs,
        epoch_samples,
    )

    signal_epochs_data = np.transpose(
        signal_epochs_data,
        (1, 0, 2),
    )

    epoch_labels_nums = [

        int(item.split("|")[-1])

        for item in epoch_labels
    ]

    epoch_labels_nums = remap_labels(
        epoch_labels_nums
    )

    final_epochs = min(
        len(signal_epochs_data),
        len(epoch_labels_nums),
    )

    signal_epochs_data = signal_epochs_data[
        :final_epochs
    ]

    epoch_labels_nums = epoch_labels_nums[
        :final_epochs
    ]

    valid_mask = [

        i

        for i, lbl in enumerate(epoch_labels_nums)

        if lbl != -1
    ]

    signal_epochs_data = signal_epochs_data[
        valid_mask
    ]

    epoch_labels_nums = [
        epoch_labels_nums[i]
        for i in valid_mask
    ]

    epochs_tensor = torch.tensor(
        signal_epochs_data,
        dtype=torch.float32,
    )

    labels_tensor = torch.tensor(
        epoch_labels_nums,
        dtype=torch.long,
    )

    return epochs_tensor, labels_tensor


# =========================================================
# PREPROCESS
# =========================================================

def preprocess():

    existing = list(
        PREPROCESSED_DIR.glob("*.pt")
    )

    if len(existing) > 0:

        print(
            f"Using existing preprocessing "
            f"({len(existing)} files)"
        )

        return

    edf_files = sorted(
        EDF_DIR.glob("*.edf")
    )

    print(
        f"Found {len(edf_files)} EDF files"
    )

    for edf_path in tqdm(
        edf_files,
        desc="Files",
    ):

        save_path = (
            PREPROCESSED_DIR
            /
            f"{edf_path.stem}.pt"
        )

        xml_path = find_annotation_file(
            edf_path,
            XML_DIR,
        )

        if xml_path is None:

            print(
                f"No XML for {edf_path.name}"
            )

            continue

        try:

            stacked, clip_t0 = (
                extract_and_preprocess_signal(
                    edf_path=edf_path,
                    xml_path=xml_path,
                )
            )

            epochs_tensor, labels_tensor = (
                segment_signal_and_labels(
                    stacked,
                    FS,
                    xml_path,
                    clip_t0=clip_t0,
                )
            )

            compute_and_save_cwt(
                epochs_tensor=epochs_tensor,
                labels_tensor=labels_tensor,
                output_path=str(save_path),
            )

        except Exception as e:

            print(e)

            continue

    print("Preprocessing complete")


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

    val_dataset = SleepDataset(
        val_files,
    )

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        val_loader,
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
):

    teacher = EnhancedViT().to(DEVICE)

    teacher.load_state_dict(
        torch.load(
            OUTPUT_DIR / "model_vit.pth",
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

def train_pipeline():

    preprocess()

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
    ) = get_data_loaders()

    print("\nTraining ViT")

    vit = EnhancedViT()

    vit = train_model(
        vit,
        train_loader,
        val_loader,
        class_weights,
    )

    evaluate(
        vit,
        test_loader,
        "ViT",
    )

    torch.save(
        vit.state_dict(),
        OUTPUT_DIR / "model_vit.pth",
    )

    print("\nTraining CNN")

    cnn = CNN()

    cnn = train_model(
        cnn,
        train_loader,
        val_loader,
        class_weights,
    )

    evaluate(
        cnn,
        test_loader,
        "CNN",
    )

    torch.save(
        cnn.state_dict(),
        OUTPUT_DIR / "model_cnn.pth",
    )

    print("\nRunning Distillation")

    train_distillation(
        train_loader,
        test_loader,
        class_weights,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    train_pipeline()


if __name__ == "__main__":
    main()