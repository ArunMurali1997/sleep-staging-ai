import os
import gc
import random
import hashlib
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
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
from sklearn.utils import shuffle
import mne
import pywt
import seaborn as sns
import matplotlib.pyplot as plt


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

LOWCUT = 0.3
HIGHCUT = 40.0

IMG_SIZE = (96, 96)
CHANNEL_NAMES = ["EEG1", "EEG2", "EEG3"]

SCALES = np.arange(1, 32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEEP_WAKE_EPOCHS = 2

STAGE_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 4,
}


# =========================================================
# TRAINING CONFIG
# =========================================================

BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30

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
# UTILS
# =========================================================

def tensor_hash(t):
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()


# =========================================================
# MESA PREPROCESSING
# =========================================================


def remap_labels(labels):
    remapped = []

    for lbl in labels:
        if lbl in STAGE_REMAP:
            remapped.append(STAGE_REMAP[lbl])
        else:
            remapped.append(-1)

    return remapped


def parse_first_stage_from_xml(xml_path):
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for event in root.findall(".//ScoredEvent"):

            start_el = event.find("Start")
            duration_el = event.find("Duration")
            type_el = event.find("EventType")
            concept_el = event.find("EventConcept")

            if None in (start_el, duration_el, type_el, concept_el):
                continue

            if "Stages|Stages" not in (type_el.text or ""):
                continue

            return (
                float(start_el.text),
                float(duration_el.text),
                concept_el.text or "",
            )

    except Exception as e:
        print(f"XML parse error: {e}")

    return None


def parse_sleep_stages(xml_path):

    annotations = []

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for event in root.findall(".//ScoredEvent"):

            start_el = event.find("Start")
            duration_el = event.find("Duration")
            type_el = event.find("EventType")
            concept_el = event.find("EventConcept")

            if None in (start_el, duration_el, type_el, concept_el):
                continue

            if "Stages|Stages" not in (type_el.text or ""):
                continue

            annotations.append({
                "start": float(start_el.text),
                "duration": float(duration_el.text),
                "stage": (concept_el.text or "").strip(),
            })

    except Exception as e:
        print(f"XML parse error: {e}")

    annotations.sort(key=lambda x: x["start"])

    return annotations



def adjust_annotations_for_clip(annotations, clip_t0):

    adjusted = []

    for item in annotations:

        start = item["start"]
        duration = item["duration"]
        stage = item["stage"]

        end = start + duration

        if end <= clip_t0:
            continue

        new_start = max(start, clip_t0)
        new_duration = end - new_start

        if new_duration <= 0:
            continue

        adjusted.append((
            new_start - clip_t0,
            new_duration,
            stage,
        ))

    return adjusted



def expand_annotations_to_epochs(annotations):

    epoch_labels = []

    for start, duration, stage in annotations:

        num_epochs = int(duration // EPOCH_SEC)

        epoch_labels.extend([stage] * num_epochs)

    return np.array(epoch_labels)


def compute_cwt(epoch):

    stack = []

    for ch in epoch:

        sig = ch.astype(np.float32)

        sig_std = np.std(sig)

        if sig_std > 0:
            sig = (sig - np.mean(sig)) / sig_std

        coef, _ = pywt.cwt(
            sig,
            SCALES,
            "morl",
            1.0 / TARGET_FS,
        )

        coef = np.abs(coef)

        coef_std = np.std(coef)

        if coef_std > 0:
            coef = (coef - np.mean(coef)) / coef_std

        stack.append(coef)

    img = np.stack(stack, axis=0)

    tensor = torch.from_numpy(img).float()

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=IMG_SIZE,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return tensor


# =========================================================
# PREPROCESS
# =========================================================


def preprocess():

    edfs = sorted(Path(EDF_DIR).glob("*.edf"))

    if not edfs:
        print("No EDF files found")
        return

    metadata = []
    sid = 0
    seen = set()

    class_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    for edf in tqdm(edfs, desc="Processing EDF"):

        base = edf.stem

        xml_file = next(
            (
                x for x in Path(XML_DIR).glob("*.xml")
                if base.lower() in x.stem.lower()
            ),
            None,
        )

        if xml_file is None:
            print(f"Missing XML: {base}")
            continue

        try:
            raw = mne.io.read_raw_edf(
                str(edf),
                preload=True,
                verbose=False,
            )

        except Exception as e:
            print(f"EDF read error {base}: {e}")
            continue

        # =====================================================
        # INITIAL WAKE CLIPPING
        # =====================================================

        clip_t0 = 0.0

        first_stage = parse_first_stage_from_xml(xml_file)

        if first_stage is not None:

            start, duration, stage = first_stage

            if "wake" in stage.lower():

                keep_seconds = KEEP_WAKE_EPOCHS * EPOCH_SEC

                if duration > keep_seconds:

                    clip_t0 = start + (duration - keep_seconds)

                    try:
                        raw.crop(tmin=clip_t0, tmax=None)

                    except Exception as e:
                        print(f"Crop failed: {e}")
                        clip_t0 = 0.0

        # =====================================================
        # CHANNEL PROCESSING
        # =====================================================

        channels_read = [
            ch for ch in CHANNEL_NAMES
            if ch in raw.ch_names
        ]

        if not channels_read:
            print(f"No EEG channels: {base}")
            continue

        channel_dict = {}

        for ch in channels_read:

            ch_raw = raw.copy().pick([ch])

            ch_raw.filter(
                LOWCUT,
                HIGHCUT,
                verbose=False,
            )

            ch_raw.resample(
                TARGET_FS,
                verbose=False,
            )

            data = ch_raw.get_data()[0]

            std = np.std(data)

            if std > 0:
                data = (data - np.mean(data)) / std

            channel_dict[ch] = data

        for ch in CHANNEL_NAMES:
            if ch not in channel_dict:
                channel_dict[ch] = None

        lengths = [
            len(v)
            for v in channel_dict.values()
            if v is not None
        ]

        if not lengths:
            continue

        target_length = min(lengths)

        for ch in channel_dict:

            v = channel_dict[ch]

            if v is None:
                channel_dict[ch] = np.zeros(target_length)

            elif len(v) > target_length:
                channel_dict[ch] = v[:target_length]

            elif len(v) < target_length:
                channel_dict[ch] = np.pad(
                    v,
                    (0, target_length - len(v)),
                )

        data = np.stack([
            channel_dict[ch]
            for ch in CHANNEL_NAMES
        ])

        # =====================================================
        # LABEL PROCESSING
        # =====================================================

        annotations = parse_sleep_stages(xml_file)

        annotations = adjust_annotations_for_clip(
            annotations,
            clip_t0,
        )

        epoch_labels = expand_annotations_to_epochs(annotations)

        epoch_labels_nums = [
            int(item.split('|')[-1])
            for item in epoch_labels
        ]

        epoch_labels_nums = remap_labels(epoch_labels_nums)

        total_epochs = data.shape[1] // SAMPLES_PER_EPOCH

        saved = 0

        for epoch in range(total_epochs):

            if epoch >= len(epoch_labels_nums):
                break

            label = epoch_labels_nums[epoch]

            if label < 0:
                continue

            start = epoch * SAMPLES_PER_EPOCH
            stop = start + SAMPLES_PER_EPOCH

            segment = data[:, start:stop]

            if segment.shape[1] != SAMPLES_PER_EPOCH:
                continue

            tensor = compute_cwt(segment)

            h = tensor_hash(tensor)

            if h in seen:
                continue

            seen.add(h)

            fname = f"s_{sid:06d}.pt"

            torch.save(
                tensor.cpu(),
                CACHE_DIR / fname,
            )

            metadata.append(
                f"{fname},{label},{base},{epoch},{sid}"
            )

            class_counts[label] += 1

            sid += 1
            saved += 1

        print(f"Saved {saved} epochs for {base}")

        gc.collect()

    random.shuffle(metadata)

    with open(META_FILE, "w") as f:

        f.write("filename,label,subject,global_epoch,sid\n")

        f.write("\n".join(metadata) + "\n")

    print("Final class distribution:")
    print(class_counts)


# =========================================================
# DATASET
# =========================================================

class SleepDataset(Dataset):

    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        x = torch.load(
            CACHE_DIR / row["filename"],
            map_location="cpu",
        )

        y = int(row["label"])

        if self.augment and random.random() < 0.5:

            shift = random.randint(-8, 8)
            x = torch.roll(x, shifts=shift, dims=2)

            noise = torch.randn_like(x) * 0.015
            x = x + noise

        return x, y


# =========================================================
# CNN MODEL
# =========================================================


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(128, 5)

    def forward(self, x):

        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)


# =========================================================
# VIT MODEL
# =========================================================


class EnhancedViT(nn.Module):

    def __init__(
        self,
        img_size=96,
        patch=8,
        dim=256,
        depth=8,
        heads=8,
    ):
        super().__init__()

        num_patches = (img_size // patch) ** 2

        self.patch_embed = nn.Conv2d(
            3,
            dim,
            patch,
            patch,
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            depth,
        )

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
# TRAINING
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
        weight_decay=1e-4,
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
            f"Epoch {epoch+1} | F1 {macro_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    return model


# =========================================================
# EVALUATION
# =========================================================


def evaluate(model, loader, name):

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

    acc = accuracy_score(trues, preds)

    f1 = f1_score(
        trues,
        preds,
        average="macro",
    )

    cm = confusion_matrix(trues, preds)

    print(f"\n{name}")

    print(classification_report(
        trues,
        preds,
        digits=3,
    ))

    return acc, f1, cm


# =========================================================
# TRAIN PIPELINE
# =========================================================


def train_pipeline():

    if not os.path.exists(META_FILE):
        preprocess()

    df = pd.read_csv(META_FILE)

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .reindex([0,1,2,3,4], fill_value=1)
    )

    weights = 1 / label_counts
    weights = weights / weights.mean()

    class_weights = torch.tensor(
        weights.values,
        dtype=torch.float32,
    )

    df = shuffle(df, random_state=42)

    train_val, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["label"],
        random_state=42,
    )

    train_loader = DataLoader(
        SleepDataset(train, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        SleepDataset(val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        SleepDataset(test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Training ViT")

    vit = EnhancedViT()

    vit = train_model(
        vit,
        train_loader,
        val_loader,
        class_weights,
    )

    evaluate(vit, test_loader, "ViT")

    torch.save(
        vit.state_dict(),
        OUTPUT_DIR / "model_vit.pth",
    )

    print("Training CNN")

    cnn = CNN()

    cnn = train_model(
        cnn,
        train_loader,
        val_loader,
        class_weights,
    )

    evaluate(cnn, test_loader, "CNN")

    torch.save(
        cnn.state_dict(),
        OUTPUT_DIR / "model_cnn.pth",
    )

    return class_weights


# =========================================================
# GET DATA LOADERS
# =========================================================

def get_data_loaders():

    df = pd.read_csv(META_FILE)

    df = shuffle(df, random_state=42)

    train_val, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["label"],
        random_state=42,
    )

    train_loader = DataLoader(
        SleepDataset(train, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        SleepDataset(test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, test_loader


# =========================================================
# KNOWLEDGE DISTILLATION
# =========================================================

def train_distillation(
    train_loader,
    test_loader,
    class_weights,
):

    print("\nKnowledge Distillation")

    teacher = EnhancedViT().to(DEVICE)

    teacher.load_state_dict(
        torch.load(
            OUTPUT_DIR / "model_vit.pth",
            map_location=DEVICE,
        )
    )

    teacher.eval()

    student = CNN().to(DEVICE)

    try:

        student.load_state_dict(
            torch.load(
                OUTPUT_DIR / "model_cnn.pth",
                map_location=DEVICE,
            )
        )

        print("Loaded pretrained CNN")

    except:
        print("Using fresh CNN")

    for p in teacher.parameters():
        p.requires_grad = False

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

        total_loss = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)

            loss_ce = ce_loss(student_logits, y)

            loss_kd = kl_loss(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
            ) * (T * T)

            loss = (
                ALPHA * loss_ce
                + (1 - ALPHA) * loss_kd
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"| Loss {total_loss/len(train_loader):.4f}"
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

    print("Distilled model saved")


# =========================================================
# MAIN
# =========================================================

def main():

    print("Torch Version:", torch.__version__)

    print("CUDA Available:", torch.cuda.is_available())

    print("Device:", DEVICE)

    class_weights = train_pipeline()

    train_loader, test_loader = get_data_loaders()

    train_distillation(
        train_loader,
        test_loader,
        class_weights,
    )


if __name__ == "__main__":
    main()