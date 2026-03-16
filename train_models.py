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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
import mne
import pywt
from scipy.signal import butter, filtfilt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")


# # BASE_DIR = Path("/content/drive/MyDrive/sleep_model")
# BASE_DIR = Path("/content/sleep_model")
# # UPDATED FOR YOUR FOLDER NAMES (Data/edf)
# EDF_DIR    = BASE_DIR / "Data" / "edf"
# XML_DIR    = BASE_DIR /  "Data" / "annot"
# OUTPUT_DIR = BASE_DIR / "output"
# CACHE_DIR  = OUTPUT_DIR / "cwt_cache"
# META_FILE  = OUTPUT_DIR / "metadata.csv"
# os.makedirs(CACHE_DIR, exist_ok=True)

# TARGET_FS = 128
# EPOCH_SEC = 30
# SAMPLES_PER_EPOCH = TARGET_FS * EPOCH_SEC
# LOWCUT = 1.0
# HIGHCUT = 32.0
# IMG_SIZE = (96, 96)
# SCALES = np.arange(1, 32)
# # MESA often uses these channels. If 0 epochs saved, we'll try others.
# CHANNEL_NAMES = ["EEG1", "EEG2", "EEG3"]
from config.config import (
    EDF_DIR,
    XML_DIR,
    OUTPUT_DIR,
    CACHE_DIR,
    META_FILE,
    TARGET_FS,
    EPOCH_SEC,
    SAMPLES_PER_EPOCH,
    LOWCUT,
    HIGHCUT,
    IMG_SIZE,
    SCALES,
    CHANNEL_NAMES,
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    EPOCHS
)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("EDF DIR:", EDF_DIR)
print("XML DIR:", XML_DIR)
print("EDF FILES:", list(Path(EDF_DIR).glob("*.edf")))


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def bandpass_filter(signal):
    nyq = TARGET_FS / 2.0
    b, a = butter(5, [LOWCUT/nyq, HIGHCUT/nyq], btype="band")
    return filtfilt(b, a, signal.astype(np.float64))

def tensor_hash(t):
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()

def parse_xml(xml_path, total_epochs):
    labels = np.full(total_epochs, -1, dtype=int)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print("XML parse error:", e)
        return labels

    for ev in root.findall(".//ScoredEvent"):
        event_type = (ev.findtext("EventType") or "").upper()
        concept = (ev.findtext("EventConcept") or "").upper()

        # Only consider sleep stages
        if "STAGE" not in concept and "REM" not in concept and "WAKE" not in concept:
            continue

        if "WAKE" in concept:
            stage = 0
        elif "STAGE 1" in concept:
            stage = 1
        elif "STAGE 2" in concept:
            stage = 2
        elif "STAGE 3" in concept or "STAGE 4" in concept:
            stage = 3
        elif "REM" in concept:
            stage = 4
        else:
            continue

        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)

        start_epoch = int(start / EPOCH_SEC)
        num_epochs = int(np.ceil(dur / EPOCH_SEC))

        for i in range(num_epochs):
            idx = start_epoch + i
            if idx < total_epochs:
                labels[idx] = stage

    return labels

def compute_cwt(epoch):
    stack = []
    for ch in epoch:
        sig = bandpass_filter(ch)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        coef, _ = pywt.cwt(sig, SCALES, "morl", 1.0 / TARGET_FS)
        coef = np.abs(coef)
        coef = (coef - coef.mean()) / (coef.std() + 1e-8)
        stack.append(coef)
    img = np.stack(stack, axis=0)
    tensor = torch.from_numpy(img).float()
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=IMG_SIZE,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    return tensor


def preprocess():
    edfs = sorted(Path(EDF_DIR).glob("*.edf"))
    if not edfs:
        print("❌ ERROR: No .edf files found in Data/edf!")
        return

    print(f"📂 Found {len(edfs)} EDF files. Starting conversion...")

    metadata = []
    sid = 0
    seen = set()
    class_counts = {0:0, 1:0, 2:0, 3:0, 4:0}

    for edf in tqdm(edfs, desc="Processing EDF files"):
        base = edf.stem
        xml_file = next((x for x in Path(XML_DIR).glob("*.xml") if base.lower() in x.stem.lower()), None)
        if xml_file is None:
            print(f"XML missing: {base}")
            continue

        try:
            raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
        except Exception as e:
            print(f"Error reading {base}: {e}")
            continue

        if raw.info['sfreq'] != TARGET_FS:
            raw.load_data()
            raw.resample(TARGET_FS, verbose=False)

        picks = mne.pick_channels(raw.ch_names, include=CHANNEL_NAMES)
        if len(picks) < 3:
            picks = mne.pick_types(raw.info, eeg=True)
            if len(picks) >= 3:
                picks = picks[:3]
            else:
                # If named differently, just take the first 3 EEG channels found
                picks = list(range(min(3, len(raw.ch_names))))

        data = raw.get_data(picks=picks)
        total_epochs = data.shape[1] // SAMPLES_PER_EPOCH
        labels = parse_xml(xml_file, total_epochs)

        start_idx = next((i for i, l in enumerate(labels) if l != 0), 0)

        saved = 0
        for epoch in range(start_idx, total_epochs):
            label = labels[epoch]
            if label < 0:
                continue

            # Skip invalid chunks
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
            torch.save(tensor.cpu(), os.path.join(CACHE_DIR, fname))

            metadata.append(f"{fname},{label},{base},{epoch},{sid}")
            class_counts[label] += 1
            sid += 1
            saved += 1

        print(f"Saved: {saved} epochs for {base}")
        gc.collect()

    print("Shuffling and saving metadata...")
    random.shuffle(metadata)
    with open(META_FILE, "w") as f:
        f.write("filename,label,subject,global_epoch,sid\n")
        f.write("\n".join(metadata) + "\n")

    print("\nFinal class distribution:", class_counts)

class SleepDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.load(CACHE_DIR / row["filename"], map_location="cpu")
        y = int(row["label"])

        if self.augment and random.random() < 0.5:
            shift = random.randint(-8, 8)
            x = torch.roll(x, shifts=shift, dims=2)
            noise = torch.randn_like(x) * 0.015
            x = x + noise

        return x, y

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class EnhancedViT(nn.Module):
    def __init__(self, img_size=96, patch=8, dim=256, depth=8, heads=8):
        super().__init__()
        num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=True
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
# MULTITHREAD SETTINGS (COLAB OPTIMIZED)
# =========================================================
import multiprocessing
torch.set_num_threads(multiprocessing.cpu_count())

# =========================================================
# TRAIN FUNCTION
# =========================================================
def train_model(model, train_loader, val_loader, class_weights, epochs=EPOCHS, patience=10):

    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs//2
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    best_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for x,y in train_loader:

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out,y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1)

            optimizer.step()

            total_loss += loss.item()*x.size(0)

        train_loss = total_loss/len(train_loader.dataset)

        model.eval()

        preds=[]
        trues=[]
        val_loss=0

        with torch.no_grad():

            for x,y in val_loader:

                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                out = model(x)

                loss = criterion(out,y)

                val_loss += loss.item()*x.size(0)

                p = out.argmax(1).cpu().numpy()

                preds.extend(p)
                trues.extend(y.cpu().numpy())

        val_loss = val_loss/len(val_loader.dataset)

        macro_f1 = f1_score(trues,preds,average="macro")

        print(
            f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val F1 {macro_f1:.4f}"
        )

        scheduler.step()

        if macro_f1>best_f1:

            best_f1 = macro_f1
            best_state = model.state_dict()

            patience_counter = 0

        else:

            patience_counter+=1

        if patience_counter>=patience:

            print("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    return model


# =========================================================
# EVALUATION FUNCTION
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

    classes = ["Wake", "N1", "N2", "N3", "REM"]
    cm = confusion_matrix(trues, preds)

    # 1. PRINT TEXT MATRIX TO TERMINAL
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Actual_{c:4}" for c in classes], 
        columns=[f"Pred_{c:4}" for c in classes]
    )
    
    print(f"\n" + "="*50)
    print(f" RAW CONFUSION MATRIX: {name}")
    print("="*50)
    print(cm_df)
    print("="*50)

    # 2. PRINT CLASSIFICATION REPORT
    print(f"\n{name} Detailed Metrics:")
    print(classification_report(trues, preds, target_names=classes, digits=3, zero_division=0))

    # 3. GENERATE VISUAL HEATMAP (Optional - keeps it in Colab output)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{name} Visual Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show() 

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    return acc, f1, cm

# =========================================================
# RESULT PLOTTING
# =========================================================
def plot_results(results):

    models=list(results.keys())

    acc=[results[m]["accuracy"] for m in models]

    f1=[results[m]["f1"] for m in models]

    # Accuracy chart
    plt.figure()

    sns.barplot(x=models,y=acc)

    plt.title("Model Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.show()

    # F1 chart
    plt.figure()

    sns.barplot(x=models,y=f1)

    plt.title("Model Macro F1 Comparison")

    plt.ylabel("Macro F1")

    plt.show()

    # Confusion matrices

    classes=["Wake","N1","N2","N3","REM"]

    for model in models:

        plt.figure(figsize=(6,5))

        sns.heatmap(
            results[model]["cm"],
            annot=True,
            fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues"
        )

        plt.title(f"{model} Confusion Matrix")

        plt.ylabel("True")

        plt.xlabel("Predicted")

        plt.show()


# =========================================================
# TRAIN PIPELINE
# =========================================================
def train_pipeline():
    if not os.path.exists(META_FILE):
        print("Starting preprocessing")
        preprocess()

    df=pd.read_csv(META_FILE)

    print("Dataset distribution")

    print(df["label"].value_counts().sort_index())

    label_counts=df["label"].value_counts().sort_index().reindex([0,1,2,3,4],fill_value=1)

    weights=1/label_counts

    weights=weights/weights.mean()

    class_weights=torch.tensor(weights.values,dtype=torch.float32)

    df=shuffle(df,random_state=42)

    train_val,test=train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train,val=train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["label"],
        random_state=42
    )

    train_loader=DataLoader(
        SleepDataset(train,augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader=DataLoader(
        SleepDataset(val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader=DataLoader(
        SleepDataset(test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"\nTrain {len(train)} | Val {len(val)} | Test {len(test)}")

    results={}

    # =====================================================
    # TRAIN VIT FIRST
    # =====================================================

    print("\nTraining ViT")

    vit=EnhancedViT()

    vit=train_model(vit,train_loader,val_loader,class_weights)

    acc,f1,cm=evaluate(vit,test_loader,"ViT")

    torch.save(vit.state_dict(),OUTPUT_DIR/"model_vit.pth")

    results["ViT"]={
        "accuracy":acc,
        "f1":f1,
        "cm":cm
    }

    # =====================================================
    # TRAIN CNN SECOND
    # =====================================================

    print("\nTraining CNN")

    cnn=CNN()

    cnn=train_model(cnn,train_loader,val_loader,class_weights)

    acc,f1,cm=evaluate(cnn,test_loader,"CNN")

    torch.save(cnn.state_dict(),OUTPUT_DIR/"model_cnn.pth")

    results["CNN"]={
        "accuracy":acc,
        "f1":f1,
        "cm":cm
    }

    # =====================================================
    # PLOT RESULTS
    # =====================================================

    plot_results(results)


def get_data_loaders():

    df = pd.read_csv(META_FILE)

    label_counts = df["label"].value_counts().sort_index().reindex([0,1,2,3,4],fill_value=1)

    weights = 1 / label_counts
    weights = weights / weights.mean()

    class_weights = torch.tensor(weights.values, dtype=torch.float32)

    df = shuffle(df, random_state=42)

    train_val, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train, val = train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["label"],
        random_state=42
    )

    train_loader = DataLoader(
        SleepDataset(train, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        SleepDataset(test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, test_loader, class_weights