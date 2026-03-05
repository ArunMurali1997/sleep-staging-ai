# Sleep Stage Classification using Deep Learning

## Overview

This project implements an **automatic sleep stage classification system** using EEG signals from the **MESA Sleep Dataset**. The pipeline converts raw EEG recordings into spectrogram representations using **Continuous Wavelet Transform (CWT)** and trains deep learning models to classify sleep stages.

Two deep learning architectures are implemented and compared:

* **Vision Transformer (ViT)**
* **Convolutional Neural Network (CNN)**

The system predicts **five sleep stages** from EEG recordings.

---

# Sleep Stages

| Label | Stage |
| ----- | ----- |
| 0     | Wake  |
| 1     | N1    |
| 2     | N2    |
| 3     | N3    |
| 4     | REM   |

Each EEG recording is segmented into **30-second epochs** before training.

---

# Project Pipeline

## 1. Data Loading

* EEG recordings are loaded from **EDF files**
* Sleep annotations are parsed from **XML files**

## 2. Signal Processing

The EEG signal is processed using:

* Resampling to **128 Hz**
* **Bandpass filtering (1–32 Hz)**
* Segmentation into **30-second epochs**

## 3. Feature Extraction

Each epoch is transformed into a **CWT spectrogram image** using the Morlet wavelet.

Model input size:

```text
3 × 96 × 96
```

---

# Dataset Distribution

| Stage | Samples |
| ----- | ------- |
| Wake  | 3453    |
| N1    | 758     |
| N2    | 3300    |
| N3    | 694     |
| REM   | 1002    |

Total epochs used:

```
9207
```

Dataset split:

| Split      | Samples |
| ---------- | ------- |
| Train      | 6444    |
| Validation | 921     |
| Test       | 1842    |

---

# Model Architectures

## CNN

Architecture:

```
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
AdaptiveAvgPool
Fully Connected Layer
```

Output: **5 sleep stage classes**

---

## Vision Transformer (ViT)

Configuration:

| Parameter           | Value |
| ------------------- | ----- |
| Patch Size          | 8     |
| Embedding Dimension | 256   |
| Transformer Layers  | 8     |
| Attention Heads     | 8     |

Pipeline:

```
Patch Embedding
→ Positional Encoding
→ Transformer Encoder
→ Classification Head
```

---

# Training Configuration

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Epochs        | 30                              |
| Batch Size    | 32                              |
| Optimizer     | AdamW                           |
| Learning Rate | 3e-4                            |
| Scheduler     | Cosine Annealing                |
| Loss          | CrossEntropy with class weights |

Early stopping is applied using **validation macro F1 score**.

---

# Results

## Vision Transformer

Accuracy

```
71%
```

Macro F1 Score

```
0.62
```

## CNN

Accuracy

```
64.5%
```

Macro F1 Score

```
0.58
```

Vision Transformer achieved **better performance than CNN** for sleep stage classification.

---

# Project Structure

```
sleep-staging-ai/
│
├── Data/
│   ├── edf/
│   └── annot/
│
├── output/
│   ├── cwt_cache/
│   ├── metadata.csv
│   ├── model_vit.pth
│   └── model_cnn.pth
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/ArunMurali1997/sleep-staging-ai.git
cd sleep-staging-ai
```

---

# CPU Installation

Install dependencies for CPU environments:

```
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Run the training script:

```
python main.py
```

CPU training works but will be **significantly slower**.

---

# GPU Installation (Recommended)

Enable GPU support and install CUDA-enabled PyTorch.

Install dependencies:

```
pip install --upgrade pip

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

before running the script move the file to 

```
#In google collab

## mount google drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

## copy file form drive to collab
!mkdir -p /content/sleep_model/Data/annot
!mkdir -p /content/sleep_model/Data/edf
!mkdir -p /content/sleep_model/output

!rsync -av /content/drive/MyDrive/sleep_model/Data/annot/ /content/sleep_model/Data/annot/
!rsync -av /content/drive/MyDrive/sleep_model/Data/edf/ /content/sleep_model/Data/edf/
!rsync -av /content/drive/MyDrive/sleep_model/output/ /content/sleep_model/output/

```

Run the training script:

```
python main.py
```

---

# Verify GPU

After installation you can check GPU availability:

```python
import torch

print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Expected output on GPU:

```
CUDA Available: True
GPU: Tesla T4
```

---

# Hardware Recommendation

| Hardware | Recommendation     |
| -------- | ------------------ |
| CPU      | Supported but slow |
| GPU      | Tesla T4 or better |
| RAM      | 12 GB+             |

---

# Applications

This project can be used for:

* Automated sleep scoring
* Sleep disorder detection
* Clinical sleep monitoring
* AI-assisted sleep analysis
* Wearable sleep tracking systems

---

# Author

Arun Murali

Machine Learning Project — Sleep Stage Classification using Deep Learning
