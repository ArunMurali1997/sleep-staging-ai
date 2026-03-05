# Sleep Stage Classification using Deep Learning

## Overview

This project implements an **automatic sleep stage classification system** using EEG signals from the **MESA Sleep Dataset**.
The pipeline processes raw EEG recordings and classifies sleep into **five stages** using deep learning models.

The system compares two architectures:

* Convolutional Neural Network (CNN)
* Vision Transformer (ViT)

The EEG signals are converted into **Continuous Wavelet Transform (CWT) spectrograms**, which are used as input images for the models.

---

# Sleep Stages

The model predicts the following sleep stages:

| Label | Stage |
| ----- | ----- |
| 0     | Wake  |
| 1     | N1    |
| 2     | N2    |
| 3     | N3    |
| 4     | REM   |

Each EEG signal is segmented into **30-second epochs**.

---

# Project Pipeline

## 1. Data Loading

* EEG recordings are read from **EDF files**
* Sleep stage annotations are parsed from **XML files**

## 2. Signal Preprocessing

The EEG signal undergoes:

* Resampling to **128 Hz**
* **Bandpass filtering (1–32 Hz)**
* Segmentation into **30-second epochs**

## 3. Feature Extraction

Each epoch is converted into a **CWT spectrogram** using the **Morlet wavelet**.

Output tensor size:

```
3 × 96 × 96
```

These spectrogram images are used as model input.

---

# Dataset Distribution

| Stage | Samples |
| ----- | ------- |
| Wake  | 3453    |
| N1    | 758     |
| N2    | 3300    |
| N3    | 694     |
| REM   | 1002    |

Total epochs used for training:

```
9207
```

Dataset split:

| Split      | Size |
| ---------- | ---- |
| Train      | 6444 |
| Validation | 921  |
| Test       | 1842 |

---

# Model Architectures

## CNN Architecture

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
| Transformer Depth   | 8     |
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
| Loss Function | CrossEntropy with class weights |

Early stopping is applied using **validation macro F1 score**.

---

# Results

## Vision Transformer (ViT)

Accuracy

```
71.0%
```

Macro F1 Score

```
0.622
```

| Stage | Precision | Recall | F1    |
| ----- | --------- | ------ | ----- |
| Wake  | 0.944     | 0.899  | 0.921 |
| N1    | 0.319     | 0.487  | 0.385 |
| N2    | 0.797     | 0.602  | 0.686 |
| N3    | 0.625     | 0.576  | 0.599 |
| REM   | 0.417     | 0.680  | 0.517 |

---

## CNN

Accuracy

```
64.5%
```

Macro F1 Score

```
0.582
```

| Stage | Precision | Recall | F1    |
| ----- | --------- | ------ | ----- |
| Wake  | 0.920     | 0.883  | 0.901 |
| N1    | 0.292     | 0.612  | 0.396 |
| N2    | 0.830     | 0.400  | 0.540 |
| N3    | 0.594     | 0.705  | 0.645 |
| REM   | 0.328     | 0.620  | 0.429 |

---

# Model Comparison

| Model | Accuracy | Macro F1 |
| ----- | -------- | -------- |
| ViT   | **71%**  | **0.62** |
| CNN   | 64.5%    | 0.58     |

The **Vision Transformer performs better** than the CNN model for sleep stage classification.

---

# Project Structure

```
sleep_model/
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

Clone the repository

```
git clone https://github.com/yourusername/sleep-stage-classification.git
cd sleep-stage-classification
```

Upgrade pip

```
pip install --upgrade pip
```

Install dependencies using the requirements file

```
RUN pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

---

# Running the Project

Run the main script:

```
python main.py
```

The pipeline will automatically:

1. Preprocess EDF files
2. Generate CWT spectrograms
3. Train ViT model
4. Train CNN model
5. Evaluate performance
6. Generate plots

---

# Hardware

Recommended hardware:

```
GPU: NVIDIA Tesla T4 or better
RAM: 12GB+
```

Training on CPU is possible but significantly slower.

---

# Applications

This system can be applied to:

* Automated sleep scoring
* Sleep disorder diagnosis
* Clinical sleep monitoring
* Wearable sleep tracking devices
* AI-assisted sleep analysis

---

# Author

Arun Murali

Machine Learning Project — Sleep Stage Classification using Deep Learning
