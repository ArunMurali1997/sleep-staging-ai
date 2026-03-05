FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by scipy/mne
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# RUN pip install --upgrade pip
# RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]