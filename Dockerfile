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
RUN pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

COPY . .

CMD ["python", "main.py"]