FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    enchant \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install conda manually (Miniforge)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# Create environment
RUN conda create -n tts-env python=3.9 && conda clean -afy
SHELL ["conda", "run", "-n", "tts-env", "/bin/bash", "-c"]

# Install PyTorch
RUN conda run -n tts-env pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install other pip dependencies
COPY requirements.txt .
RUN conda run -n tts-env pip install -r requirements.txt

# Clone Trainer and TTS repos inside container
WORKDIR /workspace
RUN git clone https://github.com/gokulkarthik/Trainer.git
WORKDIR /workspace/Trainer
RUN conda run -n tts-env pip install -e .[all]

WORKDIR /workspace
RUN git clone https://github.com/gokulkarthik/TTS.git
WORKDIR /workspace/TTS
RUN conda run -n tts-env pip install -e .[all]

# Optional - Jupyter Port
EXPOSE 8888

# Default command
WORKDIR /workspace
CMD ["conda", "run", "-n", "tts-env", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
