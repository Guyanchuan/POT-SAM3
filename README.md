# POTSAM3 (Anonymous Release)

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- CUDA 12.6 (if using the official PyTorch cu126 wheels)

## Installation

### 1. Create Conda environment

```bash
conda create -n potsam3 python=3.12
conda deactivate
conda activate potsam3
```

### 2. Install PyTorch with CUDA support

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Clone this repository

```bash
git clone https://github.com/Guyanchuan/POT-SAM3.git
cd POT-SAM3
```

### 4. Clone and install SAM3

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"

cd ..
```

Download the SAM3 checkpoint from the Hugging Face model page (facebook/sam3) and place it at checkpoints/sam3.pt.

### 5. Install POTSAM3 extra dependencies (excluding SAM3/PyTorch)

```bash
pip install -r requirements-extra.txt
```

### 6. Run with one command

```bash
python train.py
python track.py
```
