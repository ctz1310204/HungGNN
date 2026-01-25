# Google Colab Setup Guide for HungGNN

## 🚀 Quick Start on Colab

```python
# 1. Clone repository
!git clone https://github.com/ctz1310204/HungGNN.git
%cd HungGNN

# 2. Install PyTorch Geometric (REQUIRED for Colab)
!pip install torch==1.11.0
!pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install torch-geometric

# 3. Install other dependencies
!pip install scipy pandas tqdm tensorboard

# 4. Select GPU Runtime
# Runtime → Change runtime type → Hardware accelerator: GPU → T4 GPU

# 5. Verify GPU
import torch
print(f"✅ GPU Available: {torch.cuda.is_available()}")
print(f"✅ GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 6. Train
!python train_paper.py
```

---

## ⚡ One-Line Installation Script

```python
# Copy-paste all at once
!git clone https://github.com/ctz1310204/HungGNN.git && \
cd HungGNN && \
pip install -q torch==1.11.0 && \
pip install -q torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html && \
pip install -q torch-geometric scipy pandas tqdm tensorboard && \
python train_paper.py
```

---

## 📊 Monitor Training with TensorBoard

```python
# Load TensorBoard extension
%load_ext tensorboard

# View training logs
%tensorboard --logdir logs/

# Or view specific experiment
%tensorboard --logdir experiments/gnn_4x4_original/logs/
```

---

## 💾 Persistent Storage with Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone into Drive (persistent)
%cd /content/drive/MyDrive
!git clone https://github.com/ctz1310204/HungGNN.git
%cd HungGNN

# Install dependencies (still required each session)
!pip install -q torch==1.11.0
!pip install -q torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install -q torch-geometric scipy pandas tqdm tensorboard

# Train - models saved to Drive
!python train_paper.py

# Models will be saved at:
# /content/drive/MyDrive/HungGNN/experiments/
```

---

## 🔧 Training Options

### Train from Scratch
```python
!python train_paper.py
```

### Resume Training
```python
!python train_paper.py --resume --resume_epoch 30
```

### Test Model
```python
!python test_model.py
```

---

## 📦 What Gets Installed?

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | 1.11.0 | PyTorch deep learning |
| **torch-geometric** | 2.0.4 | Graph neural networks |
| **torch-scatter/sparse/cluster** | 2.0.x/0.6.x/1.6.x | PyG dependencies |
| **scipy** | 1.8.1 | Hungarian algorithm |
| **numpy** | >=1.21 | Numerical computing |
| **pandas** | >=1.3 | Data logging |
| **tensorboard** | >=2.8 | Visualization |
| **tqdm** | >=4.62 | Progress bars |

---

## ⚠️ Common Issues & Solutions

### Issue 1: PyTorch Geometric Installation Failed
```python
# Solution: Install with explicit CUDA version
!pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
!pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install torch-geometric
```

### Issue 2: CUDA Out of Memory
```python
# Solution: Reduce batch size or use smaller model
# GNN_LSAP uses online learning (batch_size=1), so this is rare
# If still happens, restart runtime: Runtime → Restart runtime
```

### Issue 3: Module Not Found
```python
# Solution: Make sure you're in the correct directory
%cd /content/HungGNN
!ls  # Should see train_paper.py, networks.py, etc.
```

---

## 🎯 Complete Workflow Template

```python
# === SETUP ===
from google.colab import drive
drive.mount('/content/drive')

# Clone or navigate
%cd /content/drive/MyDrive
!git clone https://github.com/ctz1310204/HungGNN.git 2>/dev/null || %cd HungGNN

# Install
!pip install -q torch==1.11.0
!pip install -q torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install -q torch-geometric scipy pandas tqdm tensorboard

# Verify
import torch
print(f"✅ GPU: {torch.cuda.is_available()}")
print(f"✅ Device: {torch.cuda.get_device_name(0)}")

# === TRAIN ===
!python train_paper.py

# === MONITOR ===
%load_ext tensorboard
%tensorboard --logdir logs/

# === DOWNLOAD RESULTS ===
from google.colab import files
!zip -r results.zip experiments/
files.download('results.zip')
```

---

## 📈 Expected Training Time

| Matrix Size | Samples | Epochs | GPU (T4) | CPU |
|-------------|---------|--------|----------|-----|
| **4x4** | 80k | 50 | ~2-3 hours | ~10-15 hours |
| **8x8** | 1M | 50 | ~20-30 hours | Days |

**💡 Tip:** Colab Free has ~12 hour limit. For larger experiments:
- Use Colab Pro (24 hour limit)
- Or train in multiple sessions with `--resume`

---

## 🔄 Multi-Session Training

```python
# Session 1: Train epochs 1-20
!python train_paper.py  # Will save every 10 epochs

# Session 2 (next day): Resume from epoch 20
!python train_paper.py --resume --resume_epoch 20

# Session 3: Resume from epoch 40
!python train_paper.py --resume --resume_epoch 40
```

---

## 📝 Key Differences from HungCNN

| Feature | HungGNN | HungCNN |
|---------|---------|---------|
| **Framework** | PyTorch + PyG | TensorFlow 1.x |
| **Architecture** | Graph Neural Net | CNN |
| **Batch Size** | 1 (online) | 2048 |
| **Installation** | More complex (PyG) | Simpler |
| **GPU Speed** | Faster per sample | Faster batched |

---

## ✅ Verification Checklist

After installation, verify everything works:

```python
# 1. Check imports
import torch
import torch_geometric
import scipy
import numpy as np
print("✅ All imports successful")

# 2. Check GPU
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

# 3. Check data
import os
print(f"✅ Training data: {os.path.exists('data/train_paper_80k.npy')}")
print(f"✅ Validation data: {os.path.exists('data/val_paper_20k.npy')}")

# 4. Test forward pass
from networks import HGNN
model = HGNN().cuda() if torch.cuda.is_available() else HGNN()
print("✅ Model loaded successfully")
```

---

## 🚀 Ready to Train!

If all checks pass, you're ready to start training:

```python
!python train_paper.py
```

**Expected output:**
```
Epoch 1/50: 100%|██████████| 80000/80000 [25:34<00:00, 52.16it/s]
Train Acc: 0.8523, Val Acc: 0.8456
Saved: trained_net_paper_setup_epoch1.pth
...
```
