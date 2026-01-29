# Google Colab Setup Guide for HungGNN

## 🚀 Quick Start on Colab

```python
# 1. Clone repository
!git clone https://github.com/ctz1310204/HungGNN.git
%cd HungGNN

# 2. Install dependencies (PyTorch 2.x pre-installed on Colab)
!pip install torch-geometric scipy pandas tqdm

# 3. (Optional) If torch-geometric fails, install manually:
# !pip install torch torchvision torchaudio
# !pip install torch-geometric

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
pip install -q torch-geometric scipy pandas tqdm && \
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

## 💾 Storage Options - Save Your Models

### Option 1: Google Drive (Simple)
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone into Drive (persistent)
%cd /content/drive/MyDrive
!git clone https://github.com/ctz1310204/HungGNN.git
%cd HungGNN

# Train - models saved to Drive
!python train_paper.py

# Models at: /content/drive/MyDrive/HungGNN/experiments/
```

### Option 2: Push to GitHub (No Drive!)
```python
# Train in fast temp storage
%cd /content
!git clone https://github.com/YOUR_USERNAME/GNN_LSAP.git
%cd GNN_LSAP

!pip install torch-geometric scipy pandas tqdm
!python train_paper.py

# Configure Git
!git config --global user.email "your@email.com"
!git config --global user.name "Your Name"

# Push models to GitHub
!git add experiments/
!git commit -m "Add trained GNN models"
!git push https://YOUR_TOKEN@github.com/YOUR_USERNAME/GNN_LSAP.git main
```

### Option 3: Weights & Biases (Recommended)
```python
!pip install wandb
import wandb
wandb.login()

# Add to train_paper.py:
# wandb.init(project="gnn-lsap")
# wandb.log({"train_acc": acc, "val_acc": val_acc})

# View experiments at wandb.ai
```

### GitHub with Large Files (Git LFS)
```python
# For models > 100MB
!apt-get install git-lfs
!git lfs install

# Track large files
!git lfs track "*.pth"
!git lfs track "experiments/**/*.pth"

!git add .gitattributes
!git add experiments/
!git commit -m "Add trained models with LFS"
!git push
```

### Automated Backup Script
```python
# Save as auto_backup.sh in repo
backup_script = '''
#!/bin/bash
EXP_DIR="experiments/gnn_4x4_original"
git add $EXP_DIR/trained_net_*.pth
git add $EXP_DIR/logs/
git commit -m "Checkpoint: $(date +%Y%m%d_%H%M%S)"
git push
echo "✅ Backed up to GitHub!"
'''

with open('auto_backup.sh', 'w') as f:
    f.write(backup_script)

!chmod +x auto_backup.sh

# Run after training
!./auto_backup.sh
```

### Comparison

| Method | Speed | Storage | Setup | Best For |
|--------|-------|---------|-------|----------|
| **Google Drive** | 🐢 Slow | 15GB | Easy | Personal |
| **GitHub** | 🚀 Fast | 100MB/file | Medium | Code + small models |
| **Git LFS** | 🚀 Fast | 2GB free | Medium | Large models |
| **Wandb** | 🚀 Fast | 100GB | Easy | Experiments |

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
| **torch-geom>=1.13.0 | PyTorch deep learning |
| **torch-geometric** | >=2.3.0 | Graph neural networks |
| **scipy** | >=1.9.0 | Hungarian algorithm |
| **numpy** | >=1.21 | Numerical computing |
| **pandas** | >=1.3 | Data logging |
| **tensorboard** | >=2.8 | Visualization |
| **tqdm** | >=4.62 | Progress bars |
| **matplotlib/seaborn** | >=3.5/0.11 | Plotting (optional)
---

## ⚠️ Common Issues & Solutions

### Issue 1: PyTorch Geometric Installation Failed
```python
# Solution: Install with explicit CUDA version
!pip insta 1: Install latest stable versions (recommended)
!pip install torch torchvision torchaudio
!pip install torch-geometric

# Solution 2: If still fails, install without CUDA extensions
!pip install torch-geometric --no-deps
!pip install torch scipy numpy pandas tqdm

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
!git clone https://gi torchvision torchaudio
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

## 🔄 Resume Training from Checkpoint

### Resume from Specific Epoch
```python
%cd /content/drive/MyDrive/GNN_LSAP

# Resume training from epoch 30
!python train_paper.py --exp-name my_gnn_experiment --resume --resume_epoch 30 --epochs 50

# Output:
# ✓ Loaded checkpoint from epoch 30
# Continuing training: epochs 31 → 50
```

### Resume Latest Training
```python
# Resume from epoch 40, continue to 50
!python train_paper.py --exp-name gnn_4x4_20260129 --resume --resume_epoch 40 --epochs 50
```

### Check Available Checkpoints
```python
# See what checkpoints exist
!ls experiments/my_gnn_experiment/

# Output:
# trained_net_paper_setup_epoch10.pth
# trained_net_paper_setup_epoch20.pth
# trained_net_paper_setup_epoch30.pth
# trained_net_paper_setup_epoch40.pth
# best_model.pth  ← Best validation accuracy
# log_paper_setup.csv
# logs/
```

**💡 How it works:**
- Saves checkpoint every 10 epochs
- Saves **best model** whenever validation accuracy improves
- Use `--resume_epoch` to continue from any checkpoint
- Best model updated automatically throughout training

---

## 🧪 Testing & Evaluation

### Test with Best Model (Recommended)
```python
%cd /content/drive/MyDrive/GNN_LSAP

# Test using BEST model (highest validation accuracy)
!python test_model.py --exp-name my_gnn_experiment --checkpoint best --size 4

# Output:
# ======================================================================
# COMPREHENSIVE TEST - GNN_LSAP
# ======================================================================
# Experiment: my_gnn_experiment
# Model: best_model.pth
# Problem size: 4x4
# Device: cuda
# ======================================================================
# 
# ✓ Model loaded successfully
# 
# Test samples: 10000
# 
# ======================================================================
# TEST RESULTS
# ======================================================================
# Samples tested: 10000
# 
# ACCURACY:
#   Element Acc (Raw argmax):    87.50%
#   Element Acc (After greedy):  92.30%
#   Full Row Acc (Raw):          65.20%
#   Full Row Acc (After greedy): 78.40%
# 
# PERFORMANCE:
#   Avg Loss:            0.234567
#   Forward time:        1.234 ms/sample
#   Greedy time:         0.456 ms/sample
#   Hungarian time:      0.789 ms/sample
# ======================================================================
# 
# GREEDY IMPROVEMENT:
#   Element accuracy: +4.80%
#   Full row accuracy: +13.20%
# ======================================================================
```

### Test with Specific Checkpoint
```python
# Test checkpoint from epoch 30
!python test_model.py --exp-name my_gnn_experiment --checkpoint epoch30 --size 4

# Available checkpoints: best, epoch10, epoch20, epoch30, epoch40, epoch50
```

### Test Different Problem Sizes
```python
# Test 8x8 problem
!python test_model.py --exp-name gnn_8x8_exp --checkpoint best --size 8

# Test 2x2 problem
!python test_model.py --exp-name gnn_2x2_exp --checkpoint best --size 2
```

### Test with Limited Samples (Faster)
```python
# Test on only 1000 samples (for quick validation)
!python test_model.py --exp-name my_gnn_experiment --checkpoint best --size 4 --test-samples 1000

# Useful for quick accuracy check
```

### Generate Test Data Automatically
```python
# If test data doesn't exist, it will be generated automatically
# Default: 10,000 test samples

# Test data saved to: data/test_4x4.npy
```

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
