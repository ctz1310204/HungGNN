# GNN_LSAP Repository Structure

## 📁 Directory Organization

```
GNN_LSAP/
├── experiments/              # Organized experiment results
│   ├── gnn_4x4_original/    # GNN with float [0,1) - ORIGINAL
│   │   ├── models/          # Checkpoints: epoch 10, 20, 30, 40, 50, final
│   │   ├── logs/            # TensorBoard logs
│   │   └── metadata.txt     # Experiment info
│   └── gnn_4x4_float_neg1_1/  # GNN with float [-1,1]
│       ├── models/          # (empty - models deleted)
│       ├── logs/            # TensorBoard logs (preserved)
│       └── metadata.txt     # Experiment info
│
├── data/                    # Training/validation data
│   ├── train_paper_80k.npy
│   └── val_paper_20k.npy
│
├── logs/                    # Active TensorBoard logs
│   └── gnn_lsap_YYYYMMDD_HHMMSS/
│
├── old_logs/                # Archived incomplete training logs
│   ├── gnn_lsap_20260117_123052/  # First attempt (stopped early)
│   └── gnn_lsap_20260117_140503/  # Second attempt (stopped epoch 11)
│
├── utils/                   # Utility modules
│   └── logger.py
│
├── docs/                    # Documentation
│
├── __pycache__/            # Python cache
├── .venv-1/                # Virtual environment
│
├── gnn_unified.py          # Main training script (train/test modes)
├── train_paper.py          # Original paper training script
├── test_model.py           # Testing script
├── main.py                 # Alternative main script
├── helper_fn.py            # Helper functions
├── networks.py             # Model architectures
├── save_gnn_experiment.sh  # Script to save experiments
│
├── trained_net_paper_setup_*.pth  # Current training checkpoints
├── test_float_neg1_1_results.txt  # Test results
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 🎯 Key Directories

### experiments/
**Purpose**: Organized storage for completed experiments
- Each experiment has its own directory
- Contains models, logs, and metadata
- **Never delete** - permanent storage

### data/
**Purpose**: Training and validation datasets
- Generated on-demand
- Can be regenerated if deleted

### logs/
**Purpose**: Active TensorBoard logs
- Current training logs
- Move to experiments/ when done

### old_logs/
**Purpose**: Archive of incomplete/failed training runs
- Kept for reference
- Can be deleted if space needed

## 📝 File Descriptions

### Training Scripts
- `gnn_unified.py` - **Recommended**: Unified train/test with modes
- `train_paper.py` - Original paper setup training
- `main.py` - Alternative training interface

### Testing Scripts
- `test_model.py` - Comprehensive model testing
- `gnn_unified.py --mode test` - Unified test mode

### Utilities
- `helper_fn.py` - Data generation, validation, collision avoidance
- `networks.py` - Model architectures (HGNN)
- `save_gnn_experiment.sh` - Save experiment to organized directory

## 🔧 Workflow

### 1. Training
```bash
python gnn_unified.py --mode train
```

### 2. Save Experiment
```bash
./save_gnn_experiment.sh <experiment_name>
```

### 3. Testing
```bash
python gnn_unified.py --mode test --checkpoint experiments/<name>/models/trained_net_paper_setup_final.pth
```

### 4. TensorBoard
```bash
tensorboard --logdir experiments/<name>/logs --port 6006
```

## ✅ Current Status

### Completed Experiments
1. **gnn_4x4_original** (float [0,1))
   - Models: ✅ epoch 10, 20, 30, 40 (epoch 50 finishing)
   - Logs: ✅ TensorBoard
   - Results: ⏳ Pending final epoch

2. **gnn_4x4_float_neg1_1** (float [-1,1])
   - Models: ❌ Deleted
   - Logs: ✅ TensorBoard (preserved)
   - Results: ✅ Documented (95.61% accuracy)

### Active Files
- Current training checkpoints in root (will be moved to experiments/)
- Active logs in logs/ directory

## 🧹 Cleanup Done
- ✅ Removed temporary log files (training_original.log, nohup.out)
- ✅ Moved incomplete logs to old_logs/
- ✅ Organized experiments into experiments/
- ✅ Created clear directory structure

## 📊 Experiments Summary

| Experiment | Data | Models | Logs | Results |
|------------|------|--------|------|---------|
| gnn_4x4_original | float [0,1) | ✅ 4 checkpoints | ✅ | ⏳ Training |
| gnn_4x4_float_neg1_1 | float [-1,1] | ❌ | ✅ | ✅ 95.61% |

Repository is now organized! 🎉
