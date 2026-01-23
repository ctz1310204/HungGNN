# Extreme Normalization Experiment (Jan 22-23, 2026)

## Experiment Configuration

**Date**: January 22-23, 2026  
**Purpose**: Test impact of extreme range normalization on model accuracy

### Data Generation
- **Original Range**: `[-1e10, 1e10]`
- **Normalization**: Divided by `1e10` to get `[-1, 1]`
- **Rationale**: Replicate original DL-based_LAP approach to compare with direct `[-1, 1]` generation

### Training Parameters
- **Problem Size**: 4x4
- **Training Samples**: 100,000
- **Validation Samples**: 1,000
- **Epochs**: 50
- **Batch Size**: 1 (online learning)
- **Optimizer**: SGD
- **Learning Rate**: 0.01
- **Loss Function**: Dual CrossEntropy (row-wise + column-wise)

## Results

### Final Metrics (Epoch 50)
- **Validation Accuracy**: 95.70%
- **Validation Loss**: ~0.0001 (after bug fix)
- **Training Loss**: ~0.25

### Model Checkpoints
- `trained_net_paper_setup_epoch10.pth`
- `trained_net_paper_setup_epoch20.pth`
- `trained_net_paper_setup_epoch30.pth`
- `trained_net_paper_setup_epoch40.pth`
- `trained_net_paper_setup_epoch50.pth`
- `trained_net_paper_setup_final.pth`

## Comparison with Previous Experiments

| Experiment | Data Range | Normalization | Val Accuracy |
|------------|------------|---------------|--------------|
| gnn_4x4_original | `[0, K)` | None | ~95% |
| gnn_4x4_float_neg1_1 | `[-1, 1]` | Direct generation | ~95% |
| **extreme_norm_20260122** | `[-1e10, 1e10]` | `/1e10` | **95.70%** |

## Key Findings

1. **Normalization Impact**: Extreme range normalization does NOT negatively impact accuracy
2. **Consistency**: Results are consistent with direct `[-1, 1]` generation
3. **Bug Discovery**: Found and fixed validation loss calculation bug (was being divided twice)
4. **Loss vs Accuracy**: CrossEntropy loss of ~0.25 can still achieve 95%+ accuracy due to high prediction confidence

## Code Changes

Modified `helper_fn.py::generate_data()`:
```python
# Before
cm = param_dict['K'] * np.random.random_sample((1, param_dict['N'], param_dict['N']))

# After (extreme norm test)
cm = np.random.uniform(-1e10, 1e10, (1, param_dict['N'], param_dict['N']))
cm = cm / 1e10  # Normalize to [-1, 1]
```

## TensorBoard Logs

Log directory: `logs/` (in this directory)

View with:
```bash
cd experiments/extreme_norm_20260122
tensorboard --logdir=logs --port=6008
```
