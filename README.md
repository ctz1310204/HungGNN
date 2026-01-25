# HungGNN - Graph Neural Network for Linear Sum Assignment Problems

## 📌 Original Repository & Attribution

This repository is a **modified version** based on the work by **aircarlo**:

- **Original repository**: [aircarlo/GNN_LSAP](https://github.com/aircarlo/GNN_LSAP)
- **Description**: Graph Neural Network approach to solving Linear Sum Assignment Problem using deep learning

**Special thanks to the original author for making the code publicly available.**

---

## 🔧 Modifications in This Version

This version includes the following enhancements:
- ✅ Refactored code structure for better maintainability
- ✅ Added Vietnamese documentation ([GIAI_THICH_TIENG_VIET.md](docs/GIAI_THICH_TIENG_VIET.md))
- ✅ Implemented experiment management with different data distributions
- ✅ Added TensorBoard integration for training visualization
- ✅ Enhanced greedy algorithm with bug fixes and detailed analysis
- ✅ Comprehensive logging utilities
- ✅ Support for various data ranges (uniform, extreme values, normalized)

---

## 📖 How to Use

### Training

```bash
python train_paper.py
```

### Testing

```bash
python test_model.py
```

### Data Generation

Training and validation data are provided in the `data/` folder:
- `train_paper_80k.npy` - 80k training samples
- `val_paper_20k.npy` - 20k validation samples

---

## 🗂️ Repository Structure

See [REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) for detailed explanation of the codebase.

---

## 📄 License

This work follows the MIT License from the original repository. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Special thanks to **aircarlo** for the original GNN implementation that serves as the foundation for this work.
