# 🔭 Deep Lensing – GSoC Evaluation Tasks

Multi-class classification of strong gravitational lensing images using deep learning (PyTorch).

## Dataset

Three classes of simulated strong lensing images (150×150, grayscale):

| Class | Description | Train | Val |
|-------|-------------|-------|-----|
| `no` | No substructure | 10,000 | 2,500 |
| `sphere` | Subhalo substructure | 10,000 | 2,500 |
| `vort` | Vortex substructure | 10,000 | 2,500 |

Images are `.npy` arrays, min-max normalized to [0, 1].

## Notebooks

### 1. Common Test I – CNN Classification
**[`common_test_1_cnn.ipynb`](common_test_1_cnn.ipynb)**

ResNet-18 (pretrained on ImageNet) adapted for single-channel lensing images. Includes data augmentation (random flips, rotations), training with AdamW + cosine annealing, and full evaluation with ROC curves, AUC scores, and confusion matrix.

### 2. Specific Test IV – Neural Operators (FNO)
**[`fno_test.ipynb`](fno_test.ipynb)**

Fourier Neural Operator (FNO) classifier that replaces spatial convolutions with spectral convolutions via FFT. Features custom `SpectralConv2d` layers operating in Fourier space with learnable spectral weights. Includes CNN baseline reproduction and side-by-side performance comparison.

**Key idea:** FNO learns global features in frequency space — naturally suited for lensing images where large-scale arc/ring structures are discriminative.

### 3. Specific Test VII – Physics-Guided ML (PINN)
**[`pinn_test.ipynb`](pinn_test.ipynb)**

Physics-Informed Neural Network incorporating the **gravitational lensing equation** (thin-lens approximation):

$$\vec{\beta} = \vec{\theta} - \vec{\alpha}(\vec{\theta})$$

Architecture: dual-branch model with a deflection field estimator (U-Net style) and a ResNet-18 classifier. Physics losses enforce:
- **Convergence**: ∇·α = 2κ (links deflection to mass density)
- **Curl-free**: ∇×α = 0 (deflection derives from a potential)

Includes deflection field visualization and CNN baseline comparison.

## Evaluation Metrics

All notebooks evaluate using:
- **ROC curves** (per-class one-vs-rest + macro-average)
- **AUC scores** (Area Under ROC Curve)
- **Confusion matrices** and classification reports

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Device Configuration

Each notebook auto-detects the best available device:
- **CUDA** (NVIDIA GPU)
- **MPS** (Apple Silicon GPU)
- **CPU** (fallback)

Override manually by setting `DEVICE_OVERRIDE = "cuda"` / `"mps"` / `"cpu"` in the first code cell.

## Project Structure

```
Deep Lensing/
├── README.md
├── requirements.txt
├── common_test_1_cnn.ipynb        # Common Test I
├── fno_test.ipynb                 # Specific Test IV – FNO
├── pinn_test.ipynb                # Specific Test VII – PINN
└── dataset/
    ├── train/
    │   ├── no/       (10,000 .npy)
    │   ├── sphere/   (10,000 .npy)
    │   └── vort/     (10,000 .npy)
    └── val/
        ├── no/       (2,500 .npy)
        ├── sphere/   (2,500 .npy)
        └── vort/     (2,500 .npy)
```

## References

- [Fourier Neural Operator (Li et al., 2020)](https://arxiv.org/abs/2010.08895)
- [Physics-Informed Neural Networks (Raissi et al., 2019)](https://doi.org/10.1016/j.jcp.2018.10.045)
- [Strong Gravitational Lensing](https://en.wikipedia.org/wiki/Strong_gravitational_lensing)
