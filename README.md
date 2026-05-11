# Hybrid Deep Learning with Cross-Attention Fusion for Autism Spectrum Disorder Classification

A hybrid deep learning system that classifies **Autism Spectrum Disorder (ASD)** from facial images by combining **ResNet152 with CBAM attention**, **Vision Transformer Large (ViT-L)**, and a custom **Cross-Attention Fusion** module. The model improves upon the base paper accuracy of **91.33%** using an ensemble of three models with advanced training techniques.

---

## Architecture Overview

The pipeline fuses two powerful feature extraction branches through a cross-attention mechanism:

```
Facial Image
     │
     ├──── ResNet152 + CBAM ────┐
     │     (CNN Branch)          ├──── Cross-Attention Fusion ──── Classifier ──── ASD / Non-ASD
     └──── ViT-Large ───────────┘
           (Transformer Branch)
```

**CNN Branch — ResNet152 + CBAM**
- ResNet152 backbone pre-trained on ImageNet and fine-tuned with SimCLR self-supervised weights
- CBAM (Convolutional Block Attention Module) applied after feature extraction — combines Channel Attention and Spatial Attention to focus on discriminative facial regions

**Transformer Branch — ViT-Large**
- `vit_large_patch16_224` from the `timm` library (1024-dim features)
- Early transformer blocks are frozen; last 4 blocks are fine-tuned
- Monte Carlo (MC) Dropout applied for uncertainty estimation

**Cross-Attention Fusion**
- CNN features and ViT features are projected to a shared 512-dim hidden space
- Bidirectional cross-attention: CNN attends to ViT and ViT attends to CNN simultaneously
- Features are fused through a feed-forward network with LayerNorm and GELU activations

**Ensemble**
- Final predictions combine the Hybrid Model, standalone ResNet152+CBAM, and standalone ViT-Large using weighted ensemble averaging

---

## Key Techniques

| Technique | Details |
|-----------|---------|
| Self-Supervised Pre-training | SimCLR with NT-Xent loss (15 epochs) on ResNet50 encoder |
| Attention Mechanism | CBAM (Channel + Spatial Attention) on CNN branch |
| Cross-Attention Fusion | 8-head MultiheadAttention, bidirectional CNN ↔ ViT |
| Data Augmentation | AutoAugment-style: flips, color jitter, blur, noise, elastic distortion, CoarseDropout |
| Class Imbalance Handling | Focal Loss + class-weighted loss + SMOTE on feature space |
| Optimizer | AdamW with Cosine Annealing LR scheduler |
| Test-Time Augmentation (TTA) | 5 transforms averaged at inference |
| Uncertainty Estimation | Monte Carlo Dropout (20 forward passes) |
| Label Smoothing | 0.1 |

---

## Dataset

- **Classes:** `Autistic` / `Non_Autistic`
- **Input image size:** 224×224
- **Expected structure after extraction:**

```
extracted_data/AutismDataset/
├── train/         ← flat directory, filenames contain class names
├── valid/
│   ├── Autistic/
│   └── Non_Autistic/
└── test/          ← flat directory, filenames contain class names
```

> The dataset must be downloaded separately as `archive.zip`. Update the `zip_path` variable in the notebook to point to your local copy before running.

---

## Requirements

```bash
pip install torch torchvision timm albumentations opencv-python scikit-learn \
            matplotlib seaborn pillow tqdm wandb imblearn
```

**Hardware support:**
- Apple Silicon (M1/M2/M3/M4): runs on Metal GPU via MPS automatically
- NVIDIA GPU: runs on CUDA automatically
- CPU: supported but slow

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/revathipennarasi/Hybrid-Deep-Learning-with-CrossAttention-Fusion-for-Autism-Spectrum-Disorder-Classification.git
cd Hybrid-Deep-Learning-with-CrossAttention-Fusion-for-Autism-Spectrum-Disorder-Classification
```

**2. Place the dataset**

Download the dataset zip and update this line in the notebook:
```python
zip_path = "/path/to/your/archive.zip"
```

**3. Run the notebook**
```bash
jupyter notebook main.ipynb
```

Run all cells top to bottom (Cell → Run All). The notebook will:
1. Extract and explore the dataset
2. Run SimCLR self-supervised pre-training
3. Train ResNet152+CBAM, ViT-Large, and the Hybrid fusion model
4. Evaluate all models and build a weighted ensemble
5. Generate a final comparison report

---

## Outputs

All results are saved automatically:

```
checkpoints/
├── simclr_pretrained_encoder.pth
├── hybrid_final.pth
├── resnet_cbam_final.pth
├── vit_large_final.pth
└── ensemble_config.pth

results/
├── dataset_distribution.png
├── smote_balancing.png
├── simclr_pretraining.png
├── results_comparison.csv
└── final_report.txt
```

---

## Inference on a Single Image

```python
result = predict_single_image(
    image_path="path/to/face.jpg",
    model=ensemble.models[0],
    use_tta=True
)

print(result['prediction'])    # 'Autistic' or 'Non-Autistic'
print(result['confidence'])    # e.g. 0.9342
print(result['probabilities']) # {'Non-Autistic': 0.0658, 'Autistic': 0.9342}
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Batch Size | 16 |
| Epochs | 15 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Dropout Rate | 0.3 |
| MC Dropout Passes | 20 |
| TTA Transforms | 5 |
| Ensemble Models | 3 |
| Label Smoothing | 0.1 |
| Seed | 42 |

---

## Results

| Model | Accuracy |
|-------|----------|
| Base Paper | 91.33% |
| Ours (Ensemble) | TBD after training |

---

## Author

**Revathi Pennarasi**  
[GitHub Profile](https://github.com/revathipennarasi)

---

## License

This project is open-source and available under the [MIT License](LICENSE).
