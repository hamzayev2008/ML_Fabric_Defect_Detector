# Dataset Description

## Project

ML Fabric Defect Detector

The dataset is used to train and evaluate a computer vision model that predicts fabric material and fabric defect type from RGB images.

---

## Dataset Sources

The dataset combines:

- Self-collected fabric images
- Additional images obtained from Hugging Face

The self-collected images were prepared and organized as part of the project.

The external data source is used to increase the variety of images available for model training and evaluation.

The exact source dataset and its license should be referenced in the project repository when applicable.

---

## Data Type

The dataset consists of RGB images.

Images may differ in:

- resolution
- lighting
- background
- fabric appearance
- camera angle
- image quality

Before being passed to the model, images are converted to RGB and transformed into the required input format.

---

## Fabric Classes

The fabric classification task contains 11 classes:

1. Cotton-Poly
2. Linen-Poly
3. Denim-Poly
4. Velveteen-Poly
5. Polyester-Poly
6. Satin-Core
7. Chiffon-Poly
8. Nylon-Core
9. Jacquard-Poly
10. Oxford-Core
11. Polyester (coated)-Core

---

## Defect Classes

The defect classification task contains 11 classes:

1. Normal
2. Skipped stitch
3. Broken stitch
4. Pinched fabric
5. Crooked seam
6. Thread sagging
7. Puckering
8. Stain and damage
9. Needle mark
10. Bobbin thread pulling up
11. Overlapped stitch

---

## Dataset Splits

The dataset is divided into three independent subsets:

```text
Train
Validation
Test
```

# Train

Used to update model parameters during training.

# Validation

Used to monitor model performance during training and select the best checkpoint.

# Test

Used only for final independent evaluation.

The current test set contains:
```bash
2,290 images
```
The current validation set contains:
```bash
2,127 images
```
Data Leakage Prevention

Data leakage can lead to overly optimistic evaluation results.

The project therefore includes utilities for checking possible duplication and similarity between dataset splits.

The checks include:

- duplicate image detection
- cross-split leakage checking
- strict image similarity checking

The strict similarity check compares:
```text
Train ↔ Validation
Train ↔ Test
Validation ↔ Test
```
Perceptual hashing is used to identify potentially similar images.

Image Preprocessing

The general preprocessing pipeline is:

```text
Original Image
      │
      ▼
Convert to RGB
      │
      ▼
Resize to 224 × 224
      │
      ▼
Convert to Tensor
      │
      ▼
Normalize
      │
      ▼
Model Input
```

Training can additionally apply image augmentation.

Validation, testing, and inference use evaluation/inference preprocessing without training augmentation.

Dataset Challenges

The dataset contains natural variation caused by:

- different lighting conditions
- different backgrounds
- different fabric appearances
- different image resolutions
- differences between self-collected and external images
- visual similarity between some defect classes

Some defect categories can therefore be harder to distinguish than others.

Dataset Limitations

The dataset is designed for this project and does not necessarily represent every possible fabric type, manufacturing environment, camera setup, or defect condition.

Therefore, the reported model performance should not be interpreted as a guarantee of performance in a real industrial inspection system.

Additional data from different environments would be useful for future evaluation.
