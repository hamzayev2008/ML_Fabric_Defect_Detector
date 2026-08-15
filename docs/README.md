# Technical Documentation

This directory contains technical documentation for the Fabric Defect Detector project.

---

## 1. System Overview

The project implements a computer vision classification system for fabric inspection.

For every input image, the system predicts two independent labels:

```text
Fabric Material
      +
Fabric Defect
```

The architecture uses a shared pretrained ResNet feature extractor followed by two classification heads.

```text
                         Input Image
                              │
                              ▼
                     Image Preprocessing
                              │
                              ▼
                     Pretrained ResNet
                              │
                       Feature Vector
                         /         \
                        /           \
                       ▼             ▼
                Fabric Head      Defect Head
                    │                 │
                    ▼                 ▼
              11 Fabric Classes  11 Defect Classes
```

---

# 2. Dataset

The dataset is divided into three independent splits:

```text
train
validation
test
```

The purpose of the three-way split is:

* `train` — update model parameters;
* `validation` — monitor training and select the best checkpoint;
* `test` — final independent performance evaluation.

The dataset contains:

```text
11 fabric classes
11 defect classes
```

---

# 3. Fabric Classes

The fabric classification task contains:

```text
Cotton-Poly
Linen-Poly
Denim-Poly
Velveteen-Poly
Polyester-Poly
Satin-Core
Chiffon-Poly
Nylon-Core
Jacquard-Poly
Oxford-Core
Polyester (coated)-Core
```

---

# 4. Defect Classes

The defect classification task contains:

```text
Normal
Skipped stitch
Broken stitch
Pinched fabric
Crooked seam
Thread sagging
Puckering
Stain and damage
Needle mark
Bobbin thread pulling up
Overlapped stitch
```

---

# 5. Data Leakage Prevention

Dataset leakage is an important part of the project.

If the same or nearly identical image appears in both training and evaluation data, the reported evaluation accuracy can become misleading.

The project therefore includes several checking utilities.

### Duplicate checking

```text
duplication_check.py
```

Checks for duplicate images.

### Leakage checking

```text
check_leakage.py
```

Checks possible overlap between dataset splits.

### Strict similarity checking

```text
check_strict_similarity.py
```

Uses perceptual hashing to compare:

```text
train ↔ validation
train ↔ test
validation ↔ test
```

Very small perceptual hash distances are treated as suspicious.

---

# 6. Image Preprocessing

Images are converted to RGB and transformed before being passed to the network.

The general inference pipeline is:

```text
Original Image
      │
      ▼
Load Image
      │
      ▼
Resize
      │
      ▼
ToTensor
      │
      ▼
Normalize
      │
      ▼
Model Input
```

The configured image size is:

```text
224 × 224
```

Training can additionally use augmentation.

Validation, testing, and inference do not use training augmentation.

---

# 7. Transfer Learning

The project uses pretrained ResNet architectures from Torchvision.

Available models:

```text
ResNet18
ResNet50
```

The pretrained convolutional backbone is used as a feature extractor.

The original classification layer is replaced.

Instead of a single classification output, the project uses two heads:

```text
Feature Vector
     │
     ├──────────────► Fabric Classifier
     │                     │
     │                     ▼
     │                 11 classes
     │
     └──────────────► Defect Classifier
                           │
                           ▼
                       11 classes
```

This allows both tasks to be learned from the same visual representation.

---

# 8. Loss Function

Two CrossEntropyLoss values are calculated:

```text
Fabric Loss
Defect Loss
```

The training objective is:

```text
Total Loss = Fabric Loss + Defect Loss
```

Both classification tasks therefore contribute to the optimization of the shared ResNet backbone.

---

# 9. Optimization

The project uses the Adam optimizer.

Current learning rate:

```text
0.0001
```

Current batch size:

```text
16
```

Maximum number of epochs:

```text
50
```

Early stopping patience:

```text
5 epochs
```

---

# 10. Model Selection

After each training epoch, the model is evaluated on the validation dataset.

Two accuracies are calculated:

```text
Fabric Accuracy
Defect Accuracy
```

The model-selection metric is:

```text
Average Accuracy =
(Fabric Accuracy + Defect Accuracy) / 2
```

When the average validation accuracy improves, the model checkpoint is saved.

Training stops when the validation metric does not improve for the configured number of epochs.

---

# 11. Current Validation Performance

The current ResNet18 checkpoint achieved:

```text
Fabric Accuracy:   100.00%
Defect Accuracy:    99.06%
Average Accuracy:   99.53%
```

Validation set size:

```text
2,127 images
```

---

# 12. Current Test Performance

The current ResNet18 checkpoint achieved on the test dataset:

```text
Fabric Accuracy:   100.00%
Defect Accuracy:    98.12%
Average Accuracy:   99.06%
```

Test set size:

```text
2,290 images
```

The lower defect accuracy compared with validation accuracy is expected because the test set is an independent evaluation set.

---

# 13. Defect Error Analysis

The defect confusion matrix provides information about which defect classes are confused with one another.

The most notable reported test-set difficulty is:

```text
Needle mark
```

Its reported recall is:

```text
0.91
```

This means some true `Needle mark` images were classified as other defect categories.

The confusion matrix can therefore be used to identify classes that may benefit from:

* additional data;
* improved image quality;
* better class separation;
* further model experimentation.

---

# 14. Evaluation Architecture

Evaluation logic is centralized in:

```text
src/evaluation.py
```

Both validation and evaluation can use the same core evaluation function.

This avoids maintaining separate copies of:

```text
accuracy calculation
classification reports
confusion matrix generation
```

The structure is:

```text
validate.py ─────┐
                 │
                 ▼
          evaluation.py
                 ▲
                 │
test.py ─────────┘
```

This separation keeps dataset-specific execution code separate from reusable evaluation logic.

---

# 15. Inference Architecture

Single-image inference is separated from the Streamlit interface.

The inference architecture is:

```text
Uploaded Image
      │
      ▼
image_utils.py
      │
      ▼
predict.py
      │
      ▼
FabricDefectClassifier
      │
      ├──────────────► Fabric probabilities
      │
      └──────────────► Defect probabilities
                         │
                         ▼
                    app.py
```

This means the machine learning logic is not tied directly to the user interface.

---

# 16. Streamlit Application

The application is implemented in:

```text
src/app.py
```

The application provides an interactive interface for single-image prediction.

The expected workflow is:

```text
Select Model
     ↓
Upload Image
     ↓
Analyze Image
     ↓
Fabric Prediction
     +
Defect Prediction
     ↓
Confidence Scores
     ↓
Class Probabilities
```

Run the application from the project root:

```powershell
streamlit run src/app.py
```

---

# 17. Reproducible Workflow

A typical project workflow is:

```text
1. Prepare dataset
        ↓
2. Split dataset
        ↓
3. Check duplicates
        ↓
4. Check cross-split similarity
        ↓
5. Train model
        ↓
6. Validate model
        ↓
7. Select best checkpoint
        ↓
8. Test model
        ↓
9. Analyze errors
        ↓
10. Run single-image inference
        ↓
11. Use Streamlit application
```

---

# 18. Important Commands

### Train ResNet18

```powershell
python src/train.py resnet18
```

### Train ResNet50

```powershell
python src/train.py resnet50
```

### Validate

```powershell
python src/validate.py
```

### Test

```powershell
python src/test.py
```

### Run Streamlit

```powershell
streamlit run src/app.py
```

### Strict similarity check

```powershell
python src/check_strict_similarity.py
```

---

# 19. Design Principles

The project follows several separation-of-responsibility principles.

### Configuration

```text
config.py
```

Stores configuration rather than embedding parameters throughout the project.

### Dataset

```text
dataset.py
dataLoader.py
```

Responsible for loading and preparing data.

### Model

```text
model.py
```

Responsible for neural network architecture.

### Training

```text
train.py
```

Responsible for optimization and checkpoint selection.

### Evaluation

```text
evaluation.py
validate.py
test.py
```

Responsible for measuring model performance.

### Inference

```text
predict.py
```

Responsible for predictions on individual images.

### Interface

```text
app.py
```

Responsible for user interaction.

This structure keeps the machine learning pipeline modular and makes individual components easier to test and modify.
