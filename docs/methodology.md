# Methodology

## Overview

The project follows a supervised computer vision workflow for classifying fabric material and fabric defects.

The main stages are:

```text
Dataset Preparation
        ↓
Data Splitting
        ↓
Leakage Checking
        ↓
Image Preprocessing
        ↓
Transfer Learning
        ↓
Model Training
        ↓
Validation
        ↓
Checkpoint Selection
        ↓
Test Evaluation
        ↓
Error Analysis
        ↓
Single-Image Inference
        ↓
Streamlit Application
```
# 1. Dataset Preparation

The dataset contains fabric images collected from two main sources:

- self-collected images
- additional images obtained from Hugging Face

The images are organized according to fabric material and defect labels.

# 2. Dataset Splitting

The dataset is divided into:

- training set
- validation set
- test set

The training set is used for model optimization.

The validation set is used during training to monitor performance and select the best checkpoint.

The test set is kept independent and is used for final evaluation.

# 3. Leakage Checking

Before model evaluation, the project checks for possible overlap between dataset splits.

The project contains utilities for:

- duplicate detection
- cross-split leakage checking
- strict similarity checking

Perceptual hashing is used for identifying potentially similar images.

This helps reduce the risk of evaluating the model on images that are identical or very similar to training samples.

# 4. Image Preprocessing

Input images are:

1. Converted to RGB
2. Resized to 224 × 224
3. Converted to tensors
4. Normalized

Training may additionally use data augmentation to increase variation in the training data.

# 5. Model Architecture

The project uses transfer learning with pretrained ResNet architectures from Torchvision.

Two model variants are available:

- ResNet18
- ResNet50

The pretrained ResNet is used as a shared feature extractor.

The original classification layer is replaced by two classification heads:

                 ResNet Feature Extractor
                           │
                    Feature Vector
                       /        \
                      /          \
                     ▼            ▼
             Fabric Head      Defect Head
                     │            │
                     ▼            ▼
               11 classes     11 classes

This allows the model to learn visual features that are useful for both classification tasks.

6. Training

The model is trained using two classification objectives.

The losses are:
```bash
Fabric Loss
Defect Loss
```
The total loss is:
```bash
Total Loss = Fabric Loss + Defect Loss
```
The project uses the Adam optimizer.

Current training configuration:

- Learning rate: 0.0001
- Batch size: 16
- Maximum epochs: 50
- Early stopping patience: 5
- 
# 7. Model Selection

After each training epoch, the model is evaluated on the validation set.

The project calculates:

- Fabric Accuracy
- Defect Accuracy

The checkpoint-selection metric is:
```bash
Average Accuracy = (Fabric Accuracy + Defect Accuracy) / 2
```
When validation performance improves, the corresponding model checkpoint is saved.

Training stops when the validation metric does not improve for the configured patience period.

# 8. Evaluation

The final model is evaluated on an independent test set.

The evaluation pipeline calculates classification performance separately for fabric and defect prediction.

The project also generates classification reports and confusion matrices for error analysis.

# 9. Current Results

The current ResNet18 checkpoint achieved the following validation performance:

| Metric           |  Result |
| ---------------- | ------: |
| Fabric Accuracy  | 100.00% |
| Defect Accuracy  |  99.06% |
| Average Accuracy |  99.53% |


Validation set size:
```bash
2,127 images
```
On the independent test set:

| Metric           |  Result |
| ---------------- | ------: |
| Fabric Accuracy  | 100.00% |
| Defect Accuracy  |  98.12% |
| Average Accuracy |  99.06% |

Test set size:
```bash
2,290 images
```
The difference between validation and test performance is expected because the test set is independent and provides a more realistic estimate of generalization.

# 10. Error Analysis

Confusion matrices are used to identify classes that are more difficult for the model.

One notable test-set difficulty is the Needle mark defect class, which has a reported recall of approximately 0.91.

Potential improvements include:

- collecting additional examples
- improving image quality
- improving class balance
- investigating visually similar defect categories
- experimenting with model architecture and preprocessing

# 11. Single-Image Inference

The inference pipeline is separated from the training code.

The prediction process is:

```text
Input Image
     ↓
Preprocessing
     ↓
Trained Model
     ↓
Fabric Probabilities
+
Defect Probabilities
     ↓
Predicted Classes
```

The inference functionality is implemented in:
```bash
src/predict.py
```

# 12. Streamlit Application

The project includes an interactive Streamlit application in:
```bash
src/app.py
```
The application allows users to:

1. Select a model
2. Upload an image
3. Analyze the image
4. View fabric prediction
5. View defect prediction
6. View confidence scores
7. View class probabilities

The application can be started locally with:
```bash
python -m streamlit run src/app.py
```

# 13. Limitations

The current results should be interpreted within the scope of the available dataset.

High test accuracy does not automatically guarantee equivalent performance in a real manufacturing environment.

Potential sources of distribution shift include:

- different cameras
- different lighting
- different fabric samples
- different backgrounds
- previously unseen defect appearances

Further testing on real production-line images would be needed before making industrial deployment claims.

