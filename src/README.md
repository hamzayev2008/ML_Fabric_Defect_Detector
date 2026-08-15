# Source Code Documentation

This directory contains the main Python implementation of the Fabric Defect Detector.

The source code is organized so that data preparation, model training, evaluation, inference, and the user interface are separated into different modules.

---

## 📁 Source Files

| File                         | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `app.py`                     | Streamlit user interface                                     |
| `check_leakage.py`           | Dataset leakage checks                                       |
| `check_strict_similarity.py` | Strict cross-split image similarity checking                 |
| `config.py`                  | Dataset paths, classes, model names, and training parameters |
| `dataLoader.py`              | Training and validation DataLoaders                          |
| `dataset.py`                 | PyTorch dataset implementation                               |
| `duplication_check.py`       | Duplicate-image checking                                     |
| `evaluation.py`              | Reusable model evaluation logic                              |
| `image_utils.py`             | Image loading and preprocessing utilities                    |
| `model.py`                   | ResNet-based multi-output classifier                         |
| `predict.py`                 | Single-image inference                                       |
| `split_dataset.py`           | Dataset splitting                                            |
| `test.py`                    | Final test-set evaluation                                    |
| `test_dataset.py`            | Dataset validation/testing utilities                         |
| `train.py`                   | Model training                                               |
| `transforms.py`              | Image transformations                                        |

---

# `config.py`

Central configuration file.

It contains:

* dataset paths;
* fabric class names;
* defect class names;
* model names;
* model checkpoint paths;
* image size;
* batch size;
* number of epochs;
* learning rate;
* early stopping configuration.

Current classification setup:

```text
11 fabric classes
11 defect classes
```

---

# `dataset.py`

Defines:

```python
FabricDataset
```

which extends PyTorch's:

```python
torch.utils.data.Dataset
```

The dataset:

1. discovers fabric folders;
2. creates fabric class indices;
3. discovers defect folders;
4. creates defect class indices;
5. validates defect numbering;
6. collects image paths;
7. loads images;
8. converts them to RGB;
9. applies the requested transform.

Each sample returns:

```text
image
fabric_label
defect_label
```

This allows the model to learn both classification tasks from the same image.

---

# `dataLoader.py`

Creates the DataLoaders used during training and validation.

The training DataLoader provides shuffled training batches.

The validation DataLoader provides validation batches without training augmentation.

---

# `transforms.py`

Contains image preprocessing and training transformations.

The preprocessing pipeline includes operations such as:

```text
Resize
ToTensor
Normalize
```

Training transformations may additionally include augmentation.

Validation, testing, and inference use the non-augmentation transformation pipeline.

---

# `model.py`

Defines:

```python
FabricDefectClassifier
```

The model supports:

```text
ResNet18
ResNet50
```

The pretrained ResNet backbone extracts image features.

The original fully connected layer is replaced by two classification heads:

```text
                    ResNet Backbone
                           │
                     Feature Vector
                       /         \
                      /           \
                     ▼             ▼
             Fabric Classifier  Defect Classifier
                  11 classes        11 classes
```

The forward method returns:

```python
fabric_output, defect_output
```

---

# `train.py`

Responsible for model training.

The training process:

```text
Load batch
    ↓
Forward pass
    ↓
Fabric loss
    +
Defect loss
    ↓
Total loss
    ↓
Backpropagation
    ↓
Adam optimizer
    ↓
Validation
    ↓
Best-model checkpoint
    ↓
Early stopping
```

The total loss is:

```text
Total Loss = Fabric CrossEntropyLoss
           + Defect CrossEntropyLoss
```

The best model is saved when average validation accuracy improves.

---

# `evaluation.py`

Contains the reusable:

```python
evaluate_model()
```

function.

This module was created to prevent validation and testing code from duplicating the same evaluation logic.

It calculates:

* fabric accuracy;
* defect accuracy;
* average accuracy;
* fabric classification report;
* defect classification report;
* defect confusion matrix.

The function returns these results in a dictionary.

---

# `validate.py`

Evaluates the trained model using the validation dataset.

It:

1. creates the validation dataset;
2. creates the validation DataLoader;
3. loads the trained model;
4. calls `evaluate_model()`;
5. prints the validation metrics and reports.

Run:

```powershell
python src/validate.py
```

Example current result:

```text
Fabric Accuracy:  100.00%
Defect Accuracy:  99.06%
Average Accuracy: 99.53%
```

---

# `test.py`

Performs the final evaluation using the unseen test dataset.

Unlike training validation, the test set is used as the final independent evaluation set.

The script reports:

* fabric accuracy;
* defect accuracy;
* average accuracy;
* fabric classification report;
* defect classification report;
* defect confusion matrix.

Run:

```powershell
python src/test.py
```

Current reported result:

```text
Fabric Accuracy:  100.00%
Defect Accuracy:  98.12%
Average Accuracy: 99.06%
```

---

# `predict.py`

Provides single-image inference.

It loads a trained checkpoint, preprocesses an image, runs the model, and converts the outputs into predictions and probabilities.

The module is used by:

```text
app.py
```

The important separation is:

```text
predict.py
    ↓
ML inference logic

app.py
    ↓
User interface
```

The Streamlit application should not duplicate the model inference implementation.

---

# `image_utils.py`

Contains utilities for loading and preparing images for inference.

The module is used to convert uploaded image data into a format suitable for the model.

---

# `duplication_check.py`

Checks the dataset for duplicate images.

This helps identify repeated images before model training.

---

# `check_leakage.py`

Checks for potential data leakage between dataset splits.

The important splits are:

```text
train
validation
test
```

The purpose is to reduce the possibility that the same image or an equivalent image appears in multiple splits.

---

# `check_strict_similarity.py`

Performs a stricter similarity analysis using perceptual hashing.

The script compares images between:

```text
train ↔ validation
train ↔ test
validation ↔ test
```

It reports suspicious image pairs based on perceptual hash distance.

A low Hamming distance indicates that two images are visually very similar.

---

# `split_dataset.py`

Responsible for creating the dataset splits.

The intended structure is:

```text
fabric_dataset/
├── train/
├── validation/
└── test/
```

Each split contains the same class hierarchy.

---

# `test_dataset.py`

Provides dataset-level testing utilities.

It can be used to verify that the dataset structure and loading process work correctly before training.

---

# `app.py`

Provides the interactive Streamlit interface.

The application allows users to:

* select a model;
* upload an image;
* run inference;
* view the predicted fabric;
* view the predicted defect;
* view confidence scores;
* inspect class probabilities.

Run:

```powershell
streamlit run src/app.py
```

---

## 🔗 Module Relationships

The main relationships between modules are:

```text
                    config.py
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      dataset.py    model.py    transforms.py
          │            │
          ▼            │
     dataLoader.py     │
          │            │
          └──────┬─────┘
                 ▼
              train.py
                 │
                 ▼
        classifier_resnet18.pth
                 │
          ┌──────┴───────┐
          ▼              ▼
     validate.py       test.py
          │              │
          └──────┬───────┘
                 ▼
           evaluation.py

Single image:

Image
  ↓
image_utils.py
  ↓
predict.py
  ↓
model.py
  ↓
app.py
```
