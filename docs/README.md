# Project Documentation

This directory contains the technical documentation for the **ML Toy Detector** project.

The documentation explains how the project is structured, how the machine learning pipeline works, how the dataset is organized, and how the model is evaluated.

---

## 📚 Documentation

### 1. Project Workflow

**[`project_workflow.md`](project_workflow.md)**

Explains the complete workflow of the project, from loading an image to the final prediction.

```text
Input Image
     ↓
Image Loading
     ↓
Preprocessing
     ↓
DataLoader
     ↓
ResNet18
     ↓
Training
     ↓
Validation
     ↓
Best Model
     ↓
Testing
     ↓
Prediction
     ↓
Streamlit
```

### 2. Architecture

**[`architecture.md`](architecture.md)**

Describes the architecture of the machine learning system.

It explains:

- ResNet18;
- input and output;
- feature extraction;
- classification layer;
- model training components;
- inference pipeline.

### 3. Methodology

methodology.md

Describes the methodology used to develop and train the model.

Topics include:

- image preprocessing;
- data augmentation;
- train/validation/test split;
- loss function;
- optimizer;
- training loop;
- validation;
- early stopping;
- model checkpointing.

### 4. Dataset Description

dataset_description.md

Describes the dataset used by the project.

It explains:

- dataset classes;
- dataset organization;
- training data;
- validation data;
- test data;
- image preprocessing;
- limitations of the dataset.

### 5. Project Structure

project_structure.md

Provides a detailed explanation of the project directory structure and the purpose of each important file.

For example:

```text
ML_toy_detector/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   ├── predict.py
│   └── app.py
│
├── train_dataset/
├── validation_dataset/
├── test_dataset/
│
└── docs/
```

---

## 🔄 Recommended Reading Order

If you are new to the project, it is recommended to read the documentation in the following order:

```text
1. Dataset Description
        ↓
2. Project Structure
        ↓
3. Project Workflow
        ↓
4. Architecture
        ↓
5. Methodology
```

This order makes it easier to understand how the dataset is prepared, how the project is organized, and how the model ultimately produces predictions.

---

## 🎯 Purpose of the Documentation

The documentation is intended to make the project:

- easier to understand;
- easier to reproduce;
- easier to maintain;
- easier to modify;
- easier to present as an ML project.

The main project overview is available in the root README.md.
