# Source Code

This directory contains the Python source code for the ML Toy Detector.

The files are organized according to the machine learning pipeline, from image loading and preprocessing to model training, evaluation, prediction, and the Streamlit interface.

---

## 🔄 Source Code Pipeline

The main data flow is:

```text
Input Image
     │
     ▼
image_utils.py
     │
     ▼
transforms.py
     │
     ▼
dataset.py
     │
     ▼
dataLoader.py
     │
     ▼
model.py
     │
     ├──────────────► train.py
     │                    │
     │                    ▼
     │               validate.py
     │                    │
     │                    ▼
     │               Best Model
     │
     ├──────────────► test.py
     │                    │
     │                    ▼
     │              Model Evaluation
     │
     └──────────────► predict.py
                          │
                          ▼
                    Prediction

confusion_matrix.py
        │
        ├── Classification Report
        ├── Confusion Matrix
        └── Error Analysis

app.py
   │
   ▼
Streamlit Interface
```

---

## 📁 Files

# config.py

Contains the main project configuration.

It stores values such as:

- image size;
- batch size;
- number of epochs;
- learning rate;
- model path;
- early stopping configuration;
- class names.

Keeping these values in one file makes the project easier to configure.

# image_utils.py

Contains utilities for loading and preparing individual images.

The image loading function is used by the dataset to read an image from disk before it is passed through the preprocessing pipeline.

```text
Image file
    ↓
Load image
    ↓
Resize / Transform
    ↓
Tensor
```

# transforms.py

Defines the image preprocessing and augmentation pipeline.

The transformations include:

```text
Resize
   ↓
Data Augmentation
   ↓
ToTensor
   ↓
Normalize
```

Training data can use augmentation to improve model robustness.

Validation and test data use preprocessing without training augmentation.

# dataset.py

Contains the custom PyTorch TeddyDataset.

TeddyDataset:

- searches the class directories;
- assigns numerical labels;
- stores image paths;
- loads images when requested;
- applies the configured transformations.

The dataset uses the classes defined in config.py.

Example:

normal     → 0
defective  → 1

# dataLoader.py

Creates PyTorch DataLoader objects for:

- training;
- validation;
- testing.

The DataLoaders divide the dataset into batches that can be efficiently passed to the model.

```text
Dataset
   ↓
DataLoader
   ↓
Batches of images + labels
```

# model.py

Defines the TeddyClassifier model.

The classifier is based on ResNet18.

The model receives a preprocessed image and produces two output logits:

```text
Input Image
     ↓
ResNet18
     ↓
Feature Extraction
     ↓
Classifier
     ↓
2 Logits
     ↓
normal / defective
```

# train.py

Contains the model training loop.

The training process consists of:

```text
Load batch
    ↓
Forward pass
    ↓
Calculate loss
    ↓
Backpropagation
    ↓
Optimizer step
    ↓
Validation
    ↓
Save best model
```

The training script uses:

- CrossEntropyLoss;
- Adam optimizer;
- validation after each epoch;
- early stopping;
- best-model checkpointing.

Training and validation losses, as well as validation accuracy, are stored for visualization.

# validate.py

Contains the validation procedure.

It evaluates the model without updating its weights.

The function calculates:

- validation loss;
- validation accuracy.

The results are returned to train.py.

```text
Trained Model
      ↓
Validation Dataset
      ↓
validate()
      ↓
Validation Loss
Validation Accuracy
```

# test.py

Performs final evaluation of the trained model on the test dataset.

Unlike training and validation, the test set is used to estimate the final performance of the saved model on unseen data.

# predict.py

Performs inference on an individual image.

The basic pipeline is:

```text
Input Image
     ↓
Preprocessing
     ↓
Trained ResNet18
     ↓
Model Output
     ↓
Predicted Class
```

This file is useful for testing the model on new images outside the training process.

# confusion_matrix.py

Performs detailed model evaluation and error analysis.

It generates:

- classification report;
- confusion matrix;
- incorrectly classified images.

For incorrect predictions, the script stores:

- Image path
- Actual label
- Predicted label

This makes it possible to inspect examples where the model made a mistake.

# app.py

Contains the Streamlit user interface.

The interface provides a simple way to interact with the trained model without using the command line.

The general workflow is:

```text
Upload Image
     ↓
Preprocessing
     ↓
Trained Model
     ↓
Prediction
     ↓
Display Result
```

---

## 🧩 Dependencies Between Files

The main relationships between the source files are:

```text
config.py
   │
   ├──────────────┐
   │              │
   ▼              ▼
transforms.py   model.py
   │              │
   ▼              │
image_utils.py    │
   │              │
   ▼              │
dataset.py        │
   │              │
   ▼              │
dataLoader.py ────┘
   │
   ├──────────────► train.py
   │                    │
   │                    ▼
   │               validate.py
   │
   ├──────────────► test.py
   │
   ├──────────────► predict.py
   │
   └──────────────► confusion_matrix.py

model.py
   │
   └──────────────► app.py
```

---

## ▶️ Running the Source Code

All commands should be executed from the project root.

1. Train the model:
python src/train.py
2. Test the model:
python src/test.py
3. Generate evaluation results:
python src/confusion_matrix.py
4. Run prediction:
python src/predict.py
5. Start Streamlit:
streamlit run src/app.py

---

## 📚 Related Documentation

For a complete description of the project, see the [**← Back to Main README**](https://github.com/hamzayev2008/ML_toy_detector/blob/main/README.md).

For detailed technical documentation, see the documentation directory.
