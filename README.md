# ML Toy Detector

Computer vision project for automatic teddy bear quality classification.

The system uses a deep learning image classification model to determine whether a teddy bear is **normal** or **defective** from an input image.

This project was developed as an individual AI/ML Capstone Project.

---

## 🎯 Project Goal

The main goal of this project is to build a complete machine learning pipeline for automated toy quality inspection.

The system can:

- load and preprocess teddy bear images;
- apply data augmentation during training;
- train a ResNet18-based classifier;
- validate the model during training;
- automatically save the best model;
- stop training when validation performance stops improving;
- evaluate the trained model on unseen test images;
- generate classification metrics;
- generate a confusion matrix;
- identify incorrectly classified images;
- make predictions on individual images;
- provide a simple Streamlit interface.

---

# 🧠 Model

The project uses **ResNet18** as the image classification backbone.

The model performs binary classification:

| Class | Meaning |
|---|---|
| `normal` | Teddy bear without the target defect |
| `defective` | Teddy bear containing the target defect |

The final classification layer produces two output values (logits), one for each class.

During inference, the predicted class is selected using the class with the highest output score.

---

# 🔄 Project Pipeline

The complete pipeline can be summarized as:

```text
                    Input Image
                         │
                         ▼
                 ┌───────────────┐
                 │ dataLoader.py │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  dataset.py   │
                 │ TeddyDataset  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ image_utils.py│
                 │  load_image() │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ transforms.py │
                 │               │
                 │ Resize         │
                 │ Augmentation   │
                 │ ToTensor       │
                 │ Normalize      │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   model.py    │
                 │    ResNet18   │
                 └───────┬───────┘
                         │
                         ▼
                  Class Logits
                         │
                         ▼
                 ┌───────────────┐
                 │   train.py    │
                 │               │
                 │ CrossEntropy  │
                 │ Adam          │
                 │ Backprop      │
                 │ Early Stop    │
                 └───────┬───────┘
                         │
                         ▼
                  Best Model .pth
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌───────────┐        ┌───────────────┐
        │ validate  │        │    test.py    │
        │    .py    │        │               │
        └─────┬─────┘        │ Test Metrics  │
              │              └───────┬───────┘
              │                      │
              │             ┌────────┴─────────┐
              │             ▼                  ▼
              │       confusion_matrix.py   report
              │             │
              │             ▼
              │       Error Analysis
              │
              ▼
        Training Curves


                INFERENCE PIPELINE

                Input Image
                     │
                     ▼
                predict.py
                     │
                     ▼
                Preprocessing
                     │
                     ▼
                  ResNet18
                     │
                     ▼
                   Softmax
                     │
              ┌──────┴──────┐
              ▼             ▼
            normal       defective
              │             │
              └──────┬──────┘
                     ▼
                 app.py
                Streamlit
