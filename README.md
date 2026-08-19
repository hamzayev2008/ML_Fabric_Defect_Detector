# Fabric Defect Detector

A deep learning computer vision project for automatic **fabric material and defect classification** from images.

The system uses pretrained **ResNet18** and **ResNet50** architectures with a multi-output classification design. For each input image, the model predicts:

* the fabric material;
* the fabric defect.

This project was developed as an individual AI/ML Capstone Project.

---

## 🎯 Project Goal

The goal is to build a complete machine learning pipeline for automated fabric quality inspection.

The system is designed to:

* organize and validate an image dataset;
* detect possible duplicate and highly similar images across dataset splits;
* preprocess fabric images;
* train deep learning models using transfer learning;
* classify fabric material;
* classify fabric defects;
* validate the model during training;
* save the best-performing model;
* use early stopping;
* evaluate the final model on an unseen test dataset;
* generate classification reports;
* generate confusion matrices;
* perform single-image inference;
* provide an interactive Streamlit interface.

---

## 🧵 Classification Tasks

The model performs two classification tasks simultaneously.

### Fabric classes

The project contains 11 fabric classes:

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

### Defect classes

The project contains 11 defect classes:

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

## 🧠 Model Architecture

The project uses pretrained ResNet architectures from Torchvision:

* ResNet18
* ResNet50

The original ResNet classification head is removed.

The extracted feature representation is passed into two independent classification heads:

```text
                         Input Image
                              │
                              ▼
                         ResNet Backbone
                              │
                         Feature Vector
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Fabric Classifier    Defect Classifier
                    │                   │
                    ▼                   ▼
              11 fabric classes    11 defect classes
```

The model therefore produces two outputs:

```text
fabric_output, defect_output
```

Each output contains logits for its corresponding classification task.

---

## ⚙️ Training

Training is implemented in:

```text
src/train.py
```

The training process uses:

* pretrained ResNet weights;
* CrossEntropyLoss;
* Adam optimizer;
* configurable learning rate;
* validation after every epoch;
* best-model checkpointing;
* early stopping.

The total loss is calculated as:

```text
Total Loss = Fabric Loss + Defect Loss
```

The best model is selected using the average of fabric and defect validation accuracy.

### Training configuration

The main parameters are stored in:

```text
src/config.py
```

Current configuration includes:

```text
Image size:       224 × 224
Batch size:       16
Maximum epochs:   50
Learning rate:    0.0001
Early stopping:   5 epochs
```

---

## 🔍 Dataset Leakage Checking

Before training, the dataset was checked for possible leakage between:

```text
train
validation
test
```

Several checks are included:

```text
src/duplication_check.py
src/check_leakage.py
src/check_strict_similarity.py
```

The strict similarity checker uses perceptual hashing (`pHash`) to identify highly similar images between different dataset splits.

This helps reduce the risk that nearly identical images appear in both training and evaluation datasets.

---

## 📊 Validation Results

The trained ResNet18 model achieved:

| Metric           | Validation |
| ---------------- | ---------: |
| Fabric Accuracy  |    100.00% |
| Defect Accuracy  |     99.06% |
| Average Accuracy | **99.53%** |

Validation contained **2,127 images**.

The validation process is implemented in:

```text
src/validate.py
```

---

## 🧪 Test Results

The final ResNet18 model was evaluated on the unseen test dataset.

| Metric           |       Test |
| ---------------- | ---------: |
| Fabric Accuracy  |    100.00% |
| Defect Accuracy  |     98.12% |
| Average Accuracy | **99.06%** |

The test dataset contained **2,290 images**.

The test process is implemented in:

```text
src/test.py
```

---

## 📈 Test Classification Performance

### Fabric classification

The model achieved:

* Accuracy: **100.00%**
* Macro F1-score: **1.00**
* Weighted F1-score: **1.00**

All 11 fabric classes achieved perfect precision, recall, and F1-score on the reported test set.

### Defect classification

The model achieved:

* Accuracy: **98.12%**
* Macro F1-score: approximately **0.98**
* Weighted F1-score approximately **0.98**

The most difficult reported class was:

```text
Needle mark
```

with recall of **0.91** on the test set.

---

## 🔬 Evaluation

The reusable evaluation logic is implemented in:

```text
src/evaluation.py
```

It calculates:

* fabric accuracy;
* defect accuracy;
* average accuracy;
* fabric classification report;
* defect classification report;
* defect confusion matrix.

This logic is reused by the validation pipeline instead of duplicating the complete evaluation implementation.

---

## 🔮 Single-Image Inference

Single-image prediction is implemented in:

```text
src/predict.py
```

The inference pipeline is:

```text
Input Image
     │
     ▼
Image Loading
     │
     ▼
Resize to 224×224
     │
     ▼
Tensor Conversion
     │
     ▼
Normalization
     │
     ▼
ResNet
     │
     ├───────────────┐
     ▼               ▼
Fabric Prediction  Defect Prediction
     │               │
     └───────┬───────┘
             ▼
        Probabilities
```

The inference module is used by the Streamlit application.

---

## 🖥️ Streamlit Application

The interactive application is implemented in:

```text
python -m streamlit src/app.py
```

The application allows the user to:

1. select a model;
2. upload a fabric image;
3. analyze the image;
4. view the predicted fabric;
5. view the predicted defect;
6. see prediction confidence;
7. inspect class probabilities.

Run the application with:

```powershell
python -m streamlit run src/app.py
```

---

## 📁 Project Structure

```text
ML_toy_detector/
│
├── assets/
│
├── docs/
│   └── README.md
│
├── src/
│   ├── app.py
│   ├── check_leakage.py
│   ├── check_strict_similarity.py
│   ├── config.py
│   ├── dataLoader.py
│   ├── dataset.py
│   ├── duplication_check.py
│   ├── evaluation.py
│   ├── image_utils.py
│   ├── model.py
│   ├── predict.py
│   ├── split_dataset.py
│   ├── test.py
│   ├── test_dataset.py
│   ├── train.py
│   ├── transforms.py
│   └── README.md
│
├── classifier_resnet18.pth
├── classifier_resnet50.pth
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/hamzayev2008/ML_toy_detector.git
cd ML_toy_detector
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure PyTorch is installed with CUDA support if GPU acceleration is desired.

---

## 🏋️ Train ResNet18

From the project root:

```powershell
python src/train.py resnet18
```

The best checkpoint is saved as:

```text
classifier_resnet18.pth
```

---

## 🏋️ Train ResNet50

```powershell
python src/train.py resnet50
```

The best checkpoint is saved as:

```text
classifier_resnet50.pth
```

---

## ✅ Validate the Model

Run:

```powershell
python src/validate.py
```

This evaluates the current model on the validation split.

---

## 🧪 Test the Model

Run:

```powershell
python src/test.py
```

The test script reports:

* fabric accuracy;
* defect accuracy;
* average accuracy;
* classification reports;
* defect confusion matrix.

---

## 🔎 Dataset Checks

The project contains several scripts for checking the dataset before training.

For example:

```powershell
python src/check_strict_similarity.py
```

These checks help identify duplicate or highly similar images across dataset splits.

---

## 📚 Documentation

Additional technical documentation is available in:

```text
docs/README.md
```

Source-code documentation is available in:

```text
src/README.md
```

---

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* ResNet18
* ResNet50
* Pillow
* scikit-learn
* ImageHash
* Matplotlib
* Streamlit

---

## 📌 Current Status

The core machine learning pipeline is complete:

* dataset preparation;
* dataset split;
* leakage/similarity checking;
* training;
* validation;
* testing;
* evaluation;
* single-image inference.

The project also includes a Streamlit interface for interactive inference.

---

## 📄 License

This project is released under the MIT License.
