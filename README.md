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

|    Class    |                 Meaning                 |
|-------------|-----------------------------------------|
| `normal`    | Teddy bear without the target defect    |
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
                 │ Resize        │
                 │ Augmentation  │
                 │ ToTensor      │
                 │ Normalize     │
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
```

---

# 🖼️ Image Preprocessing Pipeline

Before an image is passed to the neural network, it goes through several preprocessing steps.

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
Data Augmentation
      │
      ▼
ToTensor
      │
      ▼
Normalize
      │
      ▼
Tensor
      │
      ▼
ResNet18
```
---

## Training

During training, augmentation is applied to make the model more robust to variations in the input images.

Typical transformations are defined in:

src/transforms.py
Validation and Testing

Validation and test images are processed without training augmentation.

This prevents artificial transformations from affecting the evaluation results.

---

# 🏋️ Training Pipeline

The training process is implemented in:

```text
src/train.py

The training loop performs the following steps:

Load batch
    │
    ▼
Zero optimizer gradients
    │
    ▼
Forward pass
    │
    ▼
Calculate CrossEntropyLoss
    │
    ▼
Backpropagation
    │
    ▼
Optimizer step
    │
    ▼
Calculate training loss
    │
    ▼
Validate model
    │
    ▼
Check validation accuracy
    │
    ├── Improved → Save model
    │
    └── No improvement → Increase counter
    │
    ▼
Early stopping if necessary
```

The project uses:

Loss: CrossEntropyLoss
Optimizer: Adam
Learning rate: configured in config.py
Early stopping: enabled
Best model checkpoint: saved automatically

---

# 📊 Training Monitoring

During training, the following values are stored:

train_losses
validation_losses
validation_accuracies

These values are used to visualize model performance during training.

Training and Validation Loss

The loss curves help identify whether the model is learning and whether overfitting may be occurring.

Validation Accuracy

Validation accuracy shows how well the model performs on data that was not used to update the model weights.

---

# 🧪 Model Evaluation

The final model is evaluated using the test dataset.

The project generates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Incorrectly classified images

---

# 📈 Current Results

The current test set contains 26 images.

   Metric    |   Result   
-------------|------------
Accuracy	   |  92.31%
Precision	   |  0.92
Recall       |  0.92
F1-score     |	0.92
Test samples |	26

The classification report produced by the current model:

              precision    recall  f1-score   support

normal           0.92      0.92      0.92        13
defective        0.92      0.92      0.92        13

accuracy                              0.92        26
macro avg        0.92      0.92      0.92        26
weighted avg     0.92      0.92      0.92        26

Note: The current test set is small (26 images). Therefore, these results should be considered experimental rather than production-level performance.

---

# 🔲 Confusion Matrix

The project generates a confusion matrix using:

src/confusion_matrix.py

The matrix shows:

- correctly classified normal images;
- correctly classified defective images;
- normal images classified as defective;
- defective images classified as normal.

This makes it possible to understand not only the overall accuracy, but also the types of mistakes made by the model.

---

# 🔍 Error Analysis

Incorrect predictions are automatically collected during evaluation.

For each incorrect prediction, the project stores:

- Image path
- Actual class
- Predicted class

The incorrectly classified images can then be visualized to understand where the model makes mistakes.

Example:

┌──────────────────┬──────────────────┐
│ Actual: normal   │ Actual: defective│
│ Predicted:       │ Predicted:       │
│ defective        │ normal           │
└──────────────────┴──────────────────┘

Error analysis is useful for finding problems in the dataset and understanding limitations of the model.

---

# 🔮 Prediction

Single-image inference is implemented in:

src/predict.py

The inference process is:

```text
Input Image
     │
     ▼
Preprocessing
     │
     ▼
ResNet18
     │
     ▼
Model Outputs
     │
     ▼
Softmax
     │
     ▼
Class Probabilities
     │
     ▼
Predicted Class
```

The final result contains:

- predicted class;
- confidence score.

---

# 🖥️ Streamlit Interface

The project also contains a simple Streamlit interface:

src/app.py

The interface allows a user to:

- upload an image;
- run the trained model;
- see the predicted class;
- see the model confidence.

The Streamlit interface provides a simple way to demonstrate the trained model without manually running the prediction script.

---

# 📁 Project Structure

```text
ML_toy_detector/
│
├── assets/
│
├── dataset/
│
├── train_dataset/
│   ├── normal/
│   └── defective/
│
├── validation_dataset/
│   ├── normal/
│   └── defective/
│
├── test_dataset/
│   ├── normal/
│   └── defective/
│
├── docs/
│   ├── architecture.md
│   ├── dataset_description.md
│   ├── methodology.md
│   ├── project_structure.md
│   └── project_workflow.md
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── confusion_matrix.py
│   ├── dataLoader.py
│   ├── dataset.py
│   ├── image_utils.py
│   ├── model.py
│   ├── predict.py
│   ├── test.py
│   ├── train.py
│   ├── transforms.py
│   └── validate.py
│
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

# 📄 File-by-File Architecture

File	              |   Responsibility
--------------------|-----------------------------------------------------------
config.py	          |   Stores project configuration and hyperparameters
image_utils.py	    |   Loads and prepares individual images
transforms.py	      |   Defines image transformations and augmentation
dataset.py	        |   Implements the custom TeddyDataset
dataLoader.py	      |   Creates training, validation and test DataLoaders
model.py	          |   Defines the ResNet18-based classifier
train.py	          |   Trains the model
validate.py	        |   Calculates validation loss and accuracy
test.py	            |   Performs final model evaluation
predict.py	        |   Performs prediction on individual images
confusion_matrix.py |	  Generates metrics, confusion matrix and error analysis
app.py	            |   Provides the Streamlit interface

---

# 🗂️ Dataset

The project uses teddy bear images divided into two classes:

- normal
- defective

The dataset is organized into separate training, validation and test directories.

The images were collected from publicly available image sources, including Google Images and Roboflow Universe.

Different teddy bear appearances and defect types are included in the dataset.

---

# ⚙️ Installation

Clone the repository:

git clone https://github.com/hamzayev2008/ML_toy_detector.git

Enter the project directory:

cd ML_toy_detector

Install dependencies:

pip install -r requirements.txt

---

# 🚀 Running the Project
Train the model:
python src/train.py

The best validation model is automatically saved according to MODEL_PATH.

Validate / test the model

Run the appropriate evaluation script:

python src/test.py
Generate confusion matrix and error analysis
python src/confusion_matrix.py
Make a prediction
python src/predict.py
Run Streamlit

From the project root:

streamlit run src/app.py

---

# 🧰 Technologies
Python
PyTorch
Torchvision
ResNet18
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow
Streamlit

---

# ⚠️ Limitations

The current version has several limitations:

relatively small dataset;
relatively small test set;
binary classification only;
one teddy bear image is expected as input;
the model does not locate the exact defect;
confidence scores may not represent calibrated probabilities;
performance may decrease on images that differ significantly from the training data.

---

# 🔮 Future Improvements

Possible future improvements include:

- increasing the dataset size;
- collecting more diverse defect examples;
- improving data augmentation;
- hyperparameter optimization;
- model comparison;
- confidence calibration;
- defect localization;
- object detection;
- real-time camera inference;
- integration with a physical conveyor-belt inspection system;
- automatic separation of defective products.

---

# 🏭 Real-World Application

The long-term goal of the project is to use the classifier as part of an automated toy quality-control system.

A possible production workflow would be:

```text
Teddy Bear
     │
     ▼
Conveyor Belt
     │
     ▼
Camera
     │
     ▼
Image Capture
     │
     ▼
Preprocessing
     │
     ▼
ResNet18
     │
     ▼
┌───────────────┐
│ Classification│
└───────┬───────┘
        │
   ┌────┴────┐
   ▼         ▼
Normal    Defective
   │         │
   ▼         ▼
Continue   Reject
```

The current project is a software prototype of the classification component of such a system.

---

# 📚 Documentation

Additional documentation is available in the docs/ directory:

System Architecture
Project Workflow
Methodology
Dataset Description
Project Structure

# 👤 Author

Fazliddin Hamzayev

AI/ML Capstone Project

GitHub:
https://github.com/hamzayev2008
