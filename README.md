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
- train ResNet18 and ResNet50 classifiers;
- validate models during training;
- automatically save the best model;
- use early stopping;
- evaluate models on an unseen test dataset;
- generate classification reports;
- generate confusion matrices;
- identify incorrectly classified images;
- perform single-image inference;
- compare different models;
- provide a Streamlit interface for interactive predictions.

---

# 🧠 Model

The project uses pretrained ResNet architectures from Torchvision:

- ResNet18
- ResNet50

Both models are adapted for binary classification.

The original final classification layer is replaced with:

Linear → 2 classes

The models perform binary classification:

|    Class    |                 Meaning                 |
|-------------|-----------------------------------------|
| `normal`    | Teddy bear without the target defect    |
| `defective` | Teddy bear containing the target defect |

During inference, the model outputs logits which are converted into class probabilities using Softmax.

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
Model
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
Adam optimizer step
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

# 🚀 Training the Models

The training script accepts the model name.

ResNet18:
python src/train.py resnet18

The model is saved as:
teddy_classifier_resnet18.pth

ResNet50:
python src/train.py resnet50

The model is saved as:
teddy_classifier_resnet50.pth

The best checkpoint according to validation accuracy is saved automatically.

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

# 📊 Model Comparison

The current test results are:

```text
Metric      	        ResNet18	  ResNet50
Test Accuracy    	    80.77%	    84.62%
Normal Precision	    0.72	      0.91
Normal Recall	        100.00%	    76.92%
Defective Precision	  1.00	      0.80
Defective Recall	    61.54%	    92.31%
Macro F1	            0.80	      0.85
Wrong Predictions	    5	          4
```

### ResNet18
Confusion Matrix:

```text
              Predicted
              normal  defective

normal           13       0
defective         5       8
```

ResNet18 correctly identifies every normal image, but misses 5 defective images.

### ResNet50
Confusion Matrix:

```text
              Predicted
              normal  defective

normal           10       3
defective         1      12
```

ResNet50 correctly identifies 12 out of 13 defective images.

## Conclusion

For this particular test set, ResNet50 performs better overall.

The most important difference is defective-class recall:

ResNet18 → 61.54%
ResNet50 → 92.31%

Since the main purpose of the project is defect detection, ResNet50 is currently the preferred model.

The test set contains only 26 images, so these results should be considered experimental rather than production-level performance.

---

# 🔍 Error Analysis

Incorrect predictions are collected during evaluation.

For each wrong prediction, the project stores:

- image path;
- actual class;
- predicted class.

Example:

Actual: normal
Predicted: defective

Actual: defective
Predicted: normal

These images can be visualized to understand what types of examples are difficult for the model.

Error analysis can help identify:

- ambiguous images;
- difficult defect types;
- dataset problems;
- insufficient training examples;
- possible model weaknesses.

---

# 🎯 Single Image Prediction

Single-image inference is implemented in:

```text
src/predict.py

The prediction process is:

Input Image
     ↓
Preprocessing
     ↓
Selected Model
     ↓
Logits
     ↓
Softmax
     ↓
Class Probabilities
     ↓
Predicted Class
```

The prediction function returns:

- Predicted class
- Confidence
- Probability of normal
- Probability of defective

For example:

Prediction: defective

Normal: 7.69%
Defective: 92.31%

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit interface:
src/app.py

Run it with:
streamlit run src/app.py

The interface allows the user to:

- select ResNet18 or ResNet50;
- upload an image;
- analyze the image;
- view the uploaded image;
- see the prediction;
- see the confidence score;
- see probabilities for both classes;
- see which model was used;
- visualize the ML pipeline.

The interface displays the processing pipeline:

```text
📷 Input Image
      ↓
🔄 Resize
      ↓
🔢 ToTensor
      ↓
📊 Normalize
      ↓
🧠 ResNet
      ↓
⚡ Softmax
      ↓
🎯 Prediction
```

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
1. Train the model:
python src/train.py
2. The best validation model is automatically saved according to MODEL_PATH.
3. Validate / Test the model
4. Run the appropriate evaluation script:
python src/test.py
5. Generate confusion matrix and error analysis:
python src/confusion_matrix.py
6. Make a prediction:
python src/predict.py
7. Run Streamlit from the project root:
streamlit run src/app.py

---

# 🧰 Technologies

- Python
- PyTorch
- Torchvision
- ResNet18
- ResNet50
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

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

For a detailed explanation of the project architecture, methodology, dataset, workflow, and source code, see the [**full project documentation**](docs/README.md).

# 👤 Author

Fazliddin Hamzayev

AI/ML Capstone Project

GitHub:
https://github.com/hamzayev2008
