# Project Workflow

The ML Fabric Defect Detector follows an end-to-end computer vision workflow.

```text
                Dataset Sources
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Self-Collected          Hugging Face
      Images                  Data
          │                     │
          └──────────┬──────────┘
                     ▼
              Dataset Preparation
                     │
                     ▼
          Train / Validation / Test
                     │
                     ▼
            Duplicate Checking
                     │
                     ▼
          Cross-Split Leakage Check
                     │
                     ▼
          Image Preprocessing
                     │
                     ▼
             Transfer Learning
                     │
              ResNet18 / ResNet50
                     │
                     ▼
                Model Training
                     │
                     ▼
                 Validation
                     │
                     ▼
             Best Checkpoint
                     │
                     ▼
               Test Evaluation
                     │
                     ▼
               Error Analysis
                     │
                     ▼
             Single-Image Inference
                     │
                     ▼
             Streamlit Application
```

# 1. Dataset Collection

The dataset is built from:

* self-collected fabric images
* additional images obtained from Hugging Face

The images are organized according to fabric material and defect labels.

# 2. Dataset Preparation

Images are prepared for model training and evaluation.

The preparation process includes:

* organizing images by class
* checking image files
* converting images to the required format
* preparing the dataset splits
* 
# 3. Train / Validation / Test Split

The dataset is divided into three independent subsets:
```text
Train
Validation
Test
```
The test set is kept independent from the training process.

# 4. Duplicate and Leakage Checks

Before training and evaluation, the project checks for possible duplication and cross-split similarity.

The checks include:
```text
Duplicate checking
        ↓
Cross-split leakage checking
        ↓
Strict similarity checking
```
This helps reduce the possibility of overly optimistic evaluation results.

# 5. Image Preprocessing

Images are processed before entering the neural network.
```text
RGB Conversion
      ↓
Resize to 224 × 224
      ↓
Tensor Conversion
      ↓
Normalization
```
Training may additionally apply data augmentation.

# 6. Model Training

The project uses transfer learning with pretrained:
```bash
ResNet18
ResNet50
```
The ResNet backbone extracts visual features.

Two classification heads then predict:
```bash
Fabric Material
+
Fabric Defect
```

# 7. Validation

After each training epoch, the model is evaluated on the validation set.

The project calculates:

* Fabric Accuracy
* Defect Accuracy
* Average Accuracy

The best checkpoint is selected according to the validation average accuracy.

# 8. Test Evaluation

After training and model selection, the selected checkpoint is evaluated on the independent test set.

The current ResNet18 checkpoint achieved:
```bash
Fabric Accuracy: 100.00%
Defect Accuracy: 98.12%
Average Accuracy: 99.06%
```
on the current test set of 2,290 images.

# 9. Error Analysis

The evaluation results are analyzed using:

* classification reports
* confusion matrices
* per-class performance

The analysis helps identify classes that are harder for the model to distinguish.

# 10. Single-Image Inference

The trained model can be used to analyze individual images.

```text
Image
  ↓
Preprocessing
  ↓
Model
  ↓
Fabric Prediction
+
Defect Prediction
  ↓
Confidence Scores
```

The inference functionality is implemented separately from the Streamlit interface.

# 11. Streamlit Application

The project includes a Streamlit interface for interactive inference.

The application workflow is:

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

The application is located at:
```bash
src/app.py
```
Run it locally from the project root with:
```bash
python -m streamlit run src/app.py
```
# 12. Reproducibility

The main stages of the project can be reproduced using the scripts in src/.

Typical commands include:
```bash
python src/train.py resnet18
python src/train.py resnet50
python src/validate.py
python src/test.py
python -m streamlit run src/app.py
```
The exact dataset preparation process and environment should be documented when reproducing the training process.
