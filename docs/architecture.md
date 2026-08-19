# System Architecture

The ML Fabric Defect Detector is a computer vision classification system designed to analyze fabric images.

For each input image, the system predicts two independent properties:

1. Fabric material
2. Fabric defect

The model uses a shared pretrained ResNet backbone followed by two classification heads.

---

## High-Level Architecture

```text
                         Input Image
                              │
                              ▼
                    Image Preprocessing
                              │
                              ▼
                    Pretrained ResNet
                       ResNet18 / 50
                              │
                              ▼
                       Feature Vector
                         /         \
                        /           \
                       ▼             ▼
              Fabric Classifier   Defect Classifier
                       │             │
                       ▼             ▼
                 11 Fabric Classes  11 Defect Classes
```
1. Input

The system receives a single RGB fabric image.

The image is converted into the format expected by the neural network before inference.

2. Image Preprocessing

The preprocessing pipeline includes:

* Converting the image to RGB
* Resizing the image to 224 × 224
* Converting the image to a tensor
* Normalizing the input

Training may additionally use data augmentation.

Validation, testing, and inference use the corresponding evaluation/inference preprocessing without training augmentation.

3. Feature Extraction

The project uses pretrained ResNet architectures provided by Torchvision:

* ResNet18
* ResNet50

The pretrained ResNet acts as a shared feature extractor.

The original classification layer is replaced by two task-specific classification heads.

4. Fabric Classification Head

The first classification head predicts the fabric material.

The project currently contains 11 fabric classes:

* Cotton-Poly
* Linen-Poly
* Denim-Poly
* Velveteen-Poly
* Polyester-Poly
* Satin-Core
* Chiffon-Poly
* Nylon-Core
* Jacquard-Poly
* Oxford-Core
* Polyester (coated)-Core

5. Defect Classification Head

The second classification head predicts the fabric defect.

The project currently contains 11 defect classes:

* Normal
* Skipped stitch
* Broken stitch
* Pinched fabric
* Crooked seam
* Thread sagging
* Puckering
* Stain and damage
* Needle mark
* Bobbin thread pulling up
* Overlapped stitch

6. Multi-Task Output

The model produces two probability distributions:

```text
Input Image
     │
     ▼
Shared ResNet Backbone
     │
     ├──────────────► Fabric probabilities
     │
     └──────────────► Defect probabilities
```

The predicted class is selected from each probability distribution.

This allows the same image representation to be used for both material and defect classification.

7. Training Objective

Two classification losses are calculated:

```bash
Fabric Loss
Defect Loss
```

The total training loss is:

```bash
Total Loss = Fabric Loss + Defect Loss
```

Both tasks therefore contribute to training the shared feature extractor.

8. Evaluation

The model is evaluated separately for:

* Fabric classification
* Defect classification

The project also calculates an average accuracy:

```bash
Average Accuracy = (Fabric Accuracy + Defect Accuracy) / 2
```

This metric is used when selecting the best validation checkpoint.

9. Inference

Single-image inference is separated from the Streamlit interface.

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
      ├──────────────► Fabric Prediction
      │
      └──────────────► Defect Prediction
                              │
                              ▼
                           app.py
```

This separation keeps the machine learning logic independent from the user interface.

10. Streamlit Application

The interactive application is implemented in:

```bash
src/app.py
```

The application allows the user to:

Select a model
Upload an image
Analyze the image
View the predicted fabric material
View the predicted defect
View confidence scores and class probabilities

The application can be started from the project root with:

```bash
python -m streamlit run src/app.py
```
