# ML_toy_detector

Computer vision project for detecting defective teddy bears in a toy factory.

## Overview

This project aims to automate the quality inspection process in toy manufacturing. Instead of relying entirely on manual inspection, the system uses deep learning to determine whether a teddy bear is **normal** or **defective** from an input image.

The project is developed as an individual AI/ML Capstone Project.

---

## Project Objectives

- Detect defective teddy bears from images
- Reduce manual inspection workload
- Improve production quality control
- Build a reusable deep learning pipeline

---

## Dataset

The dataset contains real teddy bear images collected from:

- Google Images
- Roboflow Universe (Teddy Bear Dataset)

Images are labeled as:

- Normal
- Defective

The dataset includes different teddy bear colors and defect types.

---

## Technologies

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## System Architecture

```text
            Input Image
                 │
                 ▼
        Image Preprocessing
                 │
                 ▼
       CNN (YOLO Classifier is taken as an example)
                 │
                 ▼
        Binary Classification
                 │
                 ▼
      Normal / Defective
      + Confidence Score
```

## Project Structure

```
ML_toy_detector/
│
├── dataset/
├── docs/
├── models/
├── notebooks/
├── results/
├── src/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Project Workflow

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Data Augmentation
   │
   ▼
Train / Validation Split
   │
   ▼
Model Training
   │
   ▼
Evaluation
   │
   ▼
Prediction
```

---

## Development Roadmap

- [x] Project proposal
- [x] Dataset collection
- [ ] Data preprocessing
- [ ] Model implementation
- [ ] Model training
- [ ] Model evaluation
- [ ] Performance optimization
- [ ] Final deployment

---

## Expected Input

A photo containing one teddy bear.

## Expected Output

- Predicted class
  - Normal
  - Defective
- Confidence score

---

## Evaluation Metrics

The model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

---

## Model Performance

| Metric | Value |
|---------|------:|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-score | - |

*Results will be added after model training.*

## Installation

Clone the repository

```bash
git clone https://github.com/your_username/ML_toy_detector.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Repository Layout

| Folder | Description |
|---------|-------------|
| dataset | Training images |
| docs | Documentation |
| src | Source code |
| models | Saved models |
| notebooks | Experiments |
| results | Training results and evaluation |

## Future Improvements

- YOLO object detection
- Defect localization
- More training images
- Better data augmentation
- Web interface  

## Documentation

Additional documentation can be found in the `docs/` directory:

- System Architecture
- Project Workflow
- Methodology
- Dataset Description
- Project Structure

---

## Author

**Fazliddin Hamzayev**

AI/ML Capstone Project
Repository initialized successfully.