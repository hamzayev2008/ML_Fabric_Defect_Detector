# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Improved Streamlit interface for single-image inference.
- Interactive ML pipeline visualization.
- Expandable stage-specific code inspection.
- Horizontal probability bar charts for fabric and defect predictions.
- Fabric / Non-Fabric domain validation with confidence score.

### Changed
- Improved Streamlit layout and visual hierarchy.
- Redesigned image upload and model selection controls.
- Improved presentation of prediction results.
- Simplified the Results section to focus on classification outputs and probabilities.

### Fixed
- Fixed domain classifier weight loading compatibility.
- Fixed Streamlit inference pipeline errors related to the domain classifier state dictionary.

---

## [1.0.0] - 2026-08-20

### Added
- ResNet18 and ResNet50 based fabric classification.
- Multi-output architecture for:
  - fabric material classification;
  - fabric defect classification.
- 11 fabric classes.
- 11 defect classes.
- Dataset leakage and similarity checking.
- Training, validation and testing pipelines.
- Early stopping and best-model checkpointing.
- Single-image inference.
- Streamlit web interface.
- Hugging Face model-weight loading.

### Machine Learning
- Transfer learning using pretrained ResNet architectures.
- Separate classification heads for fabric material and defect prediction.
- CrossEntropyLoss for both classification tasks.
- Adam optimizer.
- Configurable training parameters.

### Evaluation
- Fabric and defect accuracy metrics.
- Average accuracy.
- Classification reports.
- Confusion matrix generation.

### Documentation
- Added project documentation.
- Added model architecture description.
- Added training and evaluation documentation.

---

## [0.1.0] - 2026-07

### Added
- Initial ML project structure.
- Initial dataset preparation pipeline.
- Initial ResNet-based classification model.
- Initial training pipeline.
- Initial Streamlit prototype.
- Initial project documentation.

---

## Notes

### Model
The project currently supports:

- ResNet18
- ResNet50

### Classification
The system predicts:

1. Whether the input image belongs to the fabric domain.
2. Fabric material.
3. Fabric defect.

### Deployment
The application is designed to run through Streamlit:

```bash
python -m streamlit run src/app.py
