# Project Structure

The repository is organized into documentation, source code, project assets, trained models, and configuration files.

```text
ML_Fabric_Defect_Detector/
│
├── assets/
│
├── docs/
│   ├── images/
│   ├── README.md
│   ├── architecture.md
│   ├── dataset_description.md
│   ├── methodology.md
│   ├── project_structure.md
│   └── project_workflow.md
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── dataLoader.py
│   ├── dataset.py
│   ├── domain_model.py
│   ├── evaluation.py
│   ├── image_utils.py
│   ├── model.py
│   ├── predict.py
│   ├── test.py
│   ├── train.py
│   ├── transforms.py
│   └── validate.py
│
├── classifier_resnet18.pth
├── classifier_resnet50.pth
│
├── CHANGELOG.md
├── LICENSE
├── README.md
├── index.html
└── requirements.txt
```

```md
assets/
```
Contains project assets used for documentation or presentation.


```markdown
docs/
```
Contains the technical documentation for the project.


```markdown
docs/README.md
```
Provides a technical overview of the complete machine learning pipeline.


```markdown
docs/architecture.md
```
Describes the system and model architecture.


```markdown
docs/dataset_description.md
```
Describes the dataset sources, classes, preprocessing, splits, and leakage checks.


```markdown
docs/methodology.md
```
Describes the methodology used for data preparation, training, validation, evaluation, and inference.


```markdown
docs/project_structure.md
```
Documents the repository organization.


```markdown
docs/project_workflow.md
```
Describes the end-to-end project workflow.


```markdown
docs/images/
```
Contains images used by the technical documentation.


```markdown
src/
```
Contains the main Python source code.


```markdown
app.py
```
Provides the Streamlit user interface for single-image inference.


```markdown
config.py
```
Stores project configuration such as image size, class definitions, and model paths.


```markdown
dataset.py
```
Provides dataset loading functionality.


```markdown
dataLoader.py
```
Handles data loading and preparation for training and evaluation.


```markdown
model.py
```
Defines the neural network architecture.


```markdown
domain_model.py
```
Contains the domain/fabric-related model components used by the inference pipeline.


```markdown
transforms.py
```
Defines image preprocessing and transformation pipelines.


```markdown
train.py
```
Contains the model training workflow.


```markdown
validate.py
```
Runs validation evaluation.


```markdown
test.py
```
Runs final test evaluation.


```markdown
evaluation.py
```
Contains reusable evaluation logic such as accuracy calculations and classification reports.


```markdown
predict.py
```
Handles single-image model inference.


```markdown
image_utils.py
```
Contains utilities for loading and processing input images.


# Trained Model Checkpoints
```bash
classifier_resnet18.pth
classifier_resnet50.pth
```
These files contain trained model weights used by the inference pipeline.

# Project Configuration Files

requirements.txt

Lists the Python dependencies required by the project.

README.md

Contains the main project overview, setup instructions, methodology summary, results, and usage information.

LICENSE

Contains the project's license information.

CHANGELOG.md

Contains project change history.

index.html

Contains the project showcase page used for the portfolio/project presentation.

Separation of Responsibilities

The repository separates major responsibilities:

```text
Configuration
     ↓
Data Loading
     ↓
Preprocessing
     ↓
Model
     ↓
Training
     ↓
Validation / Testing
     ↓
Inference
     ↓
Streamlit Interface
```

This structure makes the project easier to understand, test, and modify.
