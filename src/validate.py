import torch
from torch.utils.data import DataLoader
from config import (
    VALIDATION_DATASET_PATH,
    MODEL_RESNET18_PATH,
    FABRIC_CLASSES,
    DEFECT_CLASSES,
)
from dataset import FabricDataset
from model import FabricDefectClassifier
from evaluation import evaluate_model

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATASET
# ============================================================

validation_dataset = FabricDataset(dataset_path=VALIDATION_DATASET_PATH, augmentation=False)

validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

# ============================================================
# MODEL
# ============================================================

model = FabricDefectClassifier("resnet18")

state_dict = torch.load(MODEL_RESNET18_PATH, map_location=device)

model.load_state_dict(state_dict)
model = model.to(device)

# ============================================================
# EVALUATION
# ============================================================

results = evaluate_model(
    model=model,
    data_loader=validation_loader,
    fabric_classes=FABRIC_CLASSES,
    defect_classes=DEFECT_CLASSES,
    device=device
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(
    f"Fabric Accuracy:  "
    f"{results['fabric_accuracy'] * 100:.2f}%"
)

print(
    f"Defect Accuracy:  "
    f"{results['defect_accuracy'] * 100:.2f}%"
)

print(
    f"Average Accuracy: "
    f"{results['average_accuracy'] * 100:.2f}%"
)


# ============================================================
# FABRIC REPORT
# ============================================================

print()
print("=" * 60)
print("FABRIC CLASSIFICATION REPORT")
print("=" * 60)

print(results["fabric_report"])


# ============================================================
# DEFECT REPORT
# ============================================================

print()
print("=" * 60)
print("DEFECT CLASSIFICATION REPORT")
print("=" * 60)

print(results["defect_report"])


# ============================================================
# DEFECT CONFUSION MATRIX
# ============================================================

print()
print("=" * 60)
print("DEFECT CONFUSION MATRIX")
print("=" * 60)

matrix = results["defect_matrix"]

print()
print("Rows = TRUE")
print("Columns = PREDICTED")
print()

print("             ", end="")

for i in range(len(DEFECT_CLASSES)):
    print(f"{i:4}", end="")

print()

for i, row in enumerate(matrix):

    print(
        f"{i:2} "
        f"{DEFECT_CLASSES[i][:18]:18}",
        end=" "
    )

    for value in row:
        print(f"{value:4}", end="")

    print()