from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from domain_dataset import DomainDataset
from domain_model import DomainClassifier


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOMAIN_TEST_PATH = (
    PROJECT_ROOT
    / "domain_dataset"
    / "gate"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "domain_gate_resnet18.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "Fabric",
    "Non-Fabric",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DATASET
# ============================================================

test_dataset = DomainDataset(
    DOMAIN_TEST_PATH,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print()
print("=" * 60)
print("DOMAIN TEST DATASET")
print("=" * 60)

print(
    f"Test images: {len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

model = DomainClassifier().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()


# ============================================================
# TEST
# ============================================================

correct = 0
total = 0

all_true = []
all_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_true.extend(
            labels.cpu().tolist()
        )

        all_pred.extend(
            predictions.cpu().tolist()
        )


accuracy = correct / total


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("DOMAIN TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("DOMAIN CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_true,
        all_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()
print("Rows = TRUE")
print("Columns = PREDICTED")
print()

print(
    "             Fabric  Non-Fabric"
)

print(
    f"Fabric      {matrix[0][0]:6d}  {matrix[0][1]:10d}"
)

print(
    f"Non-Fabric  {matrix[1][0]:6d}  {matrix[1][1]:10d}"
)