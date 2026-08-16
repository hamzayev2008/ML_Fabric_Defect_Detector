import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

from domain_dataset import DomainDataset
from domain_model import DomainClassifier


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOMAIN_DATASET_PATH = PROJECT_ROOT / "domain_dataset" / "gate"

MODEL_PATH = PROJECT_ROOT / "domain_gate_resnet18.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

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
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = DomainDataset(
    DOMAIN_DATASET_PATH / "train",
    transform=train_transform
)

validation_dataset = DomainDataset(
    DOMAIN_DATASET_PATH / "validation",
    transform=validation_transform
)


print()
print("=" * 60)
print("DOMAIN DATASET")
print("=" * 60)

print(
    f"Train:      {len(train_dataset)}"
)

print(
    f"Validation: {len(validation_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = DomainClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    correct = 0
    total = 0

    all_true = []
    all_pred = []

    with torch.no_grad():

        for images, labels in validation_loader:

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

    return accuracy, all_true, all_pred


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 0.0
epochs_without_improvement = 0


print()
print("=" * 60)
print("TRAINING DOMAIN CLASSIFIER")
print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    validation_accuracy, _, _ = validate()

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Acc: {validation_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  ✓ Best model saved: "
            f"{best_accuracy * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

print()
print("=" * 60)
print("BEST VALIDATION RESULTS")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)


best_accuracy, all_true, all_pred = validate()


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

print()
print(
    f"Model saved to: {MODEL_PATH}"
)