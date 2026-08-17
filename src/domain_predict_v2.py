from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "domain_gate_resnet18_v2.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CHECK ARGUMENT
# ============================================================

if len(sys.argv) != 2:
    print()
    print("Usage:")
    print(
        'python .\\src\\domain_predict_v2.py "path\\to\\image.jpg"'
    )
    sys.exit(1)


image_path = Path(sys.argv[1])


if not image_path.exists():
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# IMAGE
# ============================================================

image = Image.open(image_path).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()


# ============================================================
# CLASSES
# ============================================================

classes = [
    "Fabric",
    "Non-Fabric"
]

predicted_class = classes[prediction]

fabric_probability = probabilities[0].item() * 100
non_fabric_probability = probabilities[1].item() * 100

confidence = probabilities[prediction].item() * 100


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("DOMAIN PREDICTION V2")
print("=" * 60)

print(f"Image:      {image_path.name}")
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

print()
print("Probabilities:")
print(f"  Fabric      : {fabric_probability:.2f}%")
print(f"  Non-Fabric  : {non_fabric_probability:.2f}%")