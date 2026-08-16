from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from domain_model import DomainClassifier


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "domain_gate_resnet18.pth"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Fabric",
    "Non-Fabric",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Check whether an image is Fabric or Non-Fabric."
)

parser.add_argument(
    "image",
    type=str,
    help="Path to image"
)

args = parser.parse_args()


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = Path(args.image)

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )

image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)


# ============================================================
# LOAD MODEL
# ============================================================

model = DomainClassifier().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image)

    probabilities = F.softmax(
        output,
        dim=1
    )

    confidence, index = torch.max(
        probabilities,
        dim=1
    )


predicted_index = index.item()
predicted_confidence = confidence.item()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("DOMAIN PREDICTION")
print("=" * 60)

print(
    f"Image:      {image_path.name}"
)

print(
    f"Prediction: {CLASS_NAMES[predicted_index]}"
)

print(
    f"Confidence: {predicted_confidence * 100:.2f}%"
)

print()
print("Probabilities:")

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    probability = (
        probabilities[0][class_index].item()
    )

    print(
        f"  {class_name:12s}: "
        f"{probability * 100:.2f}%"
    )
