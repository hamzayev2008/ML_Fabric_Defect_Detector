import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from config import (
    CLASSES,
    MODEL_RESNET18_NAME,
    MODEL_RESNET50_NAME,
    MODEL_RESNET18_PATH,
    MODEL_RESNET50_PATH,
)
from model import TeddyClassifier
from dataLoader import test_loader

if len(sys.argv) > 1:
    model_name = sys.argv[1].lower()
else:
    model_name = MODEL_RESNET18_NAME

if model_name == MODEL_RESNET18_NAME:
    model_path = MODEL_RESNET18_PATH

elif model_name == MODEL_RESNET50_NAME:
    model_path = MODEL_RESNET50_PATH

else:
    raise ValueError("Unknown model. Choose 'resnet18' or 'resnet50'.")

print(f"Testing model: {model_name}")
print(f"Model path: {model_path}")

model = TeddyClassifier(model_name)

state_dict = torch.load(model_path, map_location="cpu")

model.load_state_dict(state_dict)

model.eval()

true_labels = []
predicted_labels = []
wrong_images = []

batch_start = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

        wrong = predicted != labels

        wrong_indices = (wrong.nonzero().flatten().tolist())

        for wrong_index in wrong_indices:

            global_index = (batch_start + wrong_index)

            path, actual_label = (test_loader.dataset.images[global_index])

            predicted_label = (predicted[wrong_index].item())

            wrong_images.append((path, actual_label, predicted_label))

        batch_start += len(labels)

print()

print("Classification Report:")

report = classification_report(true_labels, predicted_labels, target_names=CLASSES)

print(report)

cm = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")

print(cm)

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

plt.xticks(range(len(CLASSES)), CLASSES)

plt.yticks(range(len(CLASSES)), CLASSES)

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color=("white" if cm[i, j] > cm.max() / 2 else "black"))

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title(f"Confusion Matrix - {model_name}")

plt.colorbar()

plt.show()

print()

print(f"Wrong predictions: {len(wrong_images)}")

for path, actual_label, predicted_label in wrong_images:

    print(
        f"Actual: {CLASSES[actual_label]} | "
        f"Predicted: {CLASSES[predicted_label]} | "
        f"Path: {path}"
    )

if len(wrong_images) > 0:

    max_images = min(len(wrong_images), 6)

    plt.figure(figsize=(12, 6))

    for i in range(max_images):
        path, actual_label, predicted_label = (wrong_images[i])
        plt.subplot(2, 3, i + 1)
        image = plt.imread(path)
        plt.imshow(image)
        plt.title(
            f"Actual: {CLASSES[actual_label]}\n"
            f"Predicted: {CLASSES[predicted_label]}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()

else:

    print("No wrong predictions!")