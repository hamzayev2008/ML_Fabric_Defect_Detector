import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import (
    EPOCHS,
    LEARNING_RATE,
    MODEL_RESNET18_PATH,
    MODEL_RESNET50_PATH,
    EARLY_STOPPING,
)
from model import FabricDefectClassifier
from dataLoader import train_loader, validation_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL_PATHS = {
    "resnet18": MODEL_RESNET18_PATH,
    "resnet50": MODEL_RESNET50_PATH,
}

def train(model_name):

    model_name = model_name.lower()

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    print("=" * 60)
    print(f"Training model: {model_name}")
    print(f"Model will be saved to: {model_path}")
    print("=" * 60)

    model = FabricDefectClassifier(model_name)
    model = model.to(device)
    fabric_criterion = nn.CrossEntropyLoss()
    defect_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0
    epochs_without_improvement = 0
    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, fabric_labels, defect_labels in train_loader:
            images = images.to(device)
            fabric_labels = fabric_labels.to(device)
            defect_labels = defect_labels.to(device)
            
            optimizer.zero_grad()
            fabric_output, defect_output = model(images)
            fabric_loss = fabric_criterion(fabric_output, fabric_labels)
            defect_loss = defect_criterion(defect_output, defect_labels)
            loss = fabric_loss + defect_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        validation_loss = 0.0
        correct_fabric = 0
        correct_defect = 0
        total_samples = 0

        model.eval()
        
        with torch.no_grad():
            for images, fabric_labels, defect_labels in validation_loader:
                images = images.to(device)
                fabric_labels = fabric_labels.to(device)
                defect_labels = defect_labels.to(device)
    
                fabric_output, defect_output = model(images)
                fabric_loss = fabric_criterion(fabric_output, fabric_labels)
                defect_loss = defect_criterion(defect_output, defect_labels)
                loss = fabric_loss + defect_loss
                validation_loss += loss.item()
                fabric_predictions = fabric_output.argmax(dim=1)
                defect_predictions = defect_output.argmax(dim=1)
                correct_fabric += (fabric_predictions == fabric_labels).sum().item()
                correct_defect += (defect_predictions == defect_labels).sum().item()
                total_samples += images.size(0)
        validation_loss /= len(validation_loader)
        fabric_accuracy = (correct_fabric / total_samples)
        defect_accuracy = (correct_defect / total_samples)
        average_accuracy = (fabric_accuracy + defect_accuracy) / 2
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}"
        )
        print(
            f"Validation Loss: {validation_loss:.4f}, "
            f"Fabric Accuracy: {fabric_accuracy * 100:.2f}%, "
            f"Defect Accuracy: {defect_accuracy * 100:.2f}%, "
            f"Average Accuracy: {average_accuracy * 100:.2f}%"
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(average_accuracy)
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            epochs_without_improvement = 0
            
            torch.save(model.state_dict(), model_path)

            print(
                f"Best model saved! "
                f"Average Accuracy: {best_accuracy * 100:.2f}%"
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best average accuracy: "
                f"{best_accuracy * 100:.2f}%"
            )
            break
    print()
    print("Training finished.")
    print(f"Best average accuracy: " f"{best_accuracy * 100:.2f}%")
    print(f"Model saved to: {model_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        [
            accuracy * 100
            for accuracy in validation_accuracies
        ],
        label="Average Validation Accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} - Validation Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python src/train.py "
            "resnet18"
        )

        print(
            "   or: python src/train.py "
            "resnet50"
        )
        sys.exit(1)
    train(sys.argv[1])