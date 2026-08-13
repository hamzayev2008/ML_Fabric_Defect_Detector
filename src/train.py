import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import (
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING,
    MODEL_RESNET18_NAME,
    MODEL_RESNET50_NAME,
    MODEL_RESNET18_PATH,
    MODEL_RESNET50_PATH,
)
from model import TeddyClassifier
from dataLoader import train_loader, validation_loader
from validate import validate

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


print(f"Training model: {model_name}")
print(f"Model will be saved to: {model_path}")

model = TeddyClassifier(model_name)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_accuracy = 0

epochs_without_improvement = 0

train_losses = []

validation_losses = []

validation_accuracies = []

for epoch in range(EPOCHS):
    
    model.train()
    
    total_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, labels)

        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss / len(train_loader):.4f}")
    
    validate_loss, val_accuracy = validate(
        model,
        validation_loader,
        criterion
    )
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), model_path)
        
        print(
            f"Best model saved! "
            f"Accuracy: {best_accuracy * 100:.2f}%"
        )
        
    else:
        epochs_without_improvement += 1
        
    print(f"Validation Loss: {validate_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.4f}%")
    
    train_losses.append(total_loss / len(train_loader))
    validation_losses.append(validate_loss)
    validation_accuracies.append(val_accuracy)
    
    if epochs_without_improvement >= EARLY_STOPPING:
        print(
            f"Early stopping at epoch {epoch + 1}. "
            f"Best validation accuracy: {best_accuracy * 100:.2f}%"
        )
        break
    
print()
print("Training finished.")
print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)
print(f"Model saved to: {model_path}")
    
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss Values")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot([val_accuracy * 100 for val_accuracy in validation_accuracies], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title(f"{model_name}: Validation Accuracy")
plt.legend()
plt.show()