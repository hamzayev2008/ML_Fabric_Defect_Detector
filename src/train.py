import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import EPOCHS
from config import LEARNING_RATE
from config import MODEL_PATH
from model import TeddyClassifier
from dataLoader import train_loader, validation_loader
from validate import validate
from config import EARLY_STOPPING

model = TeddyClassifier()

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
        torch.save(model.state_dict(), MODEL_PATH)
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
    
plt.plot(train_losses, label="Train Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss Values")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
plt.plot([val_accuracy * 100 for val_accuracy in validation_accuracies], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()
plt.show()