import torch
import torch.nn as nn
from config import MODEL_PATH
from model import TeddyClassifier
from dataLoader import test_loader

model = TeddyClassifier()

criterion = nn.CrossEntropyLoss()

def test(model, test_loader, criterion):
    
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
            
    with torch.no_grad():
        for inputs, labels in test_loader:
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            predicted = predictions.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = correct / total_samples

    return average_loss, accuracy

test_loss, test_accuracy = test(model, test_loader, criterion)
 
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%")