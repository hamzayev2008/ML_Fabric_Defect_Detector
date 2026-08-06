import torch

def validate(model, validation_loader, criterion):
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
            
    with torch.no_grad():
        for inputs, labels in validation_loader:
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            predicted = predictions.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(validation_loader)
    accuracy = correct / total_samples

    return average_loss, accuracy
    