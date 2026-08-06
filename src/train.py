import torch
import torch.nn as nn

from config import EPOCHS
from config import LEARNING_RATE
from config import MODEL_PATH
from model import TeddyClassifier
from dataLoader import train_loader, validation_loader
from validate import validate

model = TeddyClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_accuracy = 0

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
    
    print(f"Validation Loss: {validate_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.4f}%")

if val_accuracy >= best_accuracy:
    best_accuracy = val_accuracy
    torch.save(model.state_dict(), MODEL_PATH)