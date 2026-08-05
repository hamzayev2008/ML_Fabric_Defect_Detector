import torch
import torch.nn as nn

from config import EPOCHS
from config import LEARNING_RATE
from config import MODEL_PATH
from model import TeddyClassifier
from dataLoader import train_loader

model = TeddyClassifier()

model.train()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    
    total_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, labels)

        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")
    
torch.save(model.state_dict(), MODEL_PATH)