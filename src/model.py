import torch.nn as nn
from torchvision import models
from config import MODEL_NAME

WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
}

class TeddyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        model_function = getattr(models, MODEL_NAME)
        self.model = model_function(weights = WEIGHTS[MODEL_NAME])
        
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)