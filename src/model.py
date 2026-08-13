import torch.nn as nn
from torchvision import models
from config import MODEL_RESNET18_NAME, MODEL_RESNET50_NAME

WEIGHTS = {
    MODEL_RESNET18_NAME: models.ResNet18_Weights.DEFAULT,
    MODEL_RESNET50_NAME: models.ResNet50_Weights.DEFAULT,
}

class TeddyClassifier(nn.Module):
    def __init__(self, model_name=MODEL_RESNET18_NAME):
        super().__init__()
        
        model_function = getattr(models, model_name)
        
        self.model = model_function(weights = WEIGHTS[model_name])
        
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)