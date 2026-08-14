import torch.nn as nn
from torchvision import models
from config import (
    MODEL_RESNET18_NAME,
    MODEL_RESNET50_NAME,
    FABRIC_CLASSES,
    DEFECT_CLASSES,
)

WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
}

class FabricDefectClassifier(nn.Module):

    def __init__(self, model_name):

        super().__init__()

        model_name = model_name.lower()

        if model_name == MODEL_RESNET18_NAME:
            model_function = models.resnet18

        elif model_name == MODEL_RESNET50_NAME:
            model_function = models.resnet50

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = model_function(weights=WEIGHTS[model_name])
        number_of_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fabric_classifier = nn.Linear(number_of_features, len(FABRIC_CLASSES))
        self.defect_classifier = nn.Linear(number_of_features, len(DEFECT_CLASSES))

    def forward(self, x):
        features = self.model(x)
        fabric_output = self.fabric_classifier(features)
        defect_output = self.defect_classifier(features)
        return fabric_output, defect_output