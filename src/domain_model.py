import torch.nn as nn
from torchvision import models


class DomainClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # Pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Number of input features to the original classifier
        num_features = self.model.fc.in_features

        # Binary classification:
        # 0 = Fabric
        # 1 = Non-Fabric
        self.model.fc = nn.Linear(
            num_features,
            2
        )

    def forward(self, x):

        return self.model(x)