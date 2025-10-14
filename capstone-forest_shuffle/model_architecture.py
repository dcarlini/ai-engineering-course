import torch
import torch.nn as nn
from torchvision import models

class MultiOutputModel(nn.Module):
    """Model with a ResNet backbone and multiple output heads."""
    def __init__(self, num_classes_dict):
        super(MultiOutputModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove the original fully connected layer

        self.heads = nn.ModuleDict({
            key: nn.Linear(num_ftrs, num_classes_dict[key]) for key in num_classes_dict.keys()
        })

    def forward(self, x):
        features = self.backbone(x)
        outputs = {key: head(features) for key, head in self.heads.items()}
        return outputs