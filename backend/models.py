"""The three model architectures used in this project, defined once.

These match the definitions in train_cnn.py, rn_train_model.py,
train_mobilenet.py and api.py, so the saved .pth files load unchanged.
"""
import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms

# Same preprocessing as training: resize to 224x224, no normalisation.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

WEIGHT_FILES = {
    "resnet18": "rn_plant_disease_model.pth",
    "cnn": "cnn_model.pth",
    "mobilenetv2": "mobilenet_model.pth",
}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def build(name, num_classes):
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "cnn":
        m = SimpleCNN(num_classes)
    elif name == "mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    else:
        raise ValueError(name)
    return m


def load(name, weights_dir, num_classes, device="cpu"):
    m = build(name, num_classes)
    path = os.path.join(weights_dir, WEIGHT_FILES[name])
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def class_names(train_dir=None, json_path=None):
    """Class order = torchvision ImageFolder order (sorted folder names)."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return sorted(d.name for d in os.scandir(train_dir) if d.is_dir())
