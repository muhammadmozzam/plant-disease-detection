from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import io
import torch.nn.functional as F

app = FastAPI()

# ==========================
# CORS
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Load Classes
# ==========================

# Class order saved in classes.json (same order as ImageFolder used in
# training), so the API no longer needs the dataset folder to start.
import json
import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.json")) as f:
    class_names = json.load(f)
num_classes = len(class_names)

# ==========================
# LOAD MODELS
# ==========================

device = torch.device("cpu")

# --- ResNet18 ---
resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load("rn_plant_disease_model.pth", map_location=device))
resnet.eval()

# --- CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

cnn_model = SimpleCNN(num_classes)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn_model.eval()

# --- MobileNetV2 ---
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
mobilenet.load_state_dict(torch.load("mobilenet_model.pth", map_location=device))
mobilenet.eval()

# ==========================
# MODEL SELECTOR
# ==========================

def get_model(model_name):
    if model_name == "cnn":
        return cnn_model
    elif model_name == "mobilenet":
        return mobilenet
    else:
        return resnet  # default

# ==========================
# Transform
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# Prediction Route
# ==========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("resnet")  # default
):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    model = get_model(model_name)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    class_name = class_names[predicted.item()]
    conf = confidence.item()

    return {
        "model_used": model_name,
        "prediction": class_name,
        "confidence": round(conf * 100, 2)
    }