import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================
# Paths
# ==========================

train_dir = "PlantVillage/train"
val_dir = "PlantVillage/val"

# ==========================
# Parameters
# ==========================

batch_size = 32
epochs = 5
learning_rate = 0.001

# ==========================
# Data Transformations
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# Load Dataset
# ==========================

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

num_classes = len(train_data.classes)

print("Classes:", train_data.classes)
print("Number of Classes:", num_classes)

# ==========================
# Load Model
# ==========================

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ==========================
# Device
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Using device:", device)

# ==========================
# Loss & Optimizer
# ==========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==========================
# Training Loop
# ==========================

train_losses = []
val_losses = []
accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    print(f"\nStarting Epoch {epoch+1}/{epochs}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ✅ Progress print
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}] Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ==========================
    # Validation
    # ==========================

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    accuracy = 100 * correct / total
    accuracies.append(accuracy)

    print(f"\nEpoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Accuracy: {accuracy:.2f}%")

# ==========================
# Save Model
# ==========================

torch.save(model.state_dict(), "plant_disease_model.pth")
print("Model saved as plant_disease_model.pth")

# ==========================
# Graph 1: Loss Curve
# ==========================

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# ==========================
# Graph 2: Accuracy Curve
# ==========================

plt.figure()
plt.plot(accuracies, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy over Epochs")
plt.show()

# ==========================
# Graph 3: Confusion Matrix
# ==========================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()