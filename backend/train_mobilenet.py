import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
# Transform
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# Load Data
# ==========================

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

num_classes = len(train_data.classes)

print("Classes:", train_data.classes)

# ==========================
# Load MobileNetV2
# ==========================

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

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
# Training
# ==========================

train_losses = []
val_losses = []
accuracies = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ==========================
    # Validation
    # ==========================

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    val_losses.append(val_loss)
    accuracies.append(accuracy)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%")

# ==========================
# Save Model
# ==========================

torch.save(model.state_dict(), "mobilenet_model.pth")
print("Model saved as mobilenet_model.pth")

# ==========================
# Save Graphs
# ==========================

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNet Loss Graph")
plt.legend()
plt.savefig("mobilenet_loss.png")

plt.figure()
plt.plot(accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("MobileNet Accuracy Graph")
plt.legend()
plt.savefig("mobilenet_accuracy.png")

print("Graphs saved: mobilenet_loss.png, mobilenet_accuracy.png")

plt.show()