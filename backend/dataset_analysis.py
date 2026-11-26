import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

# ==========================
# Dataset Paths
# ==========================

train_path = "PlantVillage/train"
val_path = "PlantVillage/val"

# ==========================
# Count Images per Class
# ==========================

data = []

for folder in [train_path, val_path]:

    for cls in os.listdir(folder):

        class_path = os.path.join(folder, cls)

        if os.path.isdir(class_path):

            num_images = len(os.listdir(class_path))

            data.append([cls, num_images, os.path.basename(folder)])

df = pd.DataFrame(data, columns=["Class", "Images", "Dataset"])

print("\nDataset Summary\n")
print(df)

print("\nTotal Classes:", df["Class"].nunique())
print("Total Images:", df["Images"].sum())

# ==========================
# Bar Chart
# ==========================

plt.figure(figsize=(14,6))

sns.barplot(x="Class", y="Images", hue="Dataset", data=df)

plt.xticks(rotation=90)

plt.title("Image Distribution per Class (Train vs Validation)")

plt.tight_layout()

plt.show()

# ==========================
# Pie Chart (Total Images)
# ==========================

total_per_class = df.groupby("Class")["Images"].sum()

plt.figure(figsize=(8,8))

plt.pie(total_per_class, labels=total_per_class.index, autopct='%1.1f%%')

plt.title("Dataset Class Distribution")

plt.show()

# ==========================
# Show Sample Images
# ==========================

classes = os.listdir(train_path)

plt.figure(figsize=(12,8))

for i, cls in enumerate(classes[:6]):

    class_path = os.path.join(train_path, cls)

    img_name = random.choice(os.listdir(class_path))

    img_path = os.path.join(class_path, img_name)

    img = Image.open(img_path)

    plt.subplot(2,3,i+1)

    plt.imshow(img)

    plt.title(cls)

    plt.axis("off")

plt.tight_layout()

plt.show()