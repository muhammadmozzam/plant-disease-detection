import matplotlib.pyplot as plt

# ==========================
# REAL MODEL RESULTS
# ==========================

models = ["CNN", "ResNet18", "MobileNetV2"]

accuracies = [91.5, 97.7, 99.0]
loss = [0.35, 0.08, 0.05]   # approx from graphs
training_time = [10, 18, 12]  # relative (adjust if you measured)

# ==========================
# ACCURACY COMPARISON
# ==========================

plt.figure()
bars = plt.bar(models, accuracies)

# Highlight best model
bars[2].set_color('green')

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")

# Add values on top
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.2, f"{v}%", ha='center')

plt.savefig("comparison_accuracy.png")

# ==========================
# LOSS COMPARISON
# ==========================

plt.figure()
bars = plt.bar(models, loss)
bars[2].set_color('green')

plt.title("Model Loss Comparison")
plt.ylabel("Loss")

for i, v in enumerate(loss):
    plt.text(i, v + 0.01, f"{v}", ha='center')

plt.savefig("comparison_loss.png")

# ==========================
# TRAINING TIME
# ==========================

plt.figure()
bars = plt.bar(models, training_time)
bars[0].set_color('orange')  # fastest model

plt.title("Training Time Comparison")
plt.ylabel("Relative Time")

for i, v in enumerate(training_time):
    plt.text(i, v + 0.5, f"{v}", ha='center')

plt.savefig("comparison_time.png")

# ==========================
# COMBINED ACCURACY LINE GRAPH
# ==========================

plt.figure()
plt.plot(models, accuracies, marker='o')
plt.title("Final Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")

for i, v in enumerate(accuracies):
    plt.text(i, v, f"{v}%", ha='center')

plt.savefig("comparison_line.png")

print("✅ All comparison graphs saved!")
plt.show()