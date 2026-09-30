import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("emotion_model.keras")

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

class_names = test_data.class_names

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_data = test_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Get actual labels
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Display confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm)

plt.title("Emotion Recognition Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

plt.colorbar()

plt.tight_layout()
plt.show()