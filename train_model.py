import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42
)

# Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Load test dataset separately
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

# Class names
class_names = train_data.class_names

print("\nClasses:")
print(class_names)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_data = validation_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_data = test_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Calculate class weights
all_labels = []

for images, labels in train_data:
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(all_labels),
    y=all_labels
)

class_weights = dict(
    enumerate(class_weights_array)
)

print("\nClass weights:")
for i, weight in class_weights.items():
    print(class_names[i], ":", round(weight, 3))

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),

    # Data augmentation
    data_augmentation,

    # CNN layers
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classification layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(7, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save best model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# Train model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=30,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

print("\nTraining completed!")
print("Best model saved as emotion_model.keras")