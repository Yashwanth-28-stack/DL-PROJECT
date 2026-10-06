import tensorflow as tf

# Dataset folders
train_path = "dataset/train"

# Load training and validation data from the same training folder
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

# Print and verify the label order
class_names = train_data.class_names
print("Classes:", class_names)

# Normalize images to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_data = validation_data.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Mild augmentation: horizontal flips only
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal")
])

# CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    data_augmentation,

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Stop when validation loss stops improving and restore the best weights
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save the best validation-accuracy checkpoint under a new filename
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "emotion_model_v2.keras",
    monitor="val_accuracy",
    save_best_only=True
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=30,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nTraining completed!")
print("Best model saved as emotion_model_v2.keras")