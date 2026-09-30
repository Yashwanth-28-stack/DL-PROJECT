import tensorflow as tf

train_path = "dataset/train"

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True
)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Apply augmentation
for images, labels in train_data.take(1):
    augmented_images = data_augmentation(images)

    print("Original image shape:", images.shape)
    print("Augmented image shape:", augmented_images.shape)

print("\nData augmentation completed successfully!")