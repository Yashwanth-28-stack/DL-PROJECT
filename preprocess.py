import tensorflow as tf

train_path = "dataset/train"
test_path = "dataset/test"

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

# Normalize pixel values from 0-255 to 0-1
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_data = test_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Check one batch
for images, labels in train_data.take(1):
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())

print("\nPreprocessing completed successfully!")