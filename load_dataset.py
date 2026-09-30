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

print("\nClasses:")
print(train_data.class_names)

print("\nDataset loaded successfully!")