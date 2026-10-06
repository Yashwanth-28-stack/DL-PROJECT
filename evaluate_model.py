import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("emotion_model_v2.keras")

# Test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_data = test_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("\nModel Evaluation")
print("----------------")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)