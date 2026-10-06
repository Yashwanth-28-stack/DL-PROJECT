import os
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("emotion_model.keras")

# Image path
image_path = r"C:\Users\moldi\OneDrive\Pictures\Saved Pictures\abcd.png"

# Emotion classes
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Image could not be loaded. Check image_path.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade
cascade_path = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Face detector could not be loaded.")
    print("Cascade path:", cascade_path)
    exit()

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

if len(faces) == 0:
    print("No face detected.")

else:
    print("Face detected.")

    # Use the first detected face
    x, y, w, h = faces[0]

    # Crop face
    face = gray[y:y + h, x:x + w]

    # Resize to CNN input size
    face = cv2.resize(face, (48, 48))

    # Normalize pixel values
    face = face.astype("float32") / 255.0

    # Add batch and channel dimensions
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    # Predict emotion probabilities
    prediction = model.predict(face, verbose=0)[0]

    # Display all probabilities
    print("\nAll Emotion Probabilities")
    print("-------------------------")

    for emotion, probability in zip(class_names, prediction):
        print(f"{emotion:10}: {probability * 100:.2f}%")

    # Find highest probability
    predicted_index = np.argmax(prediction)
    predicted_emotion = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    print("\nFinal Prediction")
    print("----------------")

    if confidence < 40:
        print("Emotion   : Uncertain")
        print(f"Best match: {predicted_emotion}")
    else:
        print(f"Emotion   : {predicted_emotion}")

    print(f"Confidence: {confidence:.2f}%")