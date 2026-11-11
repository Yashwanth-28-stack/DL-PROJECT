import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "emotion_vgg.keras"

# Keep this order identical to the model's training folder order.
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

UNCERTAIN_BELOW = 40.0  # Display "Uncertain" below this confidence


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Place this script beside the model file, or update MODEL_PATH.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    cascade_path = os.path.join(
        os.path.dirname(cv2.__file__),
        "data",
        "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Could not load OpenCV's face detector:")
        print(cascade_path)
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open the webcam. Check that it is connected and not in use.")
        return

    print("Webcam is running. Press Q in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the video for a more natural webcam view.
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face.astype("float32") / 255.0
                face = face[np.newaxis, ..., np.newaxis]

                probabilities = model.predict(face, verbose=0)[0]
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index]) * 100

                if confidence < UNCERTAIN_BELOW:
                    label = f"Uncertain ({CLASS_NAMES[index]})"
                else:
                    label = f"{CLASS_NAMES[index]}: {confidence:.1f}%"

                # Draw the face box and label.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                text_y = max(y - 10, 25)
                cv2.putText(
                    frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA
                )

            if len(faces) == 0:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 220),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("Facial Expression Demo - Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()